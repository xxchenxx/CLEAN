import torch
import time
import os
import pickle
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
from CLEAN.distance_map import *
from CLEAN.evaluate import *
import pandas as pd
import torch.nn as nn
import argparse
from CLEAN.distance_map import get_dist_map
import torch
import numpy as np
import wandb
from functools import partial

from esm import FastaBatchedDataset, pretrained
from Bio import SeqIO

from utils import save_state_dicts, get_logger, \
    get_attention_modules, parse_args, calculate_distance_matrix_for_ecs, calculate_cosine_distance_matrix_for_ecs

from model_utils import forward_attentions, generate_from_file
import wandb
from torch.nn.utils.rnn import pad_sequence
logger = get_logger(__name__)

def mine_hard_positives(dist_map, knn=10):
    #print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    positives = {}
    for i, target in enumerate(ecs):
        sort_orders = sorted(
            dist_map[target].items(), key=lambda x: x[1], reverse=True)
        pos_ecs = [i[0] for i in sort_orders[len(sort_orders) - knn:]]
        positives[target] = {
            'positive': pos_ecs
        }
    return positives
        
def custom_collate(data, ec_list): #(2)
    anchor = [d[0] for d in data]
    positive = [d[1] for d in data]
    ec_numbers = [ec_list.index(d[4]) for d in data]
    smile = [d[2] for d in data]
    negative_smile = [d[3] for d in data]
    anchor_length = torch.tensor(list(map(len, anchor)))
    positive_length = torch.tensor(list(map(len, positive)))
    smile = torch.stack(smile, 0)
    negative_smile = torch.stack(negative_smile, 0)
    anchor = pad_sequence(anchor, batch_first=True)
    positive = pad_sequence(positive, batch_first=True)
    anchor_attn_mask = torch.ones(anchor.shape[0], anchor.shape[1], anchor.shape[1]) * -np.inf
    positive_attn_mask = torch.ones(positive.shape[0], positive.shape[1], positive.shape[1]) * -np.inf
    
    for i in range(anchor_attn_mask.shape[0]):
        anchor_attn_mask[i, :, :anchor_length[i]] = 0
        positive_attn_mask[i, :, :positive_length[i]] = 0
        
    anchor_avg_mask = torch.zeros(anchor.shape[0], anchor.shape[1])
    positive_avg_mask = torch.zeros(positive.shape[0], positive.shape[1])
    for i in range(anchor_avg_mask.shape[0]):
        anchor_avg_mask[i, :anchor_length[i]] = 1 / anchor_length[i]
        positive_avg_mask[i, :positive_length[i]] = 1 / positive_length[i]

    return anchor, positive, anchor_attn_mask, positive_attn_mask, anchor_avg_mask, positive_avg_mask, torch.tensor(ec_numbers), smile, negative_smile

def get_dataloader(dist_map, id_ec, ec_id, args, temp_esm_path="./data/esm_data/"):
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 4,
    }
    ec_list = list(ec_id.keys())
    embed_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'collate_fn': partial(custom_collate, ec_list=ec_list),
        'num_workers': 4,
    }
    positive = mine_hard_positives(dist_map, 3)
    train_data = MoCo_dataset_with_mine_EC_and_SMILE(id_ec, ec_id, positive, with_ec_number=True, use_random_augmentation=args.use_random_augmentation, return_name=True, use_SMILE_cls_token=args.use_SMILE_cls_token)
    train_embed = MoCo_dataset_with_mine_EC_and_SMILE(id_ec, ec_id, positive, path=temp_esm_path, with_ec_number=True, use_random_augmentation=args.use_random_augmentation, use_SMILE_cls_token=args.use_SMILE_cls_token)
    train_loader = torch.utils.data.DataLoader(train_data, **train_params)
    static_embed_loader = torch.utils.data.DataLoader(train_embed, **embed_params)
    return train_loader, static_embed_loader

def main():
    ensure_dirs('./data/model')
    args, wandb_logger = parse_args()
    if args.seed is not None:
        seed_everything(args.seed)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.training_data + '.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    score_matrix = calculate_distance_matrix_for_ecs(ec_id_dict)
    cosine_score_matrix = calculate_cosine_distance_matrix_for_ecs(ec_id_dict)
    seq_dict = {rec.id : rec.seq for rec in SeqIO.parse(f'./data/{args.training_data}.fasta', "fasta")}
    # seq_dict.update({rec.id : rec.seq for rec in SeqIO.parse(f'./data/{args.training_data}_single_seq_ECs.fasta', "fasta")})

    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    logger.info(args)

    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert num_devices > 0

    #======================== initialize esm model =================#
    esm_model, alphabet = pretrained.load_model_and_alphabet(
        args.esm_model)
    esm_model = esm_model.to(device)

    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    
    #======================== initialize model =================#
    model = MoCo(args.hidden_dim, args.out_dim, device, dtype, esm_model_dim=args.esm_model_dim, use_negative_smile=args.use_negative_smile, fuse_mode=args.fuse_mode, queue_size=args.queue_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1, verbose=False)
    esm_optimizer = torch.optim.AdamW(esm_model.parameters(), lr=1e-6, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss().to(device)
    ec_number_classifier = nn.Sequential(nn.Linear(args.esm_model_dim, 128), nn.ReLU(), nn.Linear(128, len(ec_id.keys()))).cuda()
    ec_classifier_optimizer = torch.optim.Adam(ec_number_classifier.parameters(), 0.001)
    best_loss = float('inf')

    learnable_k, attentions, attentions_optimizer = get_attention_modules(args, lr, device)
    if args.use_v:
        query, key, value = attentions
    else:
        query, key = attentions
        value = None
    # ======================== generate embed ================= #
    start_epoch = 1



    checkpoints = torch.load(args.checkpoint, map_location='cpu')


    query.load_state_dict(checkpoints['query_state_dict'])
    key.load_state_dict(checkpoints['key_state_dict'])
    model.load_state_dict(checkpoints['model_state_dict'])

    
    #======================== training =======-=================#
    # training
    smile_embed = None
    if True:
            train_protein_emb = {}
            dataset = FastaBatchedDataset.from_file(
                './data/' + args.training_data + '.fasta')
            batches = dataset.get_batch_indices(
                4096, extra_toks_per_seq=1)
            data_loader = torch.utils.data.DataLoader(
                dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
            )

            with torch.no_grad():
                for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                    batch_lens = (toks != alphabet.padding_idx).sum(1)
                    if batch_idx % 100 == 0:
                        logger.info(
                            f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                        )
                    toks = toks.to(device="cuda", non_blocking=True)
                    out = esm_model(toks, repr_layers=[args.repr_layer], return_contacts=False)
                    representations = {
                        layer: t.to(device="cpu") for layer, t in out["representations"].items()
                    }

                    for i, label in enumerate(labels):
                        tokens_len = batch_lens[i]
                        temp = {
                            layer: t[i, 1 : tokens_len - 1].clone()
                            for layer, t in representations.items()
                        }
                        feature = temp[args.repr_layer].cuda()                        
                        train_protein_emb[label] = forward_attentions(feature, query, key, learnable_k, args)
                            
            # train embedding construction
            train_esm_emb = []
            for ec in list(ec_id_dict.keys()):
                ids_for_query = list(ec_id_dict[ec])
                protein_emb_stacked = torch.stack([train_protein_emb[id] for id in ids_for_query]).to(device) # protein embed with same EC 
                current_train_esm_emb = model.encoder_q(protein_emb_stacked)
                train_esm_emb = train_esm_emb + [current_train_esm_emb]

            train_esm_emb = torch.cat(train_esm_emb, 0).to(device=device, dtype=dtype)
            cluster_center_model = get_cluster_center(
                train_esm_emb, ec_id_dict)
            # test embedding construction
            for eval_dataset in ['price', 'new', 'halogenase']:
            # eval_dataset = 'price'
                dataset = FastaBatchedDataset.from_file(
                    f'./data/{eval_dataset}.fasta')
                batches = dataset.get_batch_indices(
                    4096, extra_toks_per_seq=1)
                data_loader = torch.utils.data.DataLoader(
                    dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
                )

                test_protein_emb = {}
                with torch.no_grad():
                    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                        if batch_idx % 100 == 0:
                            logger.info(
                                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                            )
                        toks = toks.to(device="cuda", non_blocking=True)
                        out = esm_model(toks, repr_layers=[args.repr_layer], return_contacts=False)
                        representations = {
                            layer: t.to(device="cpu") for layer, t in out["representations"].items()
                        }
                        batch_lens = (toks != alphabet.padding_idx).sum(1)
                        
                        for i, label in enumerate(labels):
                            temp = {
                                layer: t[i, 1 : batch_lens[i] - 1].clone()
                                for layer, t in representations.items()
                            }
                            feature = temp[args.repr_layer].cuda()
                            test_protein_emb[label] = forward_attentions(feature, query, key, learnable_k, args)
                
                
                total_ec_n, out_dim = len(ec_id_dict.keys()), train_esm_emb.size(1)
                model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
                ecs = list(cluster_center_model.keys())
                id_ec_test, ec_id_dict_test = get_ec_id_dict(f'./data/{eval_dataset}.csv')
                ids_for_query = list(id_ec_test.keys())
                test_protein_emb_to_cat = [test_protein_emb[id] for id in ids_for_query]
                test_protein_emb = torch.stack(test_protein_emb_to_cat)

                for i, ec in enumerate(ecs):
                    model_lookup[i] = cluster_center_model[ec]
                
                ids = list(id_ec_test.keys())
                dist = {}
                test_protein_emb = model.encoder_q(test_protein_emb)
                
                for ec in list(ec_id_dict.keys()):
                    for i, key1 in enumerate(ids):
                        dist_norm = (test_protein_emb[i].cuda().reshape(-1) - cluster_center_model[ec].cuda().reshape(-1)).norm(dim=0, p=2)
                        dist_norm = dist_norm.detach().cpu().numpy()
                        # print(dist_norm)
                        if key1 not in dist:
                            dist[key1] = {}
                        dist[key1][ec] = dist_norm
                    
                eval_dist = dist
                eval_df = pd.DataFrame.from_dict(eval_dist)
                eval_df = eval_df.astype(float)
                out_filename = f"results/{eval_dataset}_{args.model_name}" 
                write_max_sep_choices(eval_df, out_filename, gmm=None)
                
                pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
                pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
                true_label, all_label = get_true_labels(f'./data/{eval_dataset}')
                try:
                    pre, rec, f1, roc, acc = get_eval_metrics(
                        pred_label, pred_probs, true_label, all_label)

                    logger.info(f'total samples: {len(true_label)} | total ec: {len(all_label)}')
                    logger.info(f'precision: {pre:.3} | recall: {rec:.3} | F1: {f1:.3} | AUC: {roc:.3} ')
                    wandb_logger.log({f'eval/{eval_dataset}/precision': pre, 
                                    f'eval/{eval_dataset}/recall': rec,
                                    f'eval/{eval_dataset}/F1': f1,
                                    f'eval/{eval_dataset}/AUC': roc})
                except:
                    pass
                
    wandb_logger.finish()
    # remove tmp save weights


if __name__ == '__main__':
    main()
    
