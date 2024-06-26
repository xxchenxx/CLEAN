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

def train(model, args, epoch, train_loader, static_embed_loader,
          optimizer, device, dtype, criterion, esm_model, esm_optimizer, seq_dict, batch_converter, alphabet, attentions, attentions_optimizer, wandb_logger, learnable_k=None, ec_number_classifier=None, ec_classifier_optimizer=None, score_matrix=None, cosine_score_matrix=None, print_freq=1) :
    model.train()
    total_loss = 0.
    start_time = time.time()
    esm_model.train()
    if args.use_v:
        query, key, value = attentions
    else:
        query, key = attentions
        value = None

    for batch, data in enumerate(static_embed_loader):
        optimizer.zero_grad()
        anchor_original, positive_original, anchor_attn_mask, positive_attn_mask, anchor_avg_mask, positive_avg_mask, ec_numbers, smile, negative_smile = data
        smile = smile.cuda()
        negative_smile = negative_smile.cuda()
        anchor_original = anchor_original.cuda()
        positive_original = positive_original.cuda()
        anchor_attn_mask = anchor_attn_mask.cuda()
        positive_attn_mask = positive_attn_mask.cuda()
        anchor_avg_mask = anchor_avg_mask.cuda()
        positive_avg_mask = positive_avg_mask.cuda()
        ec_numbers = ec_numbers.cuda()
        anchor = []
        positive = []
        if query is None:
            positive = torch.sum(positive_original * positive_avg_mask.unsqueeze(-1), 1)
            anchor = torch.sum(anchor_original * anchor_avg_mask.unsqueeze(-1), 1)
        else:
            positive = forward_attentions(positive_original, query, key, learnable_k, args, avg_mask=positive_avg_mask,
                                          attn_mask=positive_attn_mask, value=value)
            
            anchor = forward_attentions(anchor_original, query, key, learnable_k, args, avg_mask=anchor_avg_mask,
                                          attn_mask=anchor_attn_mask, value=value)
        if args.use_weighted_loss:
            output, target, aux_loss, metrics, buffer_ec, q = model(anchor.to(device=device, dtype=dtype), positive.to(device=device, dtype=dtype), smile, negative_smile, ec_numbers)
            loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
            predicted = torch.argmax(output, 1)
            weights = []
            for i in range(len(predicted)):
                if predicted[i] == 0:
                    weights.append(1)
                else:
                    if buffer_ec[predicted[i] - 1] is None:
                        weights.append(1)
                    else:
                        weights.append(score_matrix[buffer_ec[predicted[i] - 1]][ec_numbers[i]])
            weights = torch.tensor(weights).cuda()
            loss = (loss * weights).mean()
        elif args.use_ranking_loss:
            output, target, aux_loss, metrics, q = model(anchor.to(device=device, dtype=dtype), positive.to(device=device, dtype=dtype), smile, negative_smile)
            loss = criterion(output, target)
            distances = torch.cdist(q.to(device=device, dtype=dtype), q.to(device=device, dtype=dtype))
            distance_values = distances.detach().cpu().numpy()
            metrics['distance_values'] = wandb.Histogram(distance_values)
            label_distances = torch.zeros_like(distances)
            for i in range(len(output)):
                for j in range(len(output)):
                    label_distances[i, j] = score_matrix[ec_numbers[i], ec_numbers[j]]
            m = torch.clamp(20 - label_distances * distances, min=0)
            m = torch.triu(m, diagonal=1)
            loss_distance = torch.mean(m)
            loss += args.distance_loss_coef * loss_distance / output.shape[0]
            metrics['distance_loss'] = args.distance_loss_coef * loss_distance
        elif args.use_cosine_ranking_loss:
            output, target, aux_loss, metrics, q = model(anchor.to(device=device, dtype=dtype), positive.to(device=device, dtype=dtype), smile, negative_smile)
            loss = criterion(output, target)
            distances = q @ q.transpose(0, 1)
            distance_values = distances.detach().cpu().numpy()
            metrics['distance_values'] = wandb.Histogram(distance_values)
            label_distances = torch.zeros_like(distances)
            for i in range(len(output)):
                for j in range(len(output)):
                    label_distances[i, j] = cosine_score_matrix[ec_numbers[i], ec_numbers[j]]
            m = torch.clamp(label_distances * distances - 1, min=0)
            loss_distance = torch.mean(m)
            loss += args.distance_loss_coef * loss_distance / (output.shape[0] ** 2) 
            metrics['distance_loss'] = args.distance_loss_coef * loss_distance
        else:
            output, target, aux_loss, metrics, q = model(anchor.to(device=device, dtype=dtype), positive.to(device=device, dtype=dtype), smile, negative_smile)
            loss = criterion(output, target)
        metrics['loss'] = loss.item()
        loss = loss + aux_loss

        loss.backward()
        optimizer.step()
        if attentions_optimizer is not None:
            attentions_optimizer.step()
        total_loss += loss.item()
        wandb_logger.log({'train/' + key: metrics[key] for key in metrics})
        if batch % print_freq == 0:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            cur_loss = loss.item()
            log_string =  f'| epoch {epoch:3d} | {batch:5d}/{len(static_embed_loader):5d} batches | ' + \
                f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | ' + \
                f'loss {cur_loss:5.2f} | aux loss {aux_loss:5.2f} | '
                
            for metric_key in metrics:
                log_string = log_string + f"{metric_key} {metrics[metric_key]} | "
            logger.info(log_string)
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1)

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
    model = MoCo_with_SMILE(args.hidden_dim, args.out_dim, device, dtype, esm_model_dim=args.esm_model_dim, use_negative_smile=args.use_negative_smile, fuse_mode=args.fuse_mode, queue_size=args.queue_size).to(device)
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
    index_set = {}
    prob_set = {}
    if True:
            prediction = {}
            for eval_dataset in ['new']:# , 'new', 'halogenase']:
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
                            f, prob, raw = forward_attentions(feature, query, key, learnable_k, args, value=value, return_prob=True)
                            index = torch.argmax(raw, -1).detach().cpu().numpy()
                            
                            print(set(index))
                            # assert False

                            index_set[label] = set(index)
                        prob_set[label] = prob.mean(0).detach().cpu().numpy()

    torch.save(index_set, "index_set.pth.tar")
    torch.save(prob_set, "prob_set.pth.tar")
    wandb_logger.finish()
    # remove tmp save weights


if __name__ == '__main__':
    main()
    
