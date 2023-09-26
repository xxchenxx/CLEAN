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
    seq_dict.update({rec.id : rec.seq for rec in SeqIO.parse(f'./data/{args.training_data}_single_seq_ECs.fasta', "fasta")})

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
    attn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(attentions_optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1, verbose=False)
    if args.use_v:
        query, key, value = attentions
    else:
        query, key = attentions
        value = None
    # ======================== generate embed ================= #
    start_epoch = 1

    if True:
        new_esm_emb = {}

        file_1 = open('./data/' + args.training_data + '.fasta')
        file_2 = open('./data/' + args.training_data + '_single_seq_ECs.fasta')
        dict1 = SeqIO.to_dict(SeqIO.parse(file_1, "fasta"))
        dict2 = SeqIO.to_dict(SeqIO.parse(file_2, "fasta"))
        original = len(list(dict1.keys()))
        
        for key_ in list(dict1.keys()):
            if os.path.exists(args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt"):
                del dict1[key_]
                # print(f"{key_} founded!")
                new_esm_emb[key_] = torch.load(args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt").mean(0).detach().cpu()
        remain = len(dict1)
        logger.info(f"Need to parse {remain}/{original}")
        with open(f'temp_{args.training_data}.fasta', 'w') as handle:
            SeqIO.write(dict1.values(), handle, 'fasta')

        new_esm_emb_created = generate_from_file(f'temp_{args.training_data}.fasta', alphabet, esm_model, args, start_epoch)
        new_esm_emb.update(new_esm_emb_created)
        original = len(list(dict2.keys()))
        for key_ in list(dict2.keys()):
            if os.path.exists(args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt"):
                del dict2[key_]
        
        remain = len(dict2)
        logger.info(f"Need to parse {remain}/{original}")

        with open(f'temp_{args.training_data}.fasta', 'w') as handle:
            SeqIO.write(dict2.values(), handle, 'fasta')
        generate_from_file(f'temp_{args.training_data}.fasta', alphabet, esm_model, args, start_epoch)

        esm_emb = []
        for ec in list(ec_id_dict.keys()):
            ids_for_query = list(ec_id_dict[ec])
            esm_to_cat = [new_esm_emb[id] for id in ids_for_query]
            esm_emb = esm_emb + esm_to_cat
        esm_emb = torch.stack(esm_emb).to(device=device, dtype=dtype)

        dist_map = get_dist_map(
            ec_id_dict, esm_emb, device, dtype)

    logger.info(f"The number of unique EC numbers: {len(dist_map.keys())}")
    train_loader, static_embed_loader = get_dataloader(dist_map, id_ec, ec_id, args, args.temp_esm_path + f'/epoch{start_epoch}/')
    
    #======================== training =======-=================#
    # training
    for epoch in range(start_epoch, epochs + 1):
        smile_embed = None
        rhea_map = None
        
        if epoch % args.evaluate_freq == 0:
            if smile_embed is None:
                smile_embed = torch.load("Rhea_tensors.pt", map_location='cpu')
                rhea_map = pd.read_csv("rhea2ec.tsv", sep='\t')  

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
                        train_protein_emb[label] = forward_attentions(feature, query, key, learnable_k, args, value=value)
                            
            # train embedding construction
            train_esm_emb = []
            for ec in list(ec_id_dict.keys()):
                ids_for_query = list(ec_id_dict[ec])
                rhea_id = rhea_map.loc[rhea_map.ID == ec, 'RHEA_ID'].values
                smile_embed_ec = []
                protein_emb_stacked = torch.stack([train_protein_emb[id] for id in ids_for_query]).to(device) # protein embed with same EC 
                if len(rhea_id) > 0:
                    for rid in rhea_id:
                        for j in range(4):
                            try:
                                smile_embed_ec.extend(smile_embed[str(rid + j)])
                            except:
                                pass
                
                if len(smile_embed_ec) == 0:
                    smile_embed_ec = torch.stack([torch.zeros(384, device=device) for _ in ids_for_query])
                    current_train_esm_emb = model.encoder_q(model.fuser(protein_emb_stacked, smile_embed_ec))
                else:
                    if not args.use_random_augmentation:
                        smile_embed_ec = torch.cat(smile_embed_ec, 0).mean(0, keepdims=True).repeat(len(ids_for_query), 1).to(device)
                        current_train_esm_emb = model.encoder_q(model.fuser(protein_emb_stacked, smile_embed_ec))
                    else:
                        smile_embed_ec = [embed.mean(0, keepdims=True).repeat(len(ids_for_query), 1).to(device) for embed in smile_embed_ec]
                        train_esm_emb_i = [model.encoder_q(model.fuser(protein_emb_stacked, embed)) for embed in smile_embed_ec]
                        current_train_esm_emb = sum(train_esm_emb_i) / len(train_esm_emb_i)

                train_esm_emb = train_esm_emb + [current_train_esm_emb]

            if args.remap:
                train_esm_emb = torch.cat(train_esm_emb, 0).to(device=device, dtype=dtype)
                dist_map = get_dist_map(
                    ec_id_dict, train_esm_emb, device, dtype)
            train_loader, static_embed_loader = get_dataloader(dist_map, id_ec, ec_id, args, args.temp_esm_path + f'/epoch{start_epoch}/')
            cluster_center_model = get_cluster_center(
                train_esm_emb, ec_id_dict)
            # test embedding construction
            for eval_dataset in ['price', 'new', 'halogenase']:
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
                            test_protein_emb[label] = forward_attentions(feature, query, key, learnable_k, args, value=value)
                
                
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
                for ec in tqdm(list(ec_id_dict.keys())):
                    rhea_ids = rhea_map.loc[rhea_map.ID == ec, 'RHEA_ID'].values
                    smile_embeds = []

                    if len(rhea_ids) > 0:
                        for rid in rhea_ids:
                            for j in range(4):
                                try:
                                    smile_embeds.extend(smile_embed[str(rid + j)])
                                except:
                                    pass
  
                    if len(smile_embeds) == 0:
                        smile_embeds = torch.zeros((1, 384))

                    current_smile_embeds = [smile_embed_ec.unsqueeze(0).repeat(test_protein_emb.shape[0], 1, 1).cuda().mean(1) for smile_embed_ec in smile_embeds]
                    # print(current_smile_embed.shape)

                    test_protein_emb_i = [model.encoder_q(model.fuser(test_protein_emb, current_smile_embed)) for current_smile_embed in current_smile_embeds]

                    # eval_dist = dist_map_helper(ids, test_protein_emb, ecs, model_lookup)
                    
                    for i, key1 in enumerate(ids):
                        dist_norm = [(test_protein_emb_i[j][i].cuda().reshape(-1) - cluster_center_model[ec].cuda().reshape(-1)).norm(dim=0, p=2) for j in range(len(current_smile_embeds))]
                        dist_norm = torch.stack(dist_norm).mean(0).detach().cpu().numpy()
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
                                    f'eval/{eval_dataset}/AUC': roc,
                                    f'eval/{eval_dataset}/acc': acc})
                except:
                    pass


        # assert False
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader, static_embed_loader,
                           optimizer, device, dtype, criterion, 
                           esm_model, esm_optimizer, seq_dict, 
                           batch_converter, alphabet, attentions, attentions_optimizer, wandb_logger,
                           ec_number_classifier=ec_number_classifier, ec_classifier_optimizer=ec_classifier_optimizer,
                           score_matrix=score_matrix, cosine_score_matrix=cosine_score_matrix)
        scheduler.step()
        attn_scheduler.step()
        # only save the current best model near the end of training
        if (train_loss < best_loss):
            save_state_dicts(model, esm_model, query, key,
                             optimizer, esm_optimizer, attentions_optimizer,
                             args, is_best=True, output_name=model_name, save_esm=False)
            best_loss = train_loss
            logger.info(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')
        else:
            save_state_dicts(model, esm_model, query, key,
                            optimizer, esm_optimizer, attentions_optimizer,
                            args, is_best=train_loss < best_loss, output_name=model_name, save_esm=False)
        elapsed = time.time() - epoch_start_time

        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')

    wandb_logger.finish()
    # remove tmp save weights


if __name__ == '__main__':
    main()
    
