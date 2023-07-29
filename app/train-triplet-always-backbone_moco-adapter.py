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

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from Bio import SeqIO
from torch.distributed.pipeline.sync import Pipe
from torch.distributed.pipeline.sync.utils import partition_model

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
        
def generate_from_file(file, alphabet, esm_model, args, start_epoch=1, save=True):
    dataset = FastaBatchedDataset.from_file(file)
    batches = dataset.get_batch_indices(
            4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    os.makedirs(args.temp_esm_path + f"/epoch{start_epoch}", exist_ok=True)
    new_esm_emb = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                )
                toks = toks.to(device="cuda", non_blocking=True)
                out = esm_model(toks, repr_layers=[args.repr_layer], return_contacts=False)
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                batch_lens = (toks != alphabet.padding_idx).sum(1)

                for i, label in enumerate(labels):
                    token_lens = batch_lens[i]
                    out = {
                        layer: t[i, 1 : token_lens - 1].clone()
                        for layer, t in representations.items()
                    }
                    if save:
                        torch.save(out[args.repr_layer], args.temp_esm_path + f"/epoch{start_epoch}/" + label + ".pt")
                    new_esm_emb[label] = out[args.repr_layer].mean(0).cpu()
    return new_esm_emb

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--train_esm_rate', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--use_extra_attention', action="store_true")
    parser.add_argument('--use_top_k', action="store_true")
    parser.add_argument('--use_top_k_sum', action="store_true")
    parser.add_argument('--use_learnable_k', action="store_true")
    parser.add_argument('--temp_esm_path', type=str, required=True)
    parser.add_argument('--evaluate_freq', type=int, default=50)
    parser.add_argument('--esm-model', type=str, default='esm1b_t33_650M_UR50S.pt')
    parser.add_argument('--esm-model-dim', type=int, default=1280)
    parser.add_argument('--repr-layer', type=int, default=33)
    args = parser.parse_args()
    return args

from torch.nn.utils.rnn import pad_sequence

def custom_collate(data): #(2)
    anchor = [d[0] for d in data]
    positive = [d[1] for d in data]
    anchor_length = torch.tensor(list(map(len, anchor)))
    positive_length = torch.tensor(list(map(len, positive)))

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

    return anchor, positive, anchor_attn_mask, positive_attn_mask, anchor_avg_mask, positive_avg_mask

def get_dataloader(dist_map, id_ec, ec_id, args, temp_esm_path="./data/esm_data/"):
    train_params = {
        'batch_size': 2,
        'shuffle': True,
    }
    embed_params = {
        'batch_size': 100,
        'shuffle': True,
        'collate_fn': custom_collate
    }
    positive = mine_hard_positives(dist_map, 3)
    train_data = MoCo_dataset_with_mine_EC_text(id_ec, ec_id, positive)
    train_embed = MoCo_dataset_with_mine_EC(id_ec, ec_id, positive, path=temp_esm_path)
    train_loader = torch.utils.data.DataLoader(train_data, **train_params)
    static_embed_loader = torch.utils.data.DataLoader(train_embed, **embed_params)
    return train_loader, static_embed_loader

def train(model, args, epoch, train_loader, static_embed_loader,
          optimizer, device, dtype, criterion, esm_model, esm_optimizer, seq_dict, batch_converter, alphabet, attentions, attentions_optimizer=None, learnable_k=None):
    model.train()
    total_loss = 0.
    start_time = time.time()
    esm_model.train()
    query, key = attentions


    for batch, data in enumerate(train_loader):
            optimizer.zero_grad()
            esm_optimizer.zero_grad()
            if attentions_optimizer is not None:
                attentions_optimizer.zero_grad()
            anchor, positive = data
            anchor = [('', seq_dict[a]) for a in anchor]
            batch_labels, batch_strs, batch_tokens = batch_converter(anchor)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            results = esm_model(batch_tokens.to(device), repr_layers=[args.repr_layer], return_contacts=True)

            token_representations = results["representations"][args.repr_layer]
            anchor = []
            
            for i, tokens_len in enumerate(batch_lens):
                if query is None:
                    anchor.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                else:
                    q = query(token_representations[i, 1 : tokens_len - 1]) # N x 64
                    if learnable_k is None:
                        k = key(token_representations[i, 1 : tokens_len - 1]) # 1 x 64
                    elif args.use_input_as_k:
                        k = key(token_representations[i, 1 : tokens_len - 1].cuda().mean(keepdim=True))
                    else:
                        k = key(learnable_k) # 1 x 64
                    if args.use_top_k:
                        raw = torch.einsum('jk,lk->jl', k, q) / np.sqrt(64)
                        
                        shape = raw.shape
                        raw = raw.reshape(-1, raw.shape[-1])
                        _, smallest_value = torch.topk(raw, max(0, raw.shape[1] - 100), largest=False)
                        smallest = torch.zeros_like(raw)
                        for j in range(len(raw)):
                            smallest[j, smallest_value[j]] = 1
                        smallest = smallest.reshape(shape).bool()
                        raw = raw.reshape(shape)
                        raw[smallest] = raw[smallest] + float('-inf')
                        prob = torch.softmax(raw, -1) # N x 1
                    elif args.use_top_k_sum:
                        raw = torch.einsum('jk,lk->jl', k, q) / np.sqrt(64)
                        weights = raw.sum(-2, keepdim=True)
                        prob = torch.softmax(weights, -1)
                    else:
                        prob = torch.softmax(torch.einsum('jk,lk->jl', k, q) / np.sqrt(64), 1) # N x 1
                    anchor.append((prob @ token_representations[i, 1 : tokens_len - 1]).sum(0))
            anchor = torch.stack(anchor)
            positive = [('', seq_dict[a]) for a in positive]
            batch_labels, batch_strs, batch_tokens = batch_converter(positive)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            results = esm_model(batch_tokens.to(device), repr_layers=[args.repr_layer], return_contacts=True)
        
            token_representations = results["representations"][args.repr_layer].cuda()
            positive = []
            for i, tokens_len in enumerate(batch_lens):
                if query is None:
                    positive.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                else:
                    q = query(token_representations[i, 1 : tokens_len - 1]) # N x 64
                    if learnable_k is None:
                        k = key(token_representations[i, 1 : tokens_len - 1]) # 1 x 64
                    elif args.use_input_as_k:
                        k = key(token_representations[i, 1 : tokens_len - 1].cuda().mean(keepdim=True))
                    else:
                        k = key(learnable_k) # 1 x 64
                    if args.use_top_k:
                        raw = torch.einsum('jk,lk->jl', k, q) / np.sqrt(64)
                        
                        shape = raw.shape
                        raw = raw.reshape(-1, raw.shape[-1])
                        _, smallest_value = torch.topk(raw, max(0, raw.shape[1] - 100), largest=False)
                        smallest = torch.zeros_like(raw)
                        for j in range(len(raw)):
                            smallest[j, smallest_value[j]] = 1
                        smallest = smallest.reshape(shape).bool()
                        raw = raw.reshape(shape)
                        raw[smallest] = raw[smallest] + float('-inf')
                        prob = torch.softmax(raw, -1) # N x 1
                    elif args.use_top_k_sum:
                        raw = torch.einsum('jk,lk->jl', k, q) / np.sqrt(64)
                        weights = raw.sum(-2, keepdim=True)
                        prob = torch.softmax(weights, -1)
                    else:
                        prob = torch.softmax(torch.einsum('jk,lk->jl', k, q) / np.sqrt(64), 1) # N x 1
                    positive.append((prob @ token_representations[i, 1 : tokens_len - 1]).sum(0))
            positive = torch.stack(positive)
            anchor_out, positive_out = model(anchor.to(device=device, dtype=dtype), positive.to(device=device, dtype=dtype))
            loss = criterion(anchor_out, positive_out)
            loss.backward()
            optimizer.step()
            esm_optimizer.step()
            if attentions_optimizer is not None:
                attentions_optimizer.step()
            total_loss += loss.item()
            if args.verbose:
                lr = args.learning_rate
                ms_per_batch = (time.time() - start_time) * 1000
                cur_loss = loss.item()
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                    f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                    f'loss {cur_loss:5.2f}')
                start_time = time.time()
        # record running average training loss
    return total_loss/(batch + 1)
def main():
    seed_everything()
    ensure_dirs('./data/model')
    args = parse()
    torch.backends.cudnn.benchmark = True
    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.training_data + '.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    seq_dict = {rec.id : rec.seq for rec in SeqIO.parse(f'./data/{args.training_data}.fasta', "fasta")}
    seq_dict.update({rec.id : rec.seq for rec in SeqIO.parse(f'./data/{args.training_data}_single_seq_ECs.fasta', "fasta")})


    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)

    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert num_devices > 0
    

    esm_model, alphabet = pretrained.load_model_and_alphabet(
        args.esm_model, use_adapter=True, adapter_rank=16, use_lora=False, adapter_rank=16)
    esm_model = esm_model.to(device)

    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    
    #======================== initialize model =================#
    model = MoCo(args.hidden_dim, args.out_dim, device, dtype, esm_model_dim=args.esm_model_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    parameters = []
    for name, p in esm_model.named_parameters():
        if 'adapter' in name:
            p.requires_grad = True
            parameters.append(p)
        else:
            p.requires_grad = False
    esm_optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss().to(device)    
    best_loss = float('inf')

    if args.use_learnable_k:
        learnable_k = nn.Parameter(torch.zeros(1, args.esm_model_dim))
    else:
        learnable_k = None

    if args.use_extra_attention:
        query = nn.Linear(args.esm_model_dim, 64, bias=False).to(device)
        # nn.init.constant_(query.weight, 1e-3)
        nn.init.normal_(query.weight, std=np.sqrt(2 / (64 + args.esm_model_dim)))
        key = nn.Linear(args.esm_model_dim, 64, bias=False).to(device)
        # nn.init.constant_(key.weight, 1e-3)
        nn.init.normal_(key.weight, std=np.sqrt(2 / (64 + args.esm_model_dim)))
        if args.use_learnable_k:
            attentions_optimizer = torch.optim.Adam([{"params": query.parameters(), "lr": lr, "momentum": 0.9}, {"params": key.parameters(), "lr": lr, "momentum": 0.9}, {"params": learnable_k, "lr": lr, "momentum": 0.9}])
        else:
            attentions_optimizer = torch.optim.Adam([{"params": query.parameters(), "lr": lr, "momentum": 0.9}, {"params": key.parameters(), "lr": lr, "momentum": 0.9}])
        attentions = [query, key]
    else:
        attentions = [None, None]
        attentions_optimizer = None
    #======================== generate embed =================#
    start_epoch = 1

    if True:
        new_esm_emb = {}

        file_1 = open('./data/' + args.training_data + '.fasta')
        dict1 = SeqIO.to_dict(SeqIO.parse(file_1, "fasta"))
        original = len(list(dict1.keys()))
        
        for key_ in list(dict1.keys()):
            if os.path.exists(args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt"):
                del dict1[key_]
                # print(f"{key_} founded!")
                new_esm_emb[key_] = torch.load(args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt").mean(0).detach().cpu()
        remain = len(dict1)
        print(f"Need to parse {remain}/{original}")
        with open(f'temp_{args.training_data}.fasta', 'w') as handle:
            SeqIO.write(dict1.values(), handle, 'fasta')
        
        new_esm_emb_created = generate_from_file(f'temp_{args.training_data}.fasta', alphabet, esm_model, args, start_epoch, save=False)
        new_esm_emb.update(new_esm_emb_created)
    
        esm_emb = []
        for ec in list(ec_id_dict.keys()):
            ids_for_query = list(ec_id_dict[ec])
            esm_to_cat = [new_esm_emb[id] for id in ids_for_query]
            esm_emb = esm_emb + esm_to_cat
        esm_emb = torch.stack(esm_emb).to(device=device, dtype=dtype)

        dist_map = get_dist_map(
            ec_id_dict, esm_emb, device, dtype)

    print("The number of unique EC numbers: ", len(dist_map.keys()))
    train_loader, static_embed_loader = get_dataloader(dist_map, id_ec, ec_id, args, args.temp_esm_path + f'/epoch{start_epoch}/')
    
    #======================== training =======-=================#
    # training
    for epoch in range(start_epoch, epochs + 1):
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999))
            # save updated model
            torch.save(model.state_dict(), './data/model/' +
                       model_name + '_' + str(epoch) + '.pth')
            
            torch.save(esm_model.state_dict(), './data/model/esm_' +
                       model_name + '_' + str(epoch) + '.pth')

            
        if epoch % args.evaluate_freq == 0:
            dataset = FastaBatchedDataset.from_file(
                    './data/' + args.training_data + '.fasta')
            batches = dataset.get_batch_indices(
                4096, extra_toks_per_seq=1)
            data_loader = torch.utils.data.DataLoader(
                dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
            )
            with torch.no_grad():
                for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                    print(
                        f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                    )
                    toks = toks.to(device="cuda", non_blocking=True)
                    out = esm_model(toks, repr_layers=[args.repr_layer], return_contacts=False)
                    representations = {
                        layer: t.to(device="cpu") for layer, t in out["representations"].items()
                    }
                    batch_lens = (toks != alphabet.padding_idx).sum(1)

                    for i, label in enumerate(labels):
                        tokens_len = batch_lens[i]
                        out = {
                            layer: t[i, 1 : tokens_len - 1].clone()
                            for layer, t in representations.items()
                        }
                        if not args.use_extra_attention:
                            new_esm_emb[label] = out[args.repr_layer].mean(0)
                        else:
                            q = query(out[args.repr_layer].cuda())
                            if learnable_k is None:
                                k = key(out[args.repr_layer].cuda())
                            else:
                                k = key(learnable_k)
                            if args.use_top_k:
                                raw = k @ q.transpose(0, 1) / np.sqrt(64)
                                
                                shape = raw.shape
                                raw = raw.reshape(-1, raw.shape[-1])
                                _, smallest_value = torch.topk(raw, max(0, raw.shape[1] - 100), largest=False)
                                smallest = torch.zeros_like(raw)
                                for i in range(len(raw)):
                                    smallest[i, smallest_value[i]] = 1
                                smallest = smallest.reshape(shape).bool()
                                raw = raw.reshape(shape)
                                raw[smallest] = raw[smallest] + float('-inf')
                                prob = torch.softmax(raw, -1) # N x 1
                            elif args.use_top_k_sum:
                                raw = k @ q.transpose(0, 1) / np.sqrt(64)
                                weights = raw.sum(-2, keepdim=True)
                                prob = torch.softmax(weights, -1)
                            else:
                                prob = torch.softmax(k @ q.transpose(0, 1) / np.sqrt(64), 1) # N x 1
                            new_esm_emb[label] = (prob @ out[args.repr_layer].cuda()).sum(0)
            train_esm_emb = []
                # for ec in tqdm(list(ec_id_dict.keys())):
            for ec in list(ec_id_dict.keys()):
                ids_for_query = list(ec_id_dict[ec])
                esm_to_cat = [new_esm_emb[id] for id in ids_for_query]
                train_esm_emb = train_esm_emb + esm_to_cat
            train_esm_emb = torch.stack(train_esm_emb).to(device=device, dtype=dtype)

            dataset = FastaBatchedDataset.from_file(
                './data/price.fasta')
            batches = dataset.get_batch_indices(
                4096, extra_toks_per_seq=1)
            data_loader = torch.utils.data.DataLoader(
                dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
            )
            test_emb = {}
            with torch.no_grad():
                for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                    print(
                        f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                    )
                    toks = toks.to(device="cuda", non_blocking=True)
                    out = esm_model(toks, repr_layers=[args.repr_layer], return_contacts=False)
                    representations = {
                        layer: t.to(device="cpu") for layer, t in out["representations"].items()
                    }
                    batch_lens = (toks != alphabet.padding_idx).sum(1)

                    for i, label in enumerate(labels):
                        token_lens = batch_lens[i]
                        out = {
                            layer: t[i, 1 : token_lens - 1].clone()
                            for layer, t in representations.items()
                        }
                        if not args.use_extra_attention:
                            test_emb[label] = out[args.repr_layer].cuda().mean(0)
                        else:
                            q = query(out[args.repr_layer].cuda())
                            if learnable_k is None:
                                k = key(out[args.repr_layer].cuda())
                            else:
                                k = key(learnable_k)
                            if args.use_top_k:
                                raw = k @ q.transpose(0, 1) / np.sqrt(64)
                                
                                shape = raw.shape
                                raw = raw.reshape(-1, raw.shape[-1])
                                _, smallest_value = torch.topk(raw, max(0, raw.shape[1] - 100), largest=False)
                                smallest = torch.zeros_like(raw)
                                for i in range(len(raw)):
                                    smallest[i, smallest_value[i]] = 1
                                smallest = smallest.reshape(shape).bool()
                                raw = raw.reshape(shape)
                                raw[smallest] = raw[smallest] + float('-inf')
                                prob = torch.softmax(raw, -1) # N x 1
                            elif args.use_top_k_sum:
                                raw = k @ q.transpose(0, 1) / np.sqrt(64)
                                weights = raw.sum(-2, keepdim=True)
                                prob = torch.softmax(weights, -1)
                            else:
                                prob = torch.softmax(k @ q.transpose(0, 1) / np.sqrt(64), 1) # N x 1
                            test_emb[label] = (prob @ out[args.repr_layer].cuda()).sum(0)
            test_esm_emb = []
            id_ec_test, ec_id_dict_test = get_ec_id_dict('./data/price.csv')
            ids_for_query = list(id_ec_test.keys())
            esm_to_cat = [test_emb[id] for id in ids_for_query]
                
            test_esm_emb = torch.stack(esm_to_cat)
            train_esm_emb = model.encoder_q(train_esm_emb)
            test_esm_emb = model.encoder_q(test_esm_emb)
            eval_dist = get_dist_map_test(train_esm_emb, test_esm_emb, ec_id_dict, id_ec_test, device, dtype)
            eval_df = pd.DataFrame.from_dict(eval_dist)
            out_filename = "results/price_temp" 
            write_max_sep_choices(eval_df, out_filename, gmm=None)
            
            if True:
                pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
                pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
                true_label, all_label = get_true_labels('./data/price')
                pre, rec, f1, roc, acc = get_eval_metrics(
                    pred_label, pred_probs, true_label, all_label)
                print("############ EC calling results using maximum separation ############")
                print('-' * 75)
                print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
                    f'>>> precision: {pre:.3} | recall: {rec:.3}'
                    f'| F1: {f1:.3} | AUC: {roc:.3} ')
                print('-' * 75)

        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader, static_embed_loader,
                           optimizer, device, dtype, criterion, 
                           esm_model, esm_optimizer, seq_dict, 
                           batch_converter, alphabet, attentions, attentions_optimizer)
        scheduler.step()
        # only save the current best model near the end of training
        if (train_loss < best_loss):
            torch.save(model.state_dict(), './data/model/moco_best_' + model_name + '.pth')
            torch.save(esm_model.state_dict(), './data/model/moco_esm_best_' + model_name + '.pth')
            if args.use_extra_attention:
                torch.save(query.state_dict(), './data/model/moco_query_best_' + model_name + '.pth')
                torch.save(key.state_dict(), './data/model/moco_key_best_' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')
        elif epoch % 100 == 0:
            torch.save(model.state_dict(), './data/model/moco_' + model_name + '.pth')
            torch.save(esm_model.state_dict(), './data/model/moco_esm_' + model_name + '.pth')
            if args.use_extra_attention:
                torch.save(query.state_dict(), './data/model/moco_query_' + model_name + '.pth')
                torch.save(key.state_dict(), './data/model/moco_key_' + model_name + '.pth')
        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)
    # remove tmp save weights
    os.remove('./data/model/moco_' + model_name + '.pth')
    os.remove('./data/model/moco_' + model_name + '_' + str(epoch) + '.pth')

    os.remove('./data/model/moco_esm_' + model_name + '.pth')
    os.remove('./data/model/moco_esm_' + model_name + '_' + str(epoch) + '.pth')

    if args.use_extra_attention:
        os.remove('./data/model/moco_query_' + model_name + '.pth')
        os.remove('./data/model/moco_query_' + model_name + '_' + str(epoch) + '.pth')

        os.remove('./data/model/moco_key_' + model_name + '.pth')
        os.remove('./data/model/moco_key_' + model_name + '_' + str(epoch) + '.pth')
    # save final weights
    torch.save(model.state_dict(), './data/model/moco_' + model_name + '.pth')
    torch.save(esm_model.state_dict(), './data/model/moco_esm_' + model_name + '.pth')
    if args.use_extra_attention:
        torch.save(query.state_dict(), './data/model/moco_query_' + model_name + '.pth')
        torch.save(key.state_dict(), './data/model/moco_key_' + model_name + '.pth')


if __name__ == '__main__':
    main()
