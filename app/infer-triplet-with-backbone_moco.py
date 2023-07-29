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
import json

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
        
def generate_from_file(file, alphabet, esm_model, args, start_epoch=1):
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
                for i, label in enumerate(labels):
                    out = {
                        layer: t[i, 1 : 1023].clone()
                        for layer, t in representations.items()
                    }
                    torch.save(out[args.repr_layer], args.temp_esm_path + f"/epoch{start_epoch}" + label + ".pt")
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
    parser.add_argument('--use_input_as_k', action="store_true")
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
        'batch_size': 1,
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

def main():
    seed_everything()
    ensure_dirs('./data/model')
    args = parse()
    torch.backends.cudnn.benchmark = True
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
        args.esm_model)
    esm_model = esm_model.to(device)

    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    
    #======================== initialize model =================#
    model = MoCo(args.hidden_dim, args.out_dim, device, dtype, esm_model_dim=args.esm_model_dim).cuda()

    if args.use_learnable_k:
        learnable_k = nn.Parameter(torch.zeros(1, args.esm_model_dim).cuda())
    else:
        learnable_k = None

    if args.use_extra_attention:
        query = nn.Linear(args.esm_model_dim, 64, bias=False).to(device)
        nn.init.zeros_(query.weight)
        key = nn.Linear(args.esm_model_dim, 64, bias=False).to(device)
        nn.init.zeros_(key.weight)
        if args.use_learnable_k:
            attentions_optimizer = torch.optim.Adam([{"params": query.parameters(), "lr": lr, "momentum": 0.9}, {"params": key.parameters(), "lr": lr, "momentum": 0.9}, {"params": learnable_k, "lr": lr, "momentum": 0.9}])
        else:
            attentions_optimizer = torch.optim.Adam([{"params": query.parameters(), "lr": lr, "momentum": 0.9}, {"params": key.parameters(), "lr": lr, "momentum": 0.9}])
        attentions = [query, key]
    else:
        attentions = [None, None]
        attentions_optimizer = None
    #======================== generate embed =================#
    
    seq = 'MDGVLWRVRTAALMAALLALAAWALVWASPSVEAQSNPYQRGPNPTRSALTADGPFSVATYTVSRLSVSGFGGGVIYYPTGTSLTFGGIAMSPGYTADASSLAWLGRRLASHGFVVLVINTNSRFDYPDSRASQLSAALNYLRTSSPSAVRARLDANRLAVAGHSMGGGGTLRIAEQNPSLKAAVPLTPWHTDKTFNTSVPVLIVGAEADTVAPVSQHAIPFYQNLPSTTPKVYVELDNASHFAPNSNNAAISVYTISWMKLWVDNDTRYRQFLCNVNDPALSDFRTNNRHCQ'
    esm_model.load_state_dict(torch.load("data/model/moco_esm_best_design1.pth"))
    model.load_state_dict(torch.load("data/model/moco_best_design1.pth"))
    query.load_state_dict(torch.load("data/model/moco_query_best_design1.pth"))
    key.load_state_dict(torch.load("data/model/moco_key_best_design1.pth"))

    dataset = FastaBatchedDataset.from_file(
        './data/negative_samples.fasta')
    batches = dataset.get_batch_indices(
        4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    probs = {}
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
            print(token_lens)
            temp = {
                layer: t[i, 1 : token_lens - 1].clone()
                for layer, t in representations.items()
            }
            feature = temp[args.repr_layer].cuda()
            q = query(feature)
            if learnable_k is None:
                k = key(feature)
            elif args.use_input_as_k:
                k = key(feature.mean(1, keepdim=True))
            else:
                k = key(learnable_k)
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
            probs[label] = prob.clone()
    print(probs)
    ec = pd.read_csv("data/negative_samples.csv", sep='\t')
    result = {}
    f = open("result.json", "w")
    for label in probs:
        prob_sum = torch.sum(probs[label], 0)
        important = torch.argsort(prob_sum, 0, descending=True)
        ec_number = ec.loc[ec['Entry'] == label, 'EC number'].iloc[0]
        line = {"label": label, 'EC number': ec_number, "position": str(list(important[:10].detach().cpu().numpy() + 1))}
        print(line)
        f.write(json.dumps(line)+'\n')
    torch.save(probs, 'pur_probs.pth')
if __name__ == '__main__':
    main()
