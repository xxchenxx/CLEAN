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
                for i, label in enumerate(labels):
                    truncate_len = min(1023, len(strs[i]))
                    out = {
                        layer: t[i, 1 : truncate_len].clone()
                        for layer, t in representations.items()
                    }
                    
                    if save:
                        torch.save(out[args.repr_layer], args.temp_esm_path + label + ".pt")
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
        args.esm_model)
    esm_model = esm_model.to(device)

    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    
    #======================== initialize model =================#
    model = MoCo(args.hidden_dim, args.out_dim, device, dtype, esm_model_dim=args.esm_model_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    esm_optimizer = torch.optim.AdamW(esm_model.parameters(), lr=1e-6, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss().to(device)    
    best_loss = float('inf')

    if args.use_learnable_k:
        learnable_k = nn.Parameter(torch.zeros(1, args.esm_model_dim))
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
    start_epoch = 1


    
    if True:
        new_esm_emb = {}

        file_1 = open('./data/' + args.training_data + '.fasta')
        dict1 = SeqIO.to_dict(SeqIO.parse(file_1, "fasta"))
        original = len(list(dict1.keys()))
        
        for key_ in list(dict1.keys()):
            if os.path.exists(args.temp_esm_path + key_ + ".pt"):
                del dict1[key_]
                # print(f"{key_} founded!")
                # print(key_)
                # new_esm_emb[key_] = torch.load(args.temp_esm_path + key_ + ".pt").mean(0).detach().cpu()
        remain = len(dict1)
        print(f"Need to parse {remain}/{original}")
        with open(f'temp_{args.training_data}.fasta', 'w') as handle:
            SeqIO.write(dict1.values(), handle, 'fasta')
        
        new_esm_emb_created = generate_from_file(f'temp_{args.training_data}.fasta', alphabet, esm_model, args, start_epoch, save=True)
        new_esm_emb.update(new_esm_emb_created)
    
        # esm_emb = []
        # for ec in list(ec_id_dict.keys()):
        #     ids_for_query = list(ec_id_dict[ec])
        #     esm_to_cat = [new_esm_emb[id] for id in ids_for_query]
        #     esm_emb = esm_emb + esm_to_cat
        # esm_emb = torch.stack(esm_emb).to(device=device, dtype=dtype)

        # dist_map = get_dist_map(
        #     ec_id_dict, esm_emb, device, dtype)

if __name__ == '__main__':
    main()
