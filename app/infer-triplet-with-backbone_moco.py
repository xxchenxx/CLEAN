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
import pandas as pd

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from Bio import SeqIO
from torch.distributed.pipeline.sync import Pipe
from torch.distributed.pipeline.sync.utils import partition_model
from utils import save_state_dicts, get_logger, \
    get_attention_modules, parse_args, calculate_distance_matrix_for_ecs, calculate_cosine_distance_matrix_for_ecs
from model_utils import forward_attentions, generate_from_file
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
    args, wandb_logger = parse_args()
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
    model = MoCo_with_SMILE(args.hidden_dim, args.out_dim, device, dtype, esm_model_dim=args.esm_model_dim, use_negative_smile=args.use_negative_smile, fuse_mode=args.fuse_mode).to(device)
    #======================== generate embed =================#
    learnable_k, attentions, attentions_optimizer = get_attention_modules(args, lr, device)
    query, key = attentions

    checkpoints = torch.load("data/model/best_with_negative_smile_random_smile_bs50_CLS_attn2_cosine_ranking_remap_new_alr_1e-4_1695415684.892955_checkpoints.pth.tar", map_location='cpu')


    query.load_state_dict(checkpoints['query_state_dict'])
    key.load_state_dict(checkpoints['key_state_dict'])


    dataset = FastaBatchedDataset.from_file(
        './data/split10.fasta')
    batches = dataset.get_batch_indices(
        1024, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    probs = {}
    seqs = {}
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        # if batch_idx > 10: break
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
            temp = {
                layer: t[i, 1 : token_lens - 1].clone()
                for layer, t in representations.items()
            }
            feature = temp[args.repr_layer].cuda()
            _, prob, raw = forward_attentions(feature, query, key, learnable_k, args, return_prob=True)
            # print(prob)
            # print(raw)
            probs[label] = prob.clone().detach().cpu()
            seqs[label] = strs[i]
    # print(probs)
    ec = pd.read_csv("data/training_data_10_parsed.csv", sep=',')
    result = {}
    import matplotlib.pyplot as plt
    f = open("split10_result.json", "w")
    for label in probs:
        print(probs[label].shape)
        prob_sum = torch.sum(probs[label], 0)
        # print(prob_sum)
        # continue
        ec_number = ec.loc[ec['UniprotID'] == label, 'EC number'].iloc[0]
        pdb = ec.loc[ec['UniprotID'] == label, 'parsed_PDB'].iloc[0]
        if 'AF' in pdb:
            pdb_list = pdb.split(";")
            for pdb in pdb_list:
                print(pdb)
                values = list(map(lambda x: float(x.strip()), open(f"/mnt/vita-nas/xuxi/predicted_active_sites/{pdb}-predictions.txt", "r").readlines()))
                values = np.array(values)
                print(prob_sum.numpy())
                print(values)
                df = {"attn": list(prob_sum.numpy()), "site": list(values), "seq": list(seqs[label])}
                pd.DataFrame(df).to_csv(f"alignments/{pdb}.csv", index=False)
                """
                order = np.argsort(-values)[:min(40, int(len(values) * 0.4))]

                prob_sum = prob_sum.detach().numpy()
                mask = np.zeros_like(prob_sum)
                mask[important] = True
                prob_sum[(prob_sum > 0).astype(bool) & mask.astype(bool)] = 1
                plt.figure(figsize=(10, 10))
                plt.subplot(3, 1, 1)
                bin_mask = np.zeros_like(values)
                bin_mask[order] = 1
                print(bin_mask)
                plt.imshow(values.reshape(1, -1), aspect='auto')
                plt.subplot(3, 1, 2)
                plt.imshow(prob_sum.reshape(1, -1), aspect='auto')
                plt.subplot(3, 1, 3)
                plt.imshow(bin_mask.reshape(1, -1), aspect='auto')
                plt.savefig(f"vis/{label}.png")
                plt.close()
                """
                # assert False
        print("\n\n\n")
        # line = {"label": label, 'EC number': ec_number, "position": str(list(important[:10].detach().cpu().numpy() + 1))}
        # f.write(json.dumps(line)+'\n')
    torch.save(probs, 'split10_probs.pth')
if __name__ == '__main__':
    main()
