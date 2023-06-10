import torch
import time
import os
import pickle
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
import torch.nn as nn
import argparse
from CLEAN.distance_map import get_dist_map
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from Bio import SeqIO

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--use_extra_attention', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_dataloader(dist_map, id_ec, ec_id, args):
    params = {
        'batch_size': 1,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 30)
    train_data = Triplet_dataset_with_mine_EC_text(id_ec, ec_id, negative)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader


def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion, esm_model, esm_optimizer, seq_dict, batch_converter, alphabet, attentions, attentions_optimizer=None):
    model.train()
    total_loss = 0.
    start_time = time.time()
    esm_model.train()
    if len(attentions) > 0:
        query, key = attentions
    else:
        query = None
    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        esm_optimizer.zero_grad()
        if attentions_optimizer is not None:
            attentions_optimizer.zero_grad()
        anchor, positive, negative = data
        anchor = [('', seq_dict[a]) for a in anchor]
        batch_labels, batch_strs, batch_tokens = batch_converter(anchor)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        results = esm_model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        anchor = []
        
        for i, tokens_len in enumerate(batch_lens):
            if query is None:
                anchor.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            else:
                mean = token_representations[i, 1 : tokens_len - 1].mean(0).unsqueeze(0)
                q = query(token_representations[i, 1 : tokens_len - 1]) # N x 64
                k = key(mean) # 1 x 64
                prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64)) # N x 1
                anchor.append(prob.transpose(0, 1) @ token_representations[i, 1 : tokens_len - 1])
        anchor = torch.stack(anchor)

        positive = [('', seq_dict[a]) for a in positive]
        batch_labels, batch_strs, batch_tokens = batch_converter(positive)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        results = esm_model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].cuda()
        positive = []
        for i, tokens_len in enumerate(batch_lens):
            if query is None:
                positive.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            else:
                mean = token_representations[i, 1 : tokens_len - 1].mean(0).unsqueeze(0)
                q = query(token_representations[i, 1 : tokens_len - 1]) # N x 64
                k = key(mean) # 1 x 64
                prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64)) # N x 1
                positive.append(prob.transpose(0, 1) @ token_representations[i, 1 : tokens_len - 1])
        positive = torch.stack(positive)

        negative = [('', seq_dict[a]) for a in negative]
        batch_labels, batch_strs, batch_tokens = batch_converter(negative)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        results = esm_model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        negative = []
        for i, tokens_len in enumerate(batch_lens):
            if query is None:
                negative.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            else:
                mean = token_representations[i, 1 : tokens_len - 1].mean(0).unsqueeze(0)
                q = query(token_representations[i, 1 : tokens_len - 1]) # N x 64
                k = key(mean) # 1 x 64
                prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64)) # N x 1
                negative.append(prob.transpose(0, 1) @ token_representations[i, 1 : tokens_len - 1])
        negative = torch.stack(negative)

        anchor_out = model(anchor.to(device=device, dtype=dtype))
        positive_out = model(positive.to(device=device, dtype=dtype))
        negative_out = model(negative.to(device=device, dtype=dtype))

        loss = criterion(anchor_out, positive_out, negative_out)
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
    seq_dict = {rec.id : rec.seq for rec in SeqIO.parse('./data/' + args.training_data + '.fasta', "fasta")}
    seq_dict.update({rec.id : rec.seq for rec in SeqIO.parse('./data/' + args.training_data + '_single_seq_ECs.fasta', "fasta")})

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
    esm_model, alphabet = pretrained.load_model_and_alphabet(
        'esm1b_t33_650M_UR50S')
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device).half()
    esm_model.eval()
    # use the original esm embedding distance map
    # update every epoch
    esm_emb = pickle.load(
        open('./data/distance_map/' + args.training_data + '_esm.pkl',
                'rb')).to(device=device, dtype=dtype)
    dist_map = pickle.load(open('./data/distance_map/' + \
        args.training_data + '.pkl', 'rb')) 
    #======================== initialize model =================#
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    esm_optimizer = torch.optim.Adam(esm_model.parameters(), lr=1e-6, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    best_loss = float('inf')
    train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
    if args.use_extra_attention:
        query = nn.Linear(1280, 64, bias=False).to(device)
        key = nn.Linear(64, 1280, bias=False).to(device)
        attentions_optimizer = torch.optim.Adam([{"params": query.parameters(), "lr": 0.01, "momentum": 0.9}, {"params": key.parameters(), "lr": 0.01, "momentum": 0.9}])
        attentions = [query, key]
    else:
        attentions = []
        attentions_optimizer = None
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    
    #======================== training =======-=================#
    # training
    for epoch in range(1, epochs + 1):
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999))
            # save updated model
            torch.save(model.state_dict(), './data/model/' +
                       model_name + '_' + str(epoch) + '.pth')
            
            torch.save(esm_model.state_dict(), './data/model/esm_' +
                       model_name + '_' + str(epoch) + '.pth')
            # delete last model checkpoint
            if epoch != args.adaptive_rate:
                os.remove('./data/model/' + model_name + '_' +
                          str(epoch-args.adaptive_rate) + '.pth')
                os.remove('./data/model/esm_' + model_name + '_' +
                          str(epoch-args.adaptive_rate) + '.pth')
            # 
            # generate new esm_emb
            #
            new_esm_emb = {}
            repr_layers = [(i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in [-1]]
            dataset = FastaBatchedDataset.from_file(
                './data/' + args.training_data + '.fasta')
            batches = dataset.get_batch_indices(
                args.toks_per_batch, extra_toks_per_seq=1)
            data_loader = torch.utils.data.DataLoader(
                dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
            )
            with torch.no_grad():
                for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                    print(
                        f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                    )
                    if torch.cuda.is_available() and not args.nogpu:
                        toks = toks.to(device="cuda", non_blocking=True)

                    out = esm_model(toks, repr_layers=repr_layers, return_contacts=False)
                    representations = {
                        layer: t.to(device="cpu") for layer, t in out["representations"].items()
                    }

                    for i, label in enumerate(labels):
                        out = {
                            layer: t[i, 1 : 1023].mean(0).clone()
                            for layer, t in representations.items()
                        }

                        new_esm_emb[label] = out[33]
            
            esm_emb = []
            # for ec in tqdm(list(ec_id_dict.keys())):
            for ec in list(ec_id_dict.keys()):
                ids_for_query = list(ec_id_dict[ec])
                esm_to_cat = [new_esm_emb[id] for id in ids_for_query]
                esm_emb = esm_emb + esm_to_cat
            esm_emb = torch.cat(esm_emb).to(device=device, dtype=dtype)

            # end generate new esm_emb
            
            # sample new distance map
            
            dist_map = get_dist_map(
                ec_id_dict, esm_emb, device, dtype, model=model)
            train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion, 
                           esm_model, esm_optimizer, seq_dict, 
                           batch_converter, alphabet, attentions, attentions_optimizer)
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs):
            torch.save(model.state_dict(), './data/model/' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)
    # remove tmp save weights
    os.remove('./data/model/' + model_name + '.pth')
    os.remove('./data/model/' + model_name + '_' + str(epoch) + '.pth')
    # save final weights
    torch.save(model.state_dict(), './data/model/' + model_name + '.pth')


if __name__ == '__main__':
    main()
