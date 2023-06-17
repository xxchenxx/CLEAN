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
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


def get_first_device(model):
        if model.devices:
            return model.devices[0]
        else:
            return torch.cuda.current_device()

def get_last_device(model):
        if model.devices:
            return model.devices[-1]
        else:
            return torch.cuda.current_device()
        
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
    parser.add_argument('--use_extra_attention', type=bool, default=False)
    parser.add_argument('--temp_esm_path', type=str, required=True)
    args = parser.parse_args()
    return args

from torch.nn.utils.rnn import pad_sequence

def custom_collate(data): #(2)
    anchor = [d[0] for d in data]
    positive = [d[1] for d in data]
    negative = [d[2] for d in data]
    anchor_length = torch.tensor(list(map(len, anchor)))
    positive_length = torch.tensor(list(map(len, positive)))
    negative_length = torch.tensor(list(map(len, negative)))

    anchor = pad_sequence(anchor, batch_first=True)
    positive = pad_sequence(positive, batch_first=True)
    negative = pad_sequence(negative, batch_first=True)
    anchor_attn_mask = torch.ones(anchor.shape[0], anchor.shape[1], anchor.shape[1]) * -np.inf
    positive_attn_mask = torch.ones(positive.shape[0], positive.shape[1], positive.shape[1]) * -np.inf
    negative_attn_mask = torch.ones(negative.shape[0], negative.shape[1], negative.shape[1]) * -np.inf
    
    for i in range(anchor_attn_mask.shape[0]):
        anchor_attn_mask[i, :, :anchor_length[i]] = 0
        positive_attn_mask[i, :, :positive_length[i]] = 0
        negative_attn_mask[i, :, :negative_length[i]] = 0
        
    anchor_avg_mask = torch.ones(anchor.shape[0], anchor.shape[1])
    positive_avg_mask = torch.ones(positive.shape[0], positive.shape[1])
    negative_avg_mask = torch.ones(negative.shape[0], negative.shape[1])
    for i in range(anchor_avg_mask.shape[0]):
        anchor_avg_mask[i, :anchor_length[i]] = 1 / anchor_length[i]
        positive_avg_mask[i, :positive_length[i]] = 1 / positive_length[i]
        negative_avg_mask[i, :negative_length[i]] = 1 / negative_length[i]
    return anchor, positive, negative, anchor_attn_mask, positive_attn_mask, negative_attn_mask, anchor_avg_mask, positive_avg_mask, negative_avg_mask

def get_dataloader(dist_map, id_ec, ec_id, args, temp_esm_path="./data/esm_data/"):
    train_params = {
        'batch_size': 2,
        'shuffle': True,
    }
    embed_params = {
        'batch_size': 256,
        'shuffle': True,
        'collate_fn': custom_collate
    }
    negative = mine_hard_negative(dist_map, 30)
    train_data = Triplet_dataset_with_mine_EC_text(id_ec, ec_id, negative)
    train_embed = Triplet_dataset_with_mine_EC(id_ec, ec_id, negative, path=temp_esm_path)
    train_loader = torch.utils.data.DataLoader(train_data, **train_params)
    static_embed_loader = torch.utils.data.DataLoader(train_embed, **embed_params)
    return train_loader, static_embed_loader


def train(model, args, epoch, train_loader, static_embed_loader,
          optimizer, device, dtype, criterion, esm_model, esm_optimizer, seq_dict, batch_converter, alphabet, attentions, attentions_optimizer=None):
    model.train()
    total_loss = 0.
    start_time = time.time()
    esm_model.train()
    if len(attentions) > 0:
        query, key = attentions
    else:
        query = None
    if (epoch + 1) % args.train_esm_rate == 0:
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
                    
                    q = query(token_representations[i, 1 : tokens_len - 1]) # N x 64
                    k = key(token_representations[i, 1 : tokens_len - 1]) # 1 x 64
                    prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64), 0) # N x 1
                    anchor.append((prob.transpose(0, 1) @ token_representations[i, 1 : tokens_len - 1]).mean(0))
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
                    
                    q = query(token_representations[i, 1 : tokens_len - 1]) # N x 64
                    k = key(token_representations[i, 1 : tokens_len - 1]) # 1 x 64
                    prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64), 0) # N x 1
                    positive.append((prob.transpose(0, 1) @ token_representations[i, 1 : tokens_len - 1]).mean(0))
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
                    q = query(token_representations[i, 1 : tokens_len - 1]) # N x 64
                    k = key(token_representations[i, 1 : tokens_len - 1]) # 1 x 64
                    prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64), 0) # N x 1
                    negative.append((prob.transpose(0, 1) @ token_representations[i, 1 : tokens_len - 1]).mean(0))
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
    else:
        for batch, data in enumerate(static_embed_loader):
            optimizer.zero_grad()
            anchor_original, positive_original, negative_original, anchor_attn_mask, positive_attn_mask, negative_attn_mask, anchor_avg_mask, positive_avg_mask, negative_avg_mask = data
            anchor_original = anchor_original.cuda()
            positive_original = positive_original.cuda()
            negative_original = negative_original.cuda()
            anchor_attn_mask = anchor_attn_mask.cuda()
            positive_attn_mask = positive_attn_mask.cuda()
            negative_attn_mask = negative_attn_mask.cuda()
            anchor_avg_mask = anchor_avg_mask.cuda()
            positive_avg_mask = positive_avg_mask.cuda()
            negative_avg_mask = negative_avg_mask.cuda()
            anchor = []
            negative = []
            positive = []
            
            if query is None:
                negative = torch.sum(negative_original * negative_avg_mask, 1)
                positive = torch.sum(positive_original * positive_avg_mask, 1)
                anchor = torch.sum(anchor_original * anchor_avg_mask, 1)
            else:
                q_negative = query(negative_original) # N x 64
                k_negative = key(negative_original) # 1 x 64
                prob = torch.softmax(torch.bmm(q_negative, k_negative.transpose(1, 2)) / np.sqrt(64) + negative_attn_mask, -1) # N x 1

                negative_multiple = torch.bmm(prob, negative_original)

                negative = torch.sum(negative_multiple * negative_avg_mask.unsqueeze(-1), 1)

                q_positive = query(positive_original) # N x 64
                k_positive = key(positive_original) # 1 x 64
                prob = torch.softmax(torch.bmm(q_positive, k_positive.transpose(1, 2)) / np.sqrt(64) + positive_attn_mask, -1) 

                positive_multiple = torch.bmm(prob, positive_original)
                positive = torch.sum(positive_multiple * positive_avg_mask.unsqueeze(-1), 1)

                q_anchor = query(anchor_original) # N x 64
                k_anchor = key(anchor_original) # 1 x 64
                prob = torch.softmax(torch.bmm(q_anchor, k_anchor.transpose(1, 2)) / np.sqrt(64) + anchor_attn_mask, -1)  # N x 1

                anchor_multiple = torch.bmm(prob, anchor_original)
                anchor = torch.sum(anchor_multiple * anchor_avg_mask.unsqueeze(-1), 1)
            anchor_out = model(anchor.to(device=device, dtype=dtype))
            positive_out = model(positive.to(device=device, dtype=dtype))
            negative_out = model(negative.to(device=device, dtype=dtype))

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if args.verbose:
                lr = args.learning_rate
                ms_per_batch = (time.time() - start_time) * 1000
                cur_loss = total_loss 
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(static_embed_loader):5d} batches | '
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
    ec_id = {key_: list(ec_id_dict[key_]) for key_ in ec_id_dict.keys()}
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
        'esm1b_t33_650M_UR50S.pt')
    esm_model = esm_model.to(device)

    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    
    # esm_emb = pickle.load(
    #    open('./data/distance_map/' + args.training_data + '_esm.pkl',
    #           'rb')).to(device=device, dtype=dtype)
    #dist_map = pickle.load(open('./data/distance_map/' + \
    #    args.training_data + '.pkl', 'rb')) 
    #======================== initialize model =================#
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    esm_optimizer = torch.optim.Adam(esm_model.parameters(), lr=1e-6, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    best_loss = float('inf')

    if args.use_extra_attention:
        query = nn.Linear(1280, 64, bias=False).to(device)
        nn.init.zeros_(query.weight)
        key = nn.Linear(1280, 64, bias=False).to(device)
        nn.init.zeros_(key.weight)
        attentions_optimizer = torch.optim.Adam([{"params": query.parameters(), "lr": lr, "momentum": 0.9}, {"params": key.parameters(), "lr": lr, "momentum": 0.9}])
        attentions = [query, key]
    else:
        attentions = []
        attentions_optimizer = None
    #======================== generate embed =================#
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
        print(f"Need to parse {remain}/{original}")
        with open(f'temp_{args.training_data}.fasta', 'w') as handle:
            SeqIO.write(dict1.values(), handle, 'fasta')

        dataset = FastaBatchedDataset.from_file(f'temp_{args.training_data}.fasta')
        batches = dataset.get_batch_indices(
            4096, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
        )
        os.makedirs(args.temp_esm_path + f"/epoch{start_epoch}", exist_ok=True)
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                )
                toks = toks.to(device="cuda", non_blocking=True)
                out = esm_model(toks, repr_layers=[33], return_contacts=False)
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                for i, label in enumerate(labels):
                    out = {
                        layer: t[i, 1 : 1023].clone()
                        for layer, t in representations.items()
                    }
                    torch.save(out[33], args.temp_esm_path + f"/epoch{start_epoch}/" + label + ".pt")
                    new_esm_emb[label] = out[33].mean(0).cpu()
                    
        original = len(list(dict2.keys()))
        new_esm_emb_2 = {}
        for key_ in list(dict2.keys()):
            if os.path.exists(args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt"):
                del dict2[key_]
                new_esm_emb[key_] = torch.load(args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt").mean(0)
        
        remain = len(dict2)
        print(f"Need to parse {remain}/{original}")

        with open(f'temp_{args.training_data}.fasta', 'w') as handle:
            SeqIO.write(dict2.values(), handle, 'fasta')

        dataset = FastaBatchedDataset.from_file(f'temp_{args.training_data}.fasta')
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
                out = esm_model(toks, repr_layers=[33], return_contacts=False)
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                for i, label in enumerate(labels):
                    out = {
                        layer: t[i, 1 : 1023].clone()
                        for layer, t in representations.items()
                    }

                    torch.save(out[33], args.temp_esm_path + f"/epoch{start_epoch}/" + label + ".pt")
                    # new_esm_emb_2[label] = out[33].mean(0)

        esm_emb = []
        new_esm_emb.update(new_esm_emb_2)
        for ec in list(ec_id_dict.keys()):
            ids_for_query = list(ec_id_dict[ec])
            esm_to_cat = [new_esm_emb[id] for id in ids_for_query]
            esm_emb = esm_emb + esm_to_cat
        esm_emb = torch.stack(esm_emb).to(device=device, dtype=dtype)

        dist_map = get_dist_map(
            ec_id_dict, esm_emb, device, dtype)
        os.makedirs(args.temp_esm_path + f"/epoch{start_epoch}", exist_ok=True)
        for key_ in new_esm_emb_2:
            if not os.path.exists(args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt"):
                torch.save(new_esm_emb_2[key_], args.temp_esm_path + f"/epoch{start_epoch}/" + key_ + ".pt")
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
                    out = esm_model(toks, repr_layers=[33], return_contacts=False)
                    representations = {
                        layer: t.to(device="cpu") for layer, t in out["representations"].items()
                    }

                    for i, label in enumerate(labels):
                        out = {
                            layer: t[i, 1 : 1023].clone()
                            for layer, t in representations.items()
                        }

                        new_esm_emb[label] = out[33]
            dataset = FastaBatchedDataset.from_file(
                './data/' + args.training_data + '_single_seq_ECs.fasta')
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
                    out = esm_model(toks, repr_layers=[33], return_contacts=False)
                    representations = {
                        layer: t.to(device="cpu") for layer, t in out["representations"].items()
                    }
                    for i, label in enumerate(labels):
                        out = {
                            layer: t[i, 1 : 1023].clone()
                            for layer, t in representations.items()
                        }
                        new_esm_emb[label] = out[33]
            esm_emb = []
            # for ec in tqdm(list(ec_id_dict.keys())):
            for ec in list(ec_id_dict.keys()):
                ids_for_query = list(ec_id_dict[ec])
                if not args.use_extra_attention:
                    esm_to_cat = [new_esm_emb[id].mean(0) for id in ids_for_query]
                else:
                    esm_to_cat = []
                    with torch.no_grad():
                        for id in ids_for_query:
                            q = query(new_esm_emb[id].cuda())
                            k = key(new_esm_emb[id].cuda())
                            prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64), 0) # N x 1
                            esm_to_cat.append((prob @ new_esm_emb[id].cuda()).mean(0).cuda())
                esm_emb = esm_emb + esm_to_cat
            esm_emb = torch.stack(esm_emb).to(device=device, dtype=dtype)
            dist_map = get_dist_map(
                ec_id_dict, esm_emb, device, dtype, model=model)

            for key_ in new_esm_emb:
                torch.save(new_esm_emb[key_], args.temp_esm_path + "/" + key_ + ".pt")
            del new_esm_emb
            train_loader, static_embed_loader = get_dataloader(dist_map, id_ec, ec_id, args, args.temp_esm_path)
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader, static_embed_loader,
                           optimizer, device, dtype, criterion, 
                           esm_model, esm_optimizer, seq_dict, 
                           batch_converter, alphabet, attentions, attentions_optimizer)
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs):
            torch.save(model.state_dict(), './data/model/' + model_name + '.pth')
            torch.save(model.state_dict(), './data/model/esm_' + model_name + '.pth')
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
    torch.save(model.state_dict(), './data/model/esm_' + model_name + '.pth')

if __name__ == '__main__':
    main()
