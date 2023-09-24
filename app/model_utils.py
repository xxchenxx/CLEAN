import torch
from esm import FastaBatchedDataset, pretrained
import os
import numpy as np

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
                if batch_idx % 100 == 0:
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

def forward_attentions(feature, query, key, learnable_k, args, avg_mask=None, attn_mask=None, return_prob=False):
    
    if len(feature.shape) == 2:
        if not args.use_extra_attention:
            return feature.mean(0)
        else:
            q = query(feature)
            if learnable_k is None:
                k = key(feature)
            elif args.use_input_as_k:
                k = key(feature.mean(1, keepdim=True))
            else:
                k = key(learnable_k)
            raw = torch.einsum('jk,lk->jl', k, q) / np.sqrt(64)
            if args.use_top_k:                
                shape = raw.shape
                raw = raw.reshape(-1, raw.shape[-1])
                _, smallest_value = torch.topk(raw, max(0, raw.shape[1] - 100), largest=False)
                smallest = torch.zeros_like(raw).cpu()
                for j in range(len(raw)):
                    smallest[j, smallest_value[j]] = 1
                smallest = smallest.reshape(shape).bool().to(raw.device)
                raw = raw.reshape(shape)
                raw[smallest] = raw[smallest] + float('-inf')
                prob = torch.softmax(raw, -1) # N x 1
            elif args.use_top_k_sum:
                
                weights = raw.sum(-2, keepdim=True)
                prob = torch.softmax(weights, -1)
            else:
                prob = torch.softmax(raw / np.sqrt(64), 1) # N x 1
            if not return_prob:
                return (prob @ feature).sum(0)
            else:
                return (prob @ feature).sum(0), prob, raw
    else: # 3-D features with paddings
        assert avg_mask is not None
        assert attn_mask is not None
        q = query(feature)
        if learnable_k is not None:
            k = key(learnable_k).unsqueeze(0).repeat(q.shape[0], 1, 1)
        elif args.use_input_as_k:
            k = key(feature.mean(1, keepdim=True))
        else:
            k = key(feature)
        raw = torch.einsum('ijk,ilk->ijl', k, q) / np.sqrt(64) + attn_mask
        if args.use_top_k:
            shape = raw.shape
            raw = raw.reshape(-1, raw.shape[-1])
            _, smallest_value = torch.topk(raw, max(0, raw.shape[1] - 100), largest=False)
            smallest = torch.zeros_like(raw)
            for j in range(len(raw)):
                smallest[j, smallest_value[j]] = 1
            smallest = smallest.reshape(shape).bool()
            raw = raw.reshape(shape)
            raw[smallest] = raw[smallest] + float('-inf')
            prob = torch.softmax(raw, -1) 
        elif args.use_top_k_sum:
            
            weights = raw.sum(-2, keepdim=True)
            for i in range(len(weights)):
                weights[i, 0][avg_mask[i] == 0] = -float("inf")
            prob = torch.softmax(weights, -1)
        else:
            prob = torch.softmax(raw / np.sqrt(64) + attn_mask, -1) 
        multiplied = torch.bmm(prob, feature)
        if not args.use_top_k_sum:
            if not return_prob:
                return torch.sum(multiplied * avg_mask.unsqueeze(-1).repeat(1, 1, multiplied.shape[-1]), 1) # proj matrix is NxN
            else:
                return torch.sum(multiplied * avg_mask.unsqueeze(-1).repeat(1, 1, multiplied.shape[-1]), 1), prob
        else:
            assert multiplied.shape[1] == 1
            if not return_prob:
                return multiplied.squeeze(-2)
            else:
                return multiplied.squeeze(-2), prob, raw
