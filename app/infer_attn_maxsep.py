import torch
import torch.nn as nn
from CLEAN.utils import *
from CLEAN.model import *
from CLEAN.evaluate import *
from CLEAN.infer import *
import os
from Bio import SeqIO
file_1 = open('./data/split10.fasta')
query = nn.Linear(1280, 64, bias=False).cuda()
nn.init.zeros_(query.weight)
key = nn.Linear(1280, 64, bias=False).cuda()
nn.init.zeros_(key.weight)
device = torch.device("cuda:0")
dtype = torch.float32
model = MoCo(512, 128, device, dtype)
query.load_state_dict(torch.load("data/model/moco_query_best_split10_triplet_backbone_with_moco.pth"))
key.load_state_dict(torch.load("data/model/moco_key_best_split10_triplet_backbone_with_moco.pth"))
model.load_state_dict(torch.load("data/model/moco_best_split10_triplet_backbone_with_moco.pth"))

id_ec, ec_id_dict = get_ec_id_dict('./data/split10.csv')
ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}

new_esm_emb = {}
dict1 = SeqIO.to_dict(SeqIO.parse(file_1, "fasta"))

for key_ in list(dict1.keys()):
    with torch.no_grad():
        if os.path.exists('temp_esm_path_10/epoch1/' + key_ + ".pt"):
            embed = torch.load("temp_esm_path_10/epoch1/" + key_ + ".pt")
            q = query(embed.cuda())
            k = key(embed.cuda())
            prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64), 0) # N x 1
            new_esm_emb[key_] = (prob @ embed.cuda()).mean(0)

train_esm_emb = []
                # for ec in tqdm(list(ec_id_dict.keys())):
for ec in list(ec_id_dict.keys()):
    ids_for_query = list(ec_id_dict[ec])
    esm_to_cat = [new_esm_emb[id] for id in ids_for_query]
    train_esm_emb = train_esm_emb + esm_to_cat
train_esm_emb = torch.stack(train_esm_emb).to(device=device, dtype=dtype)

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

dataset = FastaBatchedDataset.from_file(
                './data/price.fasta')
batches = dataset.get_batch_indices(
    4096, extra_toks_per_seq=1)

esm_model, alphabet = pretrained.load_model_and_alphabet(
        'esm1b_t33_650M_UR50S.pt')
esm_model = esm_model.to(device)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
)
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

test_emb = {}
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
            feat = out[33].cuda()
            mask = toks[i, 1:1023] != 1
            feat = feat[mask]
            q = query(feat)
            k = key(feat)
            prob = torch.softmax(q @ k.transpose(0, 1) / np.sqrt(64), 0) # N x 1
            test_emb[label] = (prob @ feat).mean(0)
test_esm_emb = []
id_ec_test, ec_id_dict_test = get_ec_id_dict('./data/price.csv')
ids_for_query = list(id_ec_test.keys())
esm_to_cat = [test_emb[id] for id in ids_for_query]
    
test_esm_emb = torch.stack(esm_to_cat)
model.eval()
train_esm_emb = model.encoder_q(train_esm_emb)
test_esm_emb = model.encoder_q(test_esm_emb)
eval_dist = get_dist_map_test(train_esm_emb, test_esm_emb, ec_id_dict,id_ec_test, device, dtype)
eval_df = pd.DataFrame.from_dict(eval_dist)
out_filename = "results/new_temp" 
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