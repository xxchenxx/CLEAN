import torch
import pandas as pd
index_set = torch.load("index_set.pth.tar")
prob_set = torch.load("prob_set.pth.tar")
df = pd.read_csv("test_new_parsed.csv")
from scipy.stats import spearmanr
import scipy
results = []
collected_sample1 = []
collected_sample2 = []
for key in index_set:
    if key == 'E0VIU9':

        item = df[df['UniprotID'] == key]['parsed_PDB'].values[0]

        print(key)
        print(item)
        index = index_set[key]
        prob = prob_set[key]
        predicted = open(f"predicted_active_sites/{item}-predictions.txt").readlines()
        print(predicted)
        print(len(prob))
        print(index)
        print(len(predicted))