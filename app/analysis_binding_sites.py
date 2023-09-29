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
    item = df[df['UniprotID'] == key]['parsed_PDB'].values[0]

    if 'AF' in item:
        print(item)
        index = index_set[key]
        prob = prob_set[key]
        predicted = open(f"predicted_active_sites/{item}-predictions.txt").readlines()
        predicted = list(map(lambda x: float(x.strip()), predicted))
        if (len(prob) != len(predicted)):
            continue
        index = list(index)
        sample1 = [predicted[i] for i in range(len(predicted)) if i in index]
        sample2 = [predicted[i] for i in range(len(predicted)) if i not in index ]
        # print(sample1)
        # print(sample2)
        collected_sample1.extend(sample1)
        collected_sample2.extend(sample2)
        print(scipy.stats.ttest_ind(sample1, sample2))
        try:
            pred = spearmanr(prob, predicted)
            results.append(pred.statistic)
        except:
            continue

    # data = open("predicted_active_sites/")
with open("sample1.txt", "w") as f:
    f.write(",".join(map(str, collected_sample1)))

with open("sample2.txt", "w") as f:
    f.write(",".join(map(str, collected_sample2)))
print(sum(results) / len(results))