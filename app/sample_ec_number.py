import pandas as pd
data = pd.read_csv("data/split10.csv", sep='\t')
print(data)
indexes = []
for i in range(len(data)):
    
    if data.loc[i, 'EC number'].startswith('3.'):
        indexes.append(i)

print(indexes)
import random
indexes = random.sample(indexes, 30)
results = data.loc[indexes]

results.to_csv("data/sampled.csv", index=False, sep='\t')