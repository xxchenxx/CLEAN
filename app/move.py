from glob import glob
import os
files = glob("moco_temp_esm_path_10_esm4/*.pt")
for file in files:
    suffix = file.split("epoch1")
    os.system(f"mv {file} {suffix[0] + '/epoch1/' + suffix[1]}")