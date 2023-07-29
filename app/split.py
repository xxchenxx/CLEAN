
filelist = []
from glob import glob
filelist = glob("*.fasta")
length = len(filelist) // 10 + 1

for i in range(length):
    filelists = filelist[i * 10:(i+1) * 10]
    with open(f"jobfile{i}") as f:
        for file in filelists:
            f.write(f"singularity exec --nv $AF2_HOME/images/alphafold_2.2.0.sif /app/run_alphafold.sh --flagfile=$AF2_HOME/test/flags/full_dbs.ff --fasta_paths=/work/08298/xxchen/ls6/mutation_0/{file} --output_dir=/work/08298/xxchen/ls6/mutation_0_structure/{file.split('.')[0]} --model_preset=monomer --use_gpu_relax=True\n")
    
        