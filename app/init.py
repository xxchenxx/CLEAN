from CLEAN.utils import *
# train_file = 'split10'
# csv_to_fasta(f"data/{train_file}.csv", f"data/{train_file}.fasta")
# retrive_esm1b_embedding(f"{train_file}")
# train_fasta_file = mutate_single_seq_ECs(train_file)
# retrive_esm1b_embedding(train_fasta_file)
# compute_esm_distance(train_file, 'esm2_35m_features', 'esm2_35m_features')


from CLEAN.utils import *
train_file = 'split10'
compute_esm_distance(train_file, 'esm2_150m_features/', 'esm2_150m_features/')