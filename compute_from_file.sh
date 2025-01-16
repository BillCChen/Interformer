eval "$(conda shell.bash hook)"
conda activate interformer
# PYTHONPATH=/run_interformer/interformer/ python /run_interformer/inference_energy_from_lmdb.py \
# --pocket_lmdb /data_lmdb/pocket.lmdb \
# --ligand_lmdb /data_lmdb/mol.lmdb \
# -ensemble /checkpoints \
# -batch_size 1 \
# -posfix *val_loss* \
# -energy_output_folder /data_lmdb/ \
# -reload \
# -debug  

# OMP_NUM_THREADS="1,64" python -u /run_interformer/docking/reconstruct_1_ligand_from_lmdb.py \
# --sdf_ligand fake_input \
# --sdf_ref fake_input \
# --pdb_complex fake_input \
# --pkl_normalscore fake_input \
# --sdf_output fake_input \
# --pocket_lmdb /data_lmdb/pocket.lmdb \
# --ligand_lmdb /data_lmdb/mol.lmdb \
# --data_dir /data_lmdb/ \
# --weight_intra 30.0 \
# --weight_collision_inter 40.0


PYTHONPATH=/run_interformer/interformer  python /run_interformer/inference_from_lmdb.py \
--redocking_mode \
--pocket_lmdb /data_lmdb/pocket.lmdb \
--ligand_lmdb /data_lmdb/mol.lmdb \
--result /data_lmdb \
--pose_sel True \
-ensemble /checkpoints \
-gpus 1 \
-batch_size 20 \
-posfix *val_loss* 


