python prepare_pic50_data.py # make numpy data first
CUDA_VISIBLE_DEVICES=7 python train.py --train_keys keys/PDBBind_pIC50_sample_train_keys.pkl --test_keys keys/PDBBind_pIC50_sample_test_keys.pkl --data_dir data_numpy