import glob
import os
import shutil

# DDP inference
import pytorch_lightning as pl
import pickle
import torch.distributed as dist
from collections import defaultdict
from torchmetrics.functional import pearson_corrcoef
import torch
import numpy as np
#
from utils.parser import get_args
from utils.train_utils import load_from_checkpoint, param_count, get_checkpoint_realpath, refresh_args
from data.dataset.bindingdata import BindingData,Binding_dict_Data
from data.data_process import data_loader_warper
from utils.eval import cal_per_target
import time
os.environ["NCCL_IB_DISABLE"] = "1"
args = get_args()
args['inference'] = True
args['device'] = torch.device(args['gpus'])
os.makedirs('result', exist_ok=True)
import lmdb
def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )  
    data_l = {}
    out_list = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            data = pickle.loads(value)
            out_list.append(data)
    env.close()
    return out_list

def collect_results_gpu(result_part):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    torch.cuda.synchronize()
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for single_gpu_res in part_list:
            for res in single_gpu_res:
                ordered_results.extend([list(res)])
        ###########
        # merge gpus result together
        N = ordered_results[0][1].size(0)
        pred, Y, targets, idx = [], [], [], []
        for batch_data in ordered_results:
            pred.append(batch_data[0])
            Y.append(batch_data[1].cpu())
            targets.extend(batch_data[2])
            idx += list(batch_data[3])
        # gathering
        Y = torch.cat(Y)
        targets = np.array(targets)
        # multi-task
        num_hat = len(ordered_results[0][0])
        all_pred = []
        for i in range(num_hat):
            one_hat = [x[i].cpu() for x in pred]
            # Loss Output, skip
            if len(one_hat[0].shape) == 0:
                continue
            all_pred.append(torch.cat(one_hat))
        all_pred = torch.cat(all_pred, dim=-1)
        # reorder by idx
        real_idx = [x[1] for x in sorted([(id, i) for i, id in enumerate(idx)])]
        Y = Y[real_idx]
        all_pred = all_pred[real_idx]
        targets = targets[real_idx].tolist()
        return all_pred, Y, targets

pockets = read_lmdb(args['pocket_lmdb'])
ligands = read_lmdb(args['ligand_lmdb'])
dfs = {}
ensemble_results = defaultdict(list)
models_N = 0
rank = 0
models_names = []
ensemble_results = []
for idx,checkpoint_folder in enumerate(args['ensemble']):
    checkpoint = get_checkpoint_realpath(checkpoint_folder, posfix=args['posfix'])
    if checkpoint == '':
        print(f"# Model doesn't Exists:{checkpoint_folder}")
        continue
    # record
    models_N += 1
    models_names.append(checkpoint[:checkpoint.rfind('/check')])
    model = load_from_checkpoint(args, checkpoint).eval()
    args = refresh_args(args, model)
    model_args = model.hparams.args
    param_count(model, print_model=False)
    inferencer = pl.Trainer(devices=args['gpus'], accelerator='cuda', strategy='ddp', precision=args['precision'],
                            logger=False)
    # restrucutring for lmdb conataining pdb and sdf input,not csv input
    # construct to dataset with 2 file_path
    # inference
    result = {}
    for pocket_,ligands_ in (zip(pockets,ligands)):
        # dataset
        data = Binding_dict_Data(args,pocket_, ligands_,n_jobs=1 if args['debug'] else args['n_jobs'], istrain=False)
        loader = data_loader_warper(data, args['batch_size'],
                                    energy_mode=args['energy_mode'] if 'energy_mode' in args else False,
                                    model=args['model'],
                                    num_workers=0 if args['debug'] else 5)
        # predict
        res = inferencer.predict(model, dataloaders=loader)
        results = collect_results_gpu(res)
        rank = dist.get_rank()
        if rank == 0:
            #########
            # calculate score
            pred, Y, targets = results
            pred = pred.float()
            if pred.size(0) and not model_args['energy_mode']:
                # multi-task output
                if pred.size(-1) > 1:
                    affinity_pred, pose_pred = pred[:, 0].view(-1, 1), torch.sigmoid(pred[:, 1])
                else:
                    affinity_pred = pred
                ##
            # 把pred 和 Y 转换成numpy，再转换成list
            affinity_pred = affinity_pred.cpu().reshape(-1).numpy().tolist()
            pose_pred = pose_pred.cpu().reshape(-1).numpy().tolist()
            Y = Y.cpu().reshape(-1).numpy().tolist()
            print(f"pearson-R of {pocket_['protein']}",pearson_corrcoef(torch.tensor(affinity_pred),torch.tensor(Y)))
            result[pocket_['protein']] = (affinity_pred, Y, pose_pred) 
    print('%' * 100)
    ensemble_results.append(result)
##########
# Master Calculate
if rank == 0:
    # merge all results
    # 对每个ensemble_result的result的每个protein的affinity_pred进行均值
    # save the results
    all_results = {}
    for idx,result in enumerate(ensemble_results):
        for protein in result.keys():
            if protein not in all_results:
                all_results[protein] = result[protein]
            else:
                all_results[protein] = (np.array(all_results[protein][0]) + np.array(result[protein][0]),all_results[protein][1],np.array(all_results[protein][2]) + np.array(result[protein][2]))
    print('ensemble results merged')
    print("all pro name:  ",all_results.keys())
    for protein in all_results:
        all_results[protein] = ([i/models_N for i in all_results[protein][0]],all_results[protein][1],
                                [i/models_N for i in all_results[protein][2]])
    with open('/data_lmdb/result.pkl','wb') as f:
        pickle.dump(all_results,f)
    print('results saved to result.pkl')
    print('Affinity inference finished')