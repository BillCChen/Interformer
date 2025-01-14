# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import argparse
import logging

from reconstruct_ligands import reconstruct_1_ligand_given_paths,reconstruct_1_ligand_given_paths_

LOGGER = logging.getLogger(__name__)


def get_args_and_mainparser():
    """
    """
    mainparser = argparse.ArgumentParser()

    mainparser.add_argument(
        "--sdf_ligand", type=str, required=True,
    )
    mainparser.add_argument(
        "--sdf_ref", type=str, required=True,
    )
    mainparser.add_argument(
        "--pdb_complex", type=str, required=True,
    )
    mainparser.add_argument(
        "--pkl_normalscore", type=str, required=True,
    )
    mainparser.add_argument(
        "--sdf_output", type=str, required=True,
    )
    mainparser.add_argument("--csv_stat", type=str, default=None)
    mainparser.add_argument("--pdb_id", type=str, default=None)
    mainparser.add_argument(
        "--weight_intra",
        type=float,
        default=None,
        help="use --weight_intra=0.0 to reproduce the paper result (top1=0.56);"
             " use --weight_intra=30.0 to get a much better performance (top1=0.70)",
    )
    mainparser.add_argument(
        "--weight_collision_inter",
        type=float,
        default=None,
        help="if None, --weight_collision_inter=1.0",
    )
    # for rerank inference
    mainparser.add_argument('--pocket_lmdb', default="pocket.lmdb",type=str, required=False)
    mainparser.add_argument('--ligand_lmdb', default="ligand.lmdb",type=str, required=False)
    mainparser.add_argument('--data_dir', default="/data_lmdb",type=str, required=False)
    args = mainparser.parse_args()

    return args, mainparser
import lmdb
import pickle
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
from tqdm import tqdm
import io
from rdkit import Chem
def main() -> None:
    """
    """
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(logging.INFO)

    args, mainparser = get_args_and_mainparser()
    pockets = read_lmdb(args.pocket_lmdb)
    ligands = read_lmdb(args.ligand_lmdb)
    liagnds_ = []
    with tqdm(enumerate(pockets),total=len(pockets)) as t:
        file_path = "/data_lmdb/tmp/tmp.sdf"
        
        for idx,pocket in t:
            files = []
            mols = []
            for num,ligand in enumerate(ligands[idx]):
                save_dir = f"/data_lmdb/tmp/{pocket['protein']}_ligand_{num}.sdf"
                files.append(save_dir)
                sdf_bytes = ligand['sdf'].encode('utf-8')
                sdf_io = io.BytesIO(sdf_bytes)
                mol = [x for x in Chem.ForwardSDMolSupplier(sdf_io) if x is not None][0]
                Chem.MolToMolFile(mol,save_dir)
                # mols.append(mol)
            # w = Chem.SDWriter(file_path)
            # for mol in mols:
            #     w.write(mol)
            # w.close()
            name = pocket['protein']
            t.set_description(f"Processing {name} with Monte Carlo Sampling")
            sdf_ligand = file_path
            # f"/data_lmdb/uff/{name}_uff.sdf"
            sdf_ref = f"/data_lmdb/uff/{name}_uff.sdf"
            pdb_complex = f"/data_lmdb/complex/{name}_complex.pdb"
            pkl_normalscore = f"/data_lmdb/gaussian_predict/{name}_G.db"
            sdf_output = f"/data_lmdb/ligand_MC/{name}_MC_ligands.sdf"
            for idy,file in tqdm(enumerate(files),total=len(files),desc="Reconstructing Ligands"):
                sdf_output = f"/data_lmdb/ligand_MC/{name}_MC_ligands_{idy}.sdf"
                reconstruct_1_ligand_given_paths_(
                    db_index = idy,
                    # path_sdf_ligand=sdf_ligand,
                    path_sdf_ligand=file,
                    path_sdf_ref=sdf_ref,
                    path_pdb_complex=pdb_complex,
                    path_pkl_normalscore=pkl_normalscore,
                    path_sdf_output=sdf_output,
                    # path_csv_stat=args.csv_stat,
                    # str_pdb_id=args.pdb_id,
                    weight_intra=args.weight_intra,
                    weight_collision_inter=args.weight_collision_inter,
                    num_output_poses=1,
                    )
if __name__ == "__main__":
    main()
