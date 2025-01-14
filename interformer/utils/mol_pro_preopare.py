from openbabel import openbabel, pybel
from rdkit import Chem

def rdkit_to_pybel(rdkit_mol):
    """
    将RDKit的Mol对象转换为pybel的Molecule对象
    :param rdkit_mol: RDKit的Mol对象
    :return: pybel的Molecule对象
    """
    mol_block = Chem.MolToMolBlock(rdkit_mol)
    pybel_mol = pybel.readstring("mol", mol_block)
    return pybel_mol

def pybel_to_rdkit(pybel_mol):
    """
    将pybel的Molecule对象转换为RDKit的Mol对象
    :param pybel_mol: pybel的Molecule对象
    :return: RDKit的Mol对象
    """
    mol_block = pybel_mol.write("mol")
    rdkit_mol = Chem.MolFromMolBlock(mol_block)
    return rdkit_mol
def protonate_molecule(mol):
    """
    质子化给定的分子对象并返回质子化后的分子对象
    :param mol: pybel.Molecule对象
    :return: 质子化后的pybel.Molecule对象
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "sdf")

    # 创建一个OBMol对象并从pybel.Molecule对象中转换
    obMol = mol.OBMol

    # 质子化分子
    obMol.AddHydrogens(False, True, 7.4)

    # 将质子化后的OBMol对象转换回pybel.Molecule对象
    protonated_mol = pybel.Molecule(obMol)
    return protonated_mol

def protonate_rdkit_molecule(rdkit_mol):
    """
    质子化给定的RDKit分子对象并返回质子化后的RDKit分子对象
    :param rdkit_mol: RDKit的Mol对象
    :return: 质子化后的RDKit的Mol对象
    """
    pybel_mol = rdkit_to_pybel(rdkit_mol)
    protonated_pybel_mol = protonate_molecule(pybel_mol)
    protonated_rdkit_mol = pybel_to_rdkit(protonated_pybel_mol)
    return protonated_rdkit_mol