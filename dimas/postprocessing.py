import os
import sys 
import subprocess
import hashlib

from ase.io import read, write
from ase import Atoms

from openbabel import openbabel as ob
from openbabel import pybel

# --------------------------------
# OpenBabel 
# --------------------------------
os.environ["BABEL_DATADIR"] = "/opt/miniconda3/envs/ase/share/openbabel/3.1.0"

def comma_string_to_dict(s):   #  We now read info as a single-entry dictionary and split 
                               #  the string (value) into fields
    items = [item.strip() for item in s.split(',')]
    return dict(zip(items[0::2], items[1::2]))

# --------------------------------
# SMILES
# --------------------------------
def obmol_smiles(obmol):

    pyb = pybel.Molecule(obmol)
    return pyb.write("can").strip()

# --------------------------------
# TPSA
# --------------------------------
def obmol_tpsa(obmol):

    pyb = pybel.Molecule(obmol)
    return pyb.calcdesc(["TPSA"])["TPSA"]

# --------------------------------
# logP
# --------------------------------
def obmol_logp(obmol):
    pyb = pybel.Molecule(obmol)
    return pyb.calcdesc(["logP"])["logP"]

# --------------------------------
# Charge
# --------------------------------
def obmol_charge(obmol):

    charge_model = ob.OBChargeModel.FindType("mmff94")
    charge_model.ComputeCharges(obmol)

    total = 0.0
    for atom in ob.OBMolAtomIter(obmol):
        total += atom.GetPartialCharge()

    return round(total)

# --------------------------------
# Hash
# --------------------------------
def smiles_hash(smiles):

    return int(
        hashlib.sha256(smiles.encode()).hexdigest(),
        16
    ) % (10**12)
# --------------------------------
# OBMol → ASE
# --------------------------------
def obmol_to_ase(obmol):

    symbols = []
    positions = []

    for atom in ob.OBMolAtomIter(obmol):
        symbols.append(ob.GetSymbol(atom.GetAtomicNum()))
        positions.append([
            atom.GetX(),
            atom.GetY(),
            atom.GetZ()
        ])

    return Atoms(symbols=symbols, positions=positions)

# --------------------------------
# Classify complex
# --------------------------------
def classify_complex(frags):

    charges = [obmol_charge(f) for f in frags]

    if len(frags) == 1:
        itype=0
        label="single_molecule"

    elif any(c != 0 for c in charges):
        itype=1
        label="ionic_complex"

    else:
        itype=2
        label="neutral_complex"

    return itype,label

# ------------------------------------
# ASE -> OpenBabel reordered fragments 
# ------------------------------------
def split_fragments_obmol(atoms):
    obmol = ob.OBMol()
# --------------------------------
#  From ASE atoms to obmol
# --------------------------------
    for atom in atoms:
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(int(atom.number))
        x, y, z = atom.position
        obatom.SetVector(float(x), float(y), float(z))
# --------------------------------
#  Connectivity
# --------------------------------
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
#
#   Descriptors
#
    tpsa = obmol_tpsa(obmol)
    logp = obmol_logp(obmol)
    charge = obmol_charge(obmol)
    smiles = obmol_smiles(obmol)
    hash_id = smiles_hash(smiles)
    nrot=obmol.NumRotors()
# --------------------------------
# Split fragments
# --------------------------------
    frags = obmol.Separate()
# --------------------------------
# Reorder fragments
# --------------------------------
    frags = sorted( frags, key=lambda x: x.NumAtoms(), reverse=True)
    return frags , smiles , hash_id, charge , tpsa, logp, nrot

# =================================
# MAIN
# =================================

if len(sys.argv) != 2:
        print("Usage: python postprocessing_omol25_A.py  <filename.xyz>")
        sys.exit(1)

# Database file 
filename=sys.argv[1]
output_single=filename.replace('.xyz','_single.xyz')
output_cmplx=filename.replace('.xyz','_cmplx.xyz')

# Reading traj file 
# The properties line is parsed as a dictionary having a single-entry 

traj = read(filename, ":", format="extxyz", properties_parser=lambda s: {"title": s})

icmplx=0
isngl=0
for i, atoms in enumerate(traj):

    info=atoms.info["title"] # CSV info string
    info_dict=comma_string_to_dict( atoms.info["title"] )   # Converting CSV into dictionary 

    frags,smiles_full, hash_id_full, charge_full, tpsa_full, logp_full, nrot_full  = split_fragments_obmol(atoms)

    info= info + f",idsml,{hash_id_full},smiles,{smiles_full},tpsa,{tpsa_full:5.2f},logp,{logp_full:5.2f},nrot,{nrot_full}"

    itype, classification = classify_complex(frags)

    print(f"Frame {i} Fragments detected: {len(frags)} Type: {classification}")

    tempfile=f"TEMP_{info_dict['ID']}_{hash_id_full}.xyz"

    if ( itype >  0 ) :

        ase_fragments = []
        info=info+f",nfrag,{len(frags)},"
        for j, frag in enumerate(frags):

            smiles = obmol_smiles(frag)
            tpsa = obmol_tpsa(frag)
            logp = obmol_logp(frag)
            charge = obmol_charge(frag)
            hash_id = smiles_hash(smiles)
            nat=frag.NumAtoms()
            nrot=frag.NumRotors()

            info= info + f"nat_{j},{nat},idsml_{j},{hash_id},smiles_{j},{smiles},tpsa_{j},{tpsa:5.2f},logp_{j},{logp:5.2f},nrot_{j},{nrot},chrg_{j},{charge},"

            ase_frag = obmol_to_ase(frag)

        # External script uses ANTECHAMBER to reorder atoms
            tempfile_frag=tempfile.replace('.xyz',f"_{j}.xyz")
            write(tempfile_frag,ase_frag,format='xyz')
            subprocess.run(["/datos_pool/mldata1/QMdatasets/OMol25/AIM4ML/scripts/xyz_reord.sh",tempfile_frag])
            try:
                ase_frag_reord=read(tempfile_frag)
            except Exception:
                continue 
            os.remove(tempfile_frag)
            ase_fragments.append(ase_frag_reord)

        # Rebuild reordered ASE structure
        merged = ase_fragments[0]
        for frag in ase_fragments[1:]:
            merged += frag

        if ( icmplx == 0 ):
             write(output_cmplx,merged,comment=info,format='extxyz')
        else:
             write(output_cmplx,merged,append=True,comment=info,format='extxyz')
        icmplx=icmplx+1

    else:
        # External script uses ANTECHAMBER to reorder atoms
        write(tempfile,atoms,format='xyz')
        subprocess.run(["/datos_pool/mldata1/QMdatasets/OMol25/AIM4ML/scripts/xyz_reord.sh",tempfile])
        try:
            ase_frag_reord=read(tempfile)
        except Exception:
            continue 
        atoms_reord=read(tempfile)
        os.remove(tempfile)
        if ( isngl == 0 ):
             write(output_single,atoms_reord,comment=info,format='extxyz')
        else:
             write(output_single,atoms_reord,append=True,comment=info,format='extxyz')
        isngl=isngl+1



