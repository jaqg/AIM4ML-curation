#!/usr/bin/env python3
"""
Inspect an aselmdb file (ASE database) from the Omol25 dataset.
Processes all entries, prints details only for those that contain only CHONS atoms
and have between 10 and 25 CNSO atoms (C, N, O, S).
"""

import sys
import numpy as np
import ase.db
from ase.io import write
from ase.units import Hartree, Bohr

if len(sys.argv) != 2:
        print("Usage: python filter_oml25.py  <filename.aselmdb>")
        sys.exit(1)

# Database file 
filename=sys.argv[1]
xyzfile=filename.replace('.aselmdb','_selected.xyz')

# Allowed elements and CNSO set
allowed = {'C', 'H', 'O', 'N', 'S'}
cns_elements = {'C', 'N', 'O', 'S'}   # CNSO atoms

# List of attributes of Omol25 entries 
KEYS = [ '_constrained_forces', '_constraints', '_data', '_keys', 'calculator',
   'calculator_parameters', 'cell', 'charge', 'constrained_forces', 'constraints',
   'count_atoms', 'ctime', 'data', 'energy', 'fmax', 'forces', 'formula', 'get',
   'id', 'key_value_pairs', 'mass', 'mtime', 'natoms', 'numbers', 'pbc',
   'positions', 'symbols', 'toatoms', 'unique_id', 'user' , 'volume', 'smax' ]

# Convert eV/Å → Hartree/Bohr
# f=0.0367493
f = (1 / Hartree)
# conversion factor ev/Ang  -> hartree/borh
# fg=0.0194469
fg = (Bohr / Hartree)
# accepted maxforce Hartree/bohr
maxforce=0.01
# Metadata
metadata=np.load('metadata.npz')
# Omol25 datasets to be filtered 
filter_datasets=['elytes', 'metal_complexes', 'trans1x', 'reactivity', 'rgd']

first=True

# Open lmbd file 
with ase.db.connect(filename, use_lock_file=False) as conn:

    # Iterate over all entries in the database
    for mol in conn.select():

        # Get the atomic symbols as a list
        atoms = mol.toatoms()
        symbols = atoms.get_chemical_symbols()

        # Filter 1: All atoms must be CHONS
        if not all(s in allowed for s in symbols):
            continue

        # Filter 2: Number of CNSO atoms between 10 and 25 (inclusive)
        cns_count = sum(1 for s in symbols if s in cns_elements)
        if not (10 <= cns_count <= 25):
            continue
    
        # Filter 3: Maximum force threshold (only reasonably relaxed geometries)
        fmax=getattr(mol,'fmax')*fg
        if  ( fmax > maxforce ):
            continue

        nat=getattr(mol,'natoms')
#       xyz=getattr(mol,'positions')
#       atsym=getattr(mol,'symbols')
        energy=getattr(mol,'energy')*f
        data_dic=getattr(mol,'data')
        q=int ( data_dic['charge']  ) 
        mult=int ( data_dic['spin']  ) 
        source=metadata['data_ids'][mol.id]

        # Filter 4: Discard now ionic and radical species
        if ( abs(q) > 0 ) | ( mult > 1 ):
            continue

        # Filter 5: Subsets involving metals/electrolytes/reactivity not considered
        if source in filter_datasets:   
            continue        
 
        # Convert forces into au 
        forces = atoms.get_forces()  
        atoms.arrays['nbo_charges']= data_dic["nbo_charges"]   # Convert into hartrees bohrs
        atoms.arrays['forces_au']= forces*fg   # Convert into hartrees bohrs
        atoms.calc=None  # Needed for correct printing

        molspec=f'ID,{mol.id},Formula,{mol.formula},nat,{nat},CNSO,{cns_count},chrg,{q},mult,{mult},e,{energy:.5f},fmax,{fmax:.3f},Family,{source}'

        if ( first  ):
            write(xyzfile,atoms,comment=molspec,format='extxyz')
            first = False
        else:
            write(xyzfile,atoms,comment=molspec,append=True,format='extxyz')


