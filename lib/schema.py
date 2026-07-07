"""
schema.py — Canonical SDF property tag definitions for the AIM4ML curation pipeline.

Every stage references these constants. A single source of truth so changing
a tag name or type only requires an edit here.
"""

import hashlib

# -- Required tags (pipeline will reject molecules missing these) ----------

REQUIRED_TAGS = {
    "Energy_Ha":    float,
    "FormalCharge": int,
    "Multiplicity": int,
}

# -- Recommended tags (pipeline warns but does not reject) -----------------

RECOMMENDED_TAGS = {
    "SMILES":   str,
    "SourceID": str,
}

# -- Optional pass-through tags (validated if present, ignored if absent) ---

OPTIONAL_TAGS = {
    "HOMO_Ha":        float,
    "LUMO_Ha":        float,
    "HL_Gap_Ha":      float,
    "PartialCharges": str,          # space-separated floats
}

# -- Pipeline-computed tags (added by stages, not present in input SDF) -----

COMPUTED_TAGS = {
    "CompoundID":       str,
    "CanonicalSMILES":  str,
    "ICONF":            int,
    "sdf_status":       str,
    "stereo_status":    str,
    "MolWt":            float,
    "TPSA":             float,
    "logP":             float,
    "nrot":             int,
}

# -- Utility ---------------------------------------------------------------

def compound_id(canonical_smiles):
    """
    CompoundID = MD5(canonical SMILES).
    Stable, deterministic, SMILES-defined molecular identity.
    """
    return hashlib.md5(canonical_smiles.encode("utf-8")).hexdigest()


# -- Parquet schema ------------------------------------------------------
# Column names and dtypes for the internal Parquet batch files.

PARQUET_COLUMNS = {
    "CompoundID":       "string",
    "Energy_Ha":        "float64",
    "FormalCharge":     "int32",
    "Multiplicity":     "int16",
    "SMILES":           "string",
    "SourceID":         "string",
    "HOMO_Ha":          "float64",
    "LUMO_Ha":          "float64",
    "HL_Gap_Ha":        "float64",
    "PartialCharges":   "string",
    "num_atoms":        "int32",
    "num_bonds":        "int32",
    "mol_block":        "string",   # SDF V2000 text block
}

# Columns that are always present (no nulls after split).
PARQUET_REQUIRED = ["mol_block", "num_atoms", "num_bonds",
                    "Energy_Ha", "FormalCharge", "Multiplicity"]

# Optional metadata columns — may be null if tag absent in input.
PARQUET_OPTIONAL = ["SMILES", "SourceID", "CompoundID",
                    "HOMO_Ha", "LUMO_Ha", "HL_Gap_Ha", "PartialCharges"]
