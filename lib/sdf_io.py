"""
sdf_io.py — Read, validate, and split SDF files for the AIM4ML curation pipeline.

Public functions:
    read_sdf(path)        → list[(mol, tag_dict)]
    validate_tags(mols, tags_dict, strict) → passed, rejected, warnings
"""

import os
from collections import OrderedDict

from rdkit import Chem


# -- Reading --------------------------------------------------------------

def read_sdf(path):
    """
    Read an SDF file and return (RDKit Mol, tag dict) for every molecule.

    Parameters
    ----------
    path : str
        Path to the input SDF file.

    Returns
    -------
    list of (Chem.Mol, dict)
        Each mol paired with its raw {tagname: value} dictionary.
        Mols that RDKit cannot parse are None — skipped by SDMolSupplier
        (sanitize=True by default).
    """
    supplier = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    entries = []
    for mol in supplier:
        if mol is None:
            entries.append((None, {}))
            continue
        tags = {prop: mol.GetProp(prop) for prop in mol.GetPropNames()}
        entries.append((mol, tags))
    return entries


# -- Reject writer (shared by all stages) --------------------------------

_REJECT_TAG_ORDER = [
    "SourceID", "SMILES", "Energy_Ha", "FormalCharge", "Multiplicity",
    "HOMO_Ha", "LUMO_Ha", "HL_Gap_Ha", "PartialCharges",
    "num_atoms", "num_bonds", "energy_status",
]


def write_reject_sdf(path, rejected_rows, reject_reason=""):
    """
    Write a list of rejected-molecule rows as an SDF file for human inspection.

    Each row (dict from a Parquet batch) is reconstructed into an RDKit Mol
    via its mol_block, tagged with metadata columns (excluding mol_block
    itself), and written to SDF.

    Parameters
    ----------
    path : str
        Output SDF path.
    rejected_rows : list of dict
        Rows from Parquet batches. Must each contain a 'mol_block' key.
    reject_reason : str
        Human-readable reason written into the 'reject_reason' SDF tag.
    """
    if not rejected_rows:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with Chem.SDWriter(path) as writer:
        for row in rejected_rows:
            mol = Chem.MolFromMolBlock(row["mol_block"], sanitize=False,
                                       removeHs=False)
            if mol is None:
                continue
            # Re-attach metadata as SDF tags, skipping mol_block and None values
            for tag in _REJECT_TAG_ORDER:
                if tag == "mol_block":
                    continue
                val = row.get(tag)
                if val is not None:
                    mol.SetProp(tag, str(val))
            # Add any remaining keys not in the order list
            for tag, val in row.items():
                if tag == "mol_block" or tag in _REJECT_TAG_ORDER:
                    continue
                if val is not None:
                    mol.SetProp(tag, str(val))
            if reject_reason:
                mol.SetProp("reject_reason", reject_reason)
            writer.write(mol)


# -- Validation -----------------------------------------------------------

def validate_tags(mols_and_tags, required, recommended, optional,
                  strict=True):
    """
    Validate property tags against the pipeline contract.

    Parameters
    ----------
    mols_and_tags : list of (Chem.Mol, dict)
        Output from read_sdf.
    required : dict[tag, type]
        Required tags and their expected Python type.
    recommended : dict[tag, type]
        Recommended tags; warn if missing, no rejection.
    optional : dict[tag, type]
        Optional tags; if present, validate type; if absent, ignore.
    strict : bool
        If True, missing/wrong-type required tags cause rejection.
        If False, only warn.

    Returns
    -------
    passed : OrderedDict[int, (Chem.Mol, dict)]
        Molecules that pass all checks, indexed by entry position.
    rejected : OrderedDict[int, (Chem.Mol, dict, list of str)]
        Molecules that fail, together with error messages.
    warnings : OrderedDict[int, list of str]
        Warnings keyed by entry position (includes molecules that may also
        appear in passed or rejected).
    """
    passed   = OrderedDict()
    rejected = OrderedDict()
    warnings = OrderedDict()

    # Combine tag specs; required overrides recommended/optional in status
    all_tags = {}
    for name, pytype in required.items():
        all_tags[name] = (pytype, "required")
    for name, pytype in recommended.items():
        all_tags.setdefault(name, (pytype, "recommended"))
    for name, pytype in optional.items():
        all_tags.setdefault(name, (pytype, "optional"))

    for idx, (mol, tags) in enumerate(mols_and_tags):
        if mol is None:
            # RDKit could not parse the entry at all
            rejected[idx] = (mol, tags, ["RDKit: could not parse molecule"])
            continue

        errors   = []
        warns    = []

        for tag_name, (pytype, status) in all_tags.items():
            if tag_name not in tags:
                msg = f"{tag_name}: missing ({status})"
                if status == "required" and strict:
                    errors.append(msg)
                else:
                    warns.append(msg)
                continue

            value = tags[tag_name]
            # Multiplicity: must be integer > 0
            if tag_name == "Multiplicity":
                try:
                    mult = int(value)
                    if mult <= 0:
                        errors.append(
                            f"Multiplicity: {value} is not a positive integer"
                        )
                except (ValueError, TypeError):
                    errors.append(
                        f"Multiplicity: '{value}' is not parseable as integer"
                    )
                continue

            # FormalCharge: must be integer
            if pytype is int and tag_name == "FormalCharge":
                try:
                    int(value)
                except (ValueError, TypeError):
                    errors.append(
                        f"{tag_name}: '{value}' is not parseable as integer"
                    )
                continue

            # Float types
            if pytype is float:
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(
                        f"{tag_name}: '{value}' is not parseable as float"
                    )
                continue

        # Unknown tags — pass through silently (no error, no warning)
        if errors:
            rejected[idx] = (mol, tags, errors)
        else:
            passed[idx] = (mol, tags)

        if warns:
            warnings[idx] = warns

    return passed, rejected, warnings
