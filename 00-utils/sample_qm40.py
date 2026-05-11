import os
import pandas as pd

SRC  = "/datos_pool/mldata1/QMdatasets/QM40"
DEST = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples"

os.makedirs(DEST, exist_ok=True)

main  = pd.read_csv(f"{SRC}/main.csv")
xyz   = pd.read_csv(f"{SRC}/xyz.csv")
bond  = pd.read_csv(f"{SRC}/bond.csv")

ids = main["Zinc_id"].iloc[:200]

main.loc[main["Zinc_id"].isin(ids)].to_csv(f"{DEST}/sample_main.csv",  index=False)
xyz.loc[xyz["Zinc_id"].isin(ids)].to_csv(f"{DEST}/sample_xyz.csv",   index=False)
bond.loc[bond["Zinc_id"].isin(ids)].to_csv(f"{DEST}/sample_bond.csv",  index=False)

print(f"Done. {len(ids)} molecules → {DEST}/")
