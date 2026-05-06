import pandas as pd

BASE = "/datos_pool/mldata1/QMdatasets/QM40"

main  = pd.read_csv(f"{BASE}/main.csv")
xyz   = pd.read_csv(f"{BASE}/xyz.csv")
bond  = pd.read_csv(f"{BASE}/bond.csv")

# Take first 200 molecules
ids = main["Zinc_id"].iloc[:200]

# Generate the sample files
main.loc[main["Zinc_id"].isin(ids)].to_csv("sample_main.csv",  index=False)
xyz.loc[xyz["Zinc_id"].isin(ids)].to_csv("sample_xyz.csv",   index=False)
bond.loc[bond["Zinc_id"].isin(ids)].to_csv("sample_bond.csv",  index=False)

print("Done. Rows:", len(ids), "molecules")
