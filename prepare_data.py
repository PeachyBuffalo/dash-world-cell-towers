#!/usr/bin/env python
"""Run the prepare_cell_data notebook logic to convert CSV to parquet."""
import os
import dask.dataframe as dd
import pandas as pd
from pyproj import Transformer

cell_towers_path = os.path.join(os.path.dirname(__file__), "data", "cell_towers.csv")
data_dir = os.path.join(os.path.dirname(__file__), "data")
parquet_path = os.path.join(data_dir, "cell_towers.parq")

print(f"Reading CSV from {cell_towers_path}...")
ddf = dd.read_csv(cell_towers_path)

# Categorize radio
ddf["radio"] = ddf.radio.astype("category")

# Created and updated to datetime integers
ddf["created"] = dd.to_datetime(ddf.created, unit="s").astype("int")
ddf["updated"] = dd.to_datetime(ddf.updated, unit="s").astype("int")

# Filter out outliers created before 2003
ddf = ddf[dd.to_datetime(ddf.created) >= "2003"]

# Convert lon/lat to epsg:3857
transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

def to3857(df):
    x_3857, y_3857 = transformer.transform(df.lat.values, df.lon.values)
    return df.assign(x_3857=x_3857, y_3857=y_3857)

ddf = ddf.map_partitions(to3857)

# Download network info for mcc/mnc
print("Downloading MCC/MNC lookup table...")
html = pd.read_html("https://cellidfinder.com/mcc-mnc")
tables = [t for t in html if "MCC" in str(t.columns)]
mcc_mnc_df = pd.concat(tables).reset_index(drop=True)

# Handle column name variations
if "Operator or brand name" in mcc_mnc_df.columns:
    mcc_mnc_df["Description"] = mcc_mnc_df["Network"].where(
        ~pd.isnull(mcc_mnc_df["Network"]), mcc_mnc_df["Operator or brand name"]
    )
else:
    mcc_mnc_df["Description"] = mcc_mnc_df.get("Network", mcc_mnc_df.iloc[:, 2])
codes = mcc_mnc_df[["MCC", "MNC", "Status", "Description"]].copy()

# Categorize non-numeric columns
for col in codes.columns:
    if codes[col].dtype == "object":
        codes[col] = codes[col].astype("category")

# Merge
print("Merging with network codes...")
ddf_merged = ddf.merge(codes, left_on=["mcc", "net"], right_on=["MCC", "MNC"], how="left")

# Write parquet
os.makedirs(data_dir, exist_ok=True)
print(f"Writing parquet to {parquet_path}...")
ddf_merged.to_parquet(parquet_path, compression="snappy")
print("Done!")
