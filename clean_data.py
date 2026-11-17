"""GeoPulse - cleaning script for device_visits_sample.csv
Run:  python clean_data.py
Output: device_visits_clean.csv and cleaning_log.csv
"""
import pandas as pd, numpy as np

log = []
def record(step, issue, rows_before, rows_after, action):
    log.append({"step": step, "issue": issue, "rows_before": rows_before,
                "rows_after": rows_after, "action": action})

# 1. Load
df = pd.read_csv("device_visits_sample.csv")
stores = pd.read_csv("stores.csv")
print("Loaded:", df.shape)
print("\nMissing values per column:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# 2. Standardize types (timestamps -> datetime)
df["visit_ts"] = pd.to_datetime(df["visit_ts"], errors="coerce")
n = len(df); df = df.dropna(subset=["visit_ts"])
record(1, "Invalid timestamps", n, len(df), "Dropped rows with unparseable timestamp")

# 3. Remove exact duplicates
n = len(df); df = df.drop_duplicates()
record(2, "Duplicate rows", n, len(df), "drop_duplicates()")

# 4. Invalid coordinates: (0,0) is not in Bengaluru -> drop
#    Valid Bengaluru range is roughly lat 12.7-13.3, lon 77.3-77.9
n = len(df)
valid_geo = df["home_lat"].between(12.7, 13.3) & df["home_lon"].between(77.3, 77.9)
df = df[valid_geo]
record(3, "Invalid/out-of-area coordinates", n, len(df), "Dropped rows outside Bengaluru bounding box")

# 5. Negative dwell time is impossible -> treat as missing
neg = (df["dwell_min"] < 0).sum()
df.loc[df["dwell_min"] < 0, "dwell_min"] = np.nan
record(4, f"Negative dwell_min ({neg})", len(df), len(df), "Set to NaN (imputed in next step)")

# 6. Impute missing dwell with median of the store's category
df = df.merge(stores[["store_id", "category"]], on="store_id", how="left")
missing = df["dwell_min"].isna().sum()
df["dwell_min"] = df["dwell_min"].fillna(df.groupby("category")["dwell_min"].transform("median"))
record(5, f"Missing dwell_min ({missing})", len(df), len(df), "Filled with median dwell per store category")

# 7. Outliers: cap dwell at 99th percentile (flag, don't delete)
cap = df["dwell_min"].quantile(0.99)
df["dwell_outlier"] = (df["dwell_min"] > cap).astype(int)
df["dwell_min"] = df["dwell_min"].clip(upper=cap)
record(6, "Extreme dwell times", len(df), len(df), f"Capped at 99th percentile ({cap:.1f} min); flagged in dwell_outlier")

# 8. Referential integrity: every store_id must exist in stores.csv
n = len(df); df = df[df["store_id"].isin(stores["store_id"])]
record(7, "Unknown store_id", n, len(df), "Dropped rows with store_id not in stores.csv")

# 9. Feature engineering
df["date"] = df["visit_ts"].dt.date
df["hour"] = df["visit_ts"].dt.hour
df["day_of_week"] = df["visit_ts"].dt.day_name()
df["is_weekend"] = (df["visit_ts"].dt.dayofweek >= 5).astype(int)

# 10. Final validation
assert df["dwell_min"].isna().sum() == 0
assert (df["dwell_min"] > 0).all()
assert df.duplicated().sum() == 0

df.to_csv("device_visits_clean.csv", index=False)
pd.DataFrame(log).to_csv("cleaning_log.csv", index=False)
print("\nCleaned shape:", df.shape)
print(pd.DataFrame(log).to_string(index=False))