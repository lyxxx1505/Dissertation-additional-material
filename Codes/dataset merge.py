import pandas as pd
import os
from tkinter import Tk, filedialog

# 1. Open a dialog to select the folder
Tk().withdraw()
folder = filedialog.askdirectory(title="Select the folder containing CSV files")

# 2. Read all CSV files in the selected folder
files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
print(f"Found {len(files)} CSV files")

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"Total rows after merging: {len(df)}")

# 3. Deduplicate (based on 'url')
before = len(df)
df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
after = len(df)
print(f"Removed {before - after} duplicate rows")
print(f"Remaining rows after deduplication: {after}")

# 4 Remove rows where 'section' is in unwanted categories
unwanted_sections = [
    "Art and design", "Football", "Games", "Music",
    "Media", "Sport", "Television & radio", "Film", "Fashion"
]
before_filter = len(df)
df = df[~df["section"].isin(unwanted_sections)].reset_index(drop=True)
after_filter = len(df)

print(f"Removed {before_filter - after_filter} rows from unwanted sections")
print(f"Remaining rows after filtering: {after_filter}")

# 5. Export the merged and deduplicated dataset
df.to_csv("guardian_esg_merged_dedup.csv", index=False, encoding="utf-8-sig")
print("Merged and deduplicated data saved to guardian_esg_merged_dedup.csv")
