import pandas as pd
import re

df = pd.read_csv("country_second_pass_input.csv")

def extract_acl_id(url):
    if pd.isna(url):
        return ""
    match = re.search(r'aclanthology\.org/([^/]+)/?', url)
    return match.group(1) if match else ""

df["url"] = df["Paper Link (URL)"]
#print(df["url"].head())

# for url in df["url"].head():
#     if "aclanthology.org" in url and not url.lower().endswith(".pdf"):
#         print(url.rstrip("/") + ".pdf")
df["paper_id"] = df["url"].apply(extract_acl_id)
#print(df["paper_id"].head())
df.to_csv("country_second_pass_manifest.csv", index=False)

print("Saved fixed manifest")