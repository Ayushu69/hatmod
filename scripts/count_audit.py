import pandas as pd, math, re
df = pd.read_csv("results/audit_100.csv", encoding="utf-8")
s = df.my_tag.fillna("").str.strip()
df["tag"] = s.str.replace(r"[?!~\s]", "", regex=True)
for m, n in [("?", "contestable"), ("~", "context"), ("!", "extreme")]:
    df[n] = s.str.contains(re.escape(m))

valid = {"group_hate", "personal_abuse", "political_criticism", "neutral"}
print("rows:", len(df), "| bad/blank tags:\n", df[~df.tag.isin(valid)][["my_tag"]])

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return round(c-h, 2), round(c+h, 2)

print("\ntag counts (95% CI):")
for t, k in df.tag.value_counts().items(): print(f"  {t:20s} {k:3d}  {wilson(k, len(df))}")
print("\nflags:", df[["contestable", "context", "extreme"]].sum().to_dict())
print(df.groupby("tag")[["contestable", "context", "extreme"]].sum())

P = "mean_prob"   # change to the model-probability column name in your file
if P in df: print(df.groupby("tag")[P].agg(["count", "mean", "median"]).round(3))

th = 0.746   # the ensemble threshold from Stage 5
df["flagged"] = df.mean_prob >= th
print(df.groupby("tag").flagged.agg(["sum", "count", "mean"]).round(2))
conf = df[~df.contestable & ~df.context]
print(conf.groupby("tag").flagged.agg(["sum", "count", "mean"]).round(2))