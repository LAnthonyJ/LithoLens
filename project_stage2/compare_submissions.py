import pandas as pd
test = pd.read_csv("/data/数据科学实践project/repro_test/submission_test.csv")
best = pd.read_csv("/data/数据科学实践project/runs/final_top86/submission_blend_top86.csv")
d = (test["label"] != best["label"]).sum()
print(f"Differences: {d}/194")
print(f"Test output : {test['label'].sum()} positives")
print(f"Best known : {best['label'].sum()} positives")
if d == 0:
    print("MATCH! Code is reproducible.")
else:
    print(f"MISMATCH - {d} samples differ")
    diff = test[test["label"] != best["label"]]
    for _, r in diff.iterrows():
        bl = best[best['id']==r['id']]['label'].values[0]
        print(f"  {r['id']}: test={r['label']}, best={bl}")
