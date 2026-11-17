import csv, sys, json, os
R = sys.argv[1]
def rows(p):
    with open(os.path.join(R, p), newline='') as f:
        return list(csv.DictReader(f))
def f(x, n=4):
    try: return f"{float(x):.{n}f}"
    except: return x
print("## size_only_reference")
for r in rows("reports/legacy/size_only_reference.csv"):
    print(r['phase'], r['k'], r['available'], f(r['mean']), f(r['sd']), f(r['p05']), f(r['p95']))
print("## support_draw_summary (legacy)")
for r in rows("reports/legacy/support_draw_summary.csv"):
    print(r['arm'], r['classifier'], r['k'], r['available'], f(r['mean']), f(r['sd']), f(r['p05']), f(r['p95']))
print("## paired (legacy) vs Inception / all prototypes")
for r in rows("reports/legacy/paired_differences.csv"):
    print(r['arm_a'], r['arm_b'], r['role'], r['classifier'], r['k'], r['available_pairs'], f(r['mean_a_minus_b']), f(r['sd']), f(r['p05']), f(r['p95']))
print("## source audit")
for r in rows("reports/external/external_source_audit.csv"):
    print(r['arm'], r['source'], r['endpoint'], r['numerator'], r['denominator'], f(r['value']), f(r['interval_low']), f(r['interval_high']), r['interval_method'][:40])
print("## two-way")
for r in rows("reports/external/external_two_way.csv"):
    print(r['arm'], r['role'], r['k'], r['n_CSC'], r['n_MH'], f(r['macro_mean']), f(r['macro_sd']), f(r['macro_p05']), f(r['macro_p95']), f(r['recall_CSC_mean']), f(r['recall_MH_mean']), r['collapsed_draws'])
