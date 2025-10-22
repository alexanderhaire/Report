# gp_dynamic_learner.py
import json, os, subprocess, sys, pandas as pd
PY = sys.executable

CHEM_CSV   = "chemicals.csv"      # produced by export_gp_prices.py
FUT_CSV    = "futures.csv"        # keep this updated (settles); same tidy format
OUTDIR     = "out"
SYMBOLS    = ["KTS", "CN", "CAN", "UREA"]   # put your item codes here
MODEL_DIR  = os.path.join(OUTDIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def run(*args):
    print("•", " ".join(args))
    cp = subprocess.run([PY, *args], capture_output=True, text=True)
    if cp.returncode != 0:
        print(cp.stdout)
        raise SystemExit(cp.stderr)
    print(cp.stdout.strip())

def main():
    # 1) Export (incremental) from GP -> chemicals.csv
    run("export_gp_prices.py")

    # 2) Build aligned dataset (chem + futures)
    run("chem_futures_linkage.py", "prep",
        "--chem", CHEM_CSV, "--fut", FUT_CSV, "--outdir", OUTDIR)

    # 3) (Optional) updated correlations
    run("chem_futures_linkage.py", "corr",
        "--dataset", f"{OUTDIR}/dataset.parquet", "--outdir", OUTDIR)

    # 4) Keep training each product model (SGD partial_fit under the hood)
    for sym in SYMBOLS:
        run("chem_futures_linkage.py", "train",
            "--dataset", f"{OUTDIR}/dataset.parquet",
            "--symbol", sym, "--horizon", "1D", "--lags", "5",
            "--outdir", OUTDIR,
            "--model", os.path.join(MODEL_DIR, f"model_{sym}.pkl"))

if __name__ == "__main__":
    main()
