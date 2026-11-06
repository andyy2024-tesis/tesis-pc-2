"""Wind (catch-all noise in the quietest 20% of frames) for given runs, same metric as E9_findings."""
import sys
import pandas as pd
import torch
sys.path.insert(0, "scripts"); sys.path.insert(0, "analysis")
from compare_runs import load, RES
from e6_adversarial import quiet_noise_db
ids = pd.read_csv(RES / "e4_recordings.csv").id.tolist()
for run in sys.argv[1:]:
    m, recs, splits = load(run, ids)
    with torch.no_grad():
        print(run, "wind", round(quiet_noise_db(m, recs, splits), 1), flush=True)
