"""E0: validate the simulator against M/M/c theory and Little's law. Writes results/tables/E0_sim_validation.csv"""
import _common  # noqa
import pandas as pd
from autoscale.config import RESULTS
from autoscale.runner.validate import mmc_check

rows = [mmc_check(c=c, util=u) for c in (1, 2, 4) for u in (0.5, 0.8)]
df = pd.DataFrame(rows)
df["rt_err_%"] = 100 * (df.rt_sim_s - df.rt_theory_s) / df.rt_theory_s
df["L_err_%"] = 100 * (df.L_sim - df.L_littles) / df.L_littles
print(df.round(3).to_string(index=False))
(RESULTS / "tables").mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS / "tables" / "E0_sim_validation.csv", index=False)
assert df["rt_err_%"].abs().max() < 8, "simulator disagrees with M/M/c theory"
print("OK: simulator matches M/M/c theory and Little's law.")
