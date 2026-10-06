# Market guides and notebooks

`master_overview.md` walks through the market end to end, and one page details each
step: `01_jobs.md`, from traces to the jobs file; `02_bids.md`, the bids;
`03_platforms.md`, the platforms; `04_offers.md`, candidates and offers, with one window
followed call by call under all four mechanisms; `05_loop.md`, the market loop, with the
fixture's run window by window; `06_mechanisms.md`, the mechanisms, with their math
and three jobs followed through each; and `07_outputs.md`, the outputs and the command
line.

`market.ipynb` explains the fixture market from bids through final outputs. `lc_demo/`
holds the local three-source LC trace study, one notebook per step, and remains ignored.

Run a notebook from this directory with the package's environment, without storing
execution metadata:
```bash
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ClearMetadataPreprocessor.enabled=True --ExecutePreprocessor.record_timing=False \
  market.ipynb
```
