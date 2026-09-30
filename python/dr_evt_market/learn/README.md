# Market guides and notebooks

`master_overview.md` walks through the market end to end. `01_jobs.md` details step 1,
from traces to the jobs file, `02_bids.md` step 2, the bids, and `03_platforms.md` step
3, the platforms.

`market.ipynb` explains the fixture market from bids through final outputs.
`lc_demo.ipynb` runs the local three-source LC trace study and remains ignored.

Run either notebook from this directory with the project environment:
```bash
$HOME/Desktop/AuctionMVP/.venv/bin/jupyter nbconvert --to notebook --execute \
  --inplace market.ipynb
```
