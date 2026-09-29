# Market guides and notebooks

`master_overview.md` walks through the market end to end, and `01_jobs.md` details step
1, from traces to the jobs file.

`market.ipynb` explains the fixture market from bids through final outputs.
`lc_demo.ipynb` runs the local three-source LC trace study and remains ignored.

Run either notebook from this directory with the project environment:
```bash
$HOME/Desktop/AuctionMVP/.venv/bin/jupyter nbconvert --to notebook --execute \
  --inplace market.ipynb
```
