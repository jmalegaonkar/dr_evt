# learn/: exploring dr_evt as the execution layer of a federated market

Five executed notebooks and two probe scripts written while designing a federated
market client on top of dr_evt: a controller that owns arrivals and the clock, decides
which cluster each job goes to and what it pays, and submits it to that cluster's
`Simulation` through the streaming API, in process or over gRPC. They record what we
learned about dr_evt itself along the way, with every claim demonstrated by running it.

| notebook | what it shows |
|---|---|
| `01_repo_tour.ipynb` | the one-cluster model, the source layout, building with CMake 4, the CLI, the input and output formats, an EASY schedule walked by hand, a policy comparison |
| `02_python_streaming_api.ipynb` | `append_job` and `advance_to`, the append-then-advance rule, the backfill window stepped through a trace, what `Statistics` does and does not mean, the limits of the in-process API, several simulations in one process, determinism |
| `03_client_server.ipynb` | starting `dr_evt_server`, sessions through the generated stubs and `ServerSession`, per-job records from the session CSV, error responses, two sessions as two platforms and a composite job that partial-starts |
| `04_mini_federation.ipynb` | four platforms with different node counts and speeds, a start-time predictor from the backfill window, two routers, a controller loop, service metrics from records |
| `05_composite_jobs.ipynb` | dr_evt's composite-fragment format, coordinator protocol and `partial_start` observation, its test oracle, and the one gate that makes partial starts impossible when a single controller submits |

## Running them

Build the checkout with Python bindings and gRPC (the recipe is in notebook 01) so that
`install/bin/dr_evt_server` and `install/lib/python/dr_evt.*.so` exist, then, from this
directory:

```bash
python3 -m pip install grpcio grpcio-tools protobuf pandas matplotlib jupyter nbconvert
for nb in 01_repo_tour 02_python_streaming_api 03_client_server 04_mini_federation 05_composite_jobs; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

The notebooks find the repository root by walking up from the current directory and use
`install/` under it; set `DR_EVT_INSTALL` to use another prefix. Scratch files, server
working directories and generated stubs go under `learn/output/`, which is gitignored.
Each notebook starts its own server on a free port and stops it at the end.

The outputs saved in the notebooks were produced at the commit they were committed on;
the queue input mode, fixture columns and CLI output they show are those of that commit.

## Probes

`probes/` holds two scripts that settle specific behavioral questions by running the
installed module; see `probes/README.md`.
