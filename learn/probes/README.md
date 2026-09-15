# Probes

Small scripts that settle behavioral questions by running the installed Python module.
Each prints its own expectation next to the measured value. They locate the checkout's
`install/` prefix themselves; set `DR_EVT_INSTALL` to point elsewhere.

```bash
python3 learn/probes/probe_exclusive.py
python3 learn/probes/probe_composite.py
```

- `probe_exclusive.py`: `Simulation.run_until_exclusive` does not advance past an end
  event that lies strictly before the target, and fractional-second timestamps can be
  missed by `advance_to` because the event time is stored as a float fraction (a job
  appended at 10.123 does not start on `advance_to(10.123)`).
- `probe_composite.py`: submitting every fragment of a composite job only when each
  fragment is predicted to start now yields a deterministic simultaneous start under
  EASY; a waiting head on any platform is the signal to defer.
- `probe_custom_scheduler.py`: the custom FCFS scheduler with an arrival-order cost and a
  lowest-cost selector reproduces standard EASY exactly, and `get_prediction_horizon()` is
  an estimate of when the whole waiting queue drains, not of when one job would start.
