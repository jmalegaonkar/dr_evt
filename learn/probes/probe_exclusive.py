import sys, tempfile
from pathlib import Path
import os
INSTALL = os.environ.get("DR_EVT_INSTALL", str(Path(__file__).resolve().parents[2] / "install"))   # this checkout's install prefix
sys.path.insert(0, INSTALL + "/lib/python")
import dr_evt
QUEUE = "pbatch" if getattr(dr_evt, "legacy_queue_input", True) else "1"   # numeric queue ids by default
print(f"dr_evt from {INSTALL}, queue value {QUEUE!r}")
hdr = Path(tempfile.gettempdir()) / "dr_evt_header_only.csv"; hdr.write_text("job_submit_time,num_nodes,queue,time_limit\n")
def mk(n=100):
    p = dr_evt.SimParams(); p.infile = str(hdr); p.total_nodes = n; p.trace_format = "simple"; p.timestamp_format = "epoch"
    p.run_time_mode = dr_evt.RunTimeMode.LIMIT; p.backfill_policy = dr_evt.BackfillPolicy.EASY; p.priority_policy = dr_evt.PriorityPolicy.FCFS
    return p, dr_evt.Simulation(p)

print("== run_until_exclusive discriminating test ==")
p, s = mk()
s.append_job(0.0, 10, QUEUE, 10.0)   # A ends at 10
s.append_job(0.0, 10, QUEUE, 30.0)   # B ends at 30
s.advance_to(0.0)
print("after advance_to(0):           t=%s in_use=%s" % (s.get_current_time(), s.get_nodes_in_use()))
s.run_until_exclusive(20.0)
print("after run_until_exclusive(20): t=%s in_use=%s   (functional => t=10,in_use=10; no-op => t=0,in_use=20)" % (s.get_current_time(), s.get_nodes_in_use()))
s.run_until_exclusive(30.0)
print("after run_until_exclusive(30): t=%s in_use=%s   (functional => t=10 still, B's end at 30 excluded)" % (s.get_current_time(), s.get_nodes_in_use()))
s.advance_to(30.0)
print("after advance_to(30):          t=%s in_use=%s" % (s.get_current_time(), s.get_nodes_in_use()))

print("\n== fractional-second (millisecond) precision test ==")
for submit, limit in [(10.123, 5), (10.5, 5), (10.001, 5), (10.7, 5)]:
    p, s = mk()
    s.append_job(submit, 10, QUEUE, float(limit))
    s.advance_to(submit)
    started = s.get_nodes_in_use()
    end = submit + limit
    s.advance_to(end)
    at_end = s.get_nodes_in_use()
    s.advance_to(end + 0.001)
    after = s.get_nodes_in_use()
    print(f"submit={submit}: in_use after advance_to(submit)={started}, after advance_to(end={end})={at_end}, after advance_to(end+0.001)={after}")
