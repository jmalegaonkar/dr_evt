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

def predict_now(sim, nodes, limit, now):
    w = sim.get_backfill_window()
    if w.shadow_time < 0: return nodes <= w.available_nodes, "no_head"
    return (nodes <= w.available_nodes and now + limit < w.shadow_time), "head_waiting"

print("== composite, both legs predicted-now => both start (deterministic) ==")
pa, A = mk(); pb, B = mk()
A.append_job(0.0, 60, QUEUE, 1000.0); B.append_job(0.0, 30, QUEUE, 1000.0)
for s in (A, B): s.advance_to(25.0)
legs = {"A": (A, 30, 120.0), "B": (B, 40, 120.0)}
ok = {k: predict_now(s, n, l, 25.0) for k, (s, n, l) in legs.items()}
print("predictions:", ok)
if all(v[0] for v in ok.values()):
    for k, (s, n, l) in legs.items(): s.append_job(25.0, n, QUEUE, l)
    for s in (A, B): s.advance_to(25.0)
    print("in_use after submit+advance: A=%d (expect 90) B=%d (expect 70) waiting A=%d B=%d" % (A.get_nodes_in_use(), B.get_nodes_in_use(), A.get_active_job_count(), B.get_active_job_count()))

print("\n== composite, one platform has a waiting head => predicted behind_head => defer ==")
pa, A = mk(); pb, B = mk()
A.append_job(0.0, 60, QUEUE, 1000.0); A.append_job(1.0, 50, QUEUE, 100.0)   # 50 > 40 free: head waits, shadow=1000
B.append_job(0.0, 30, QUEUE, 1000.0)
for s in (A, B): s.advance_to(25.0)
ok = {k: predict_now(s, n, l, 25.0) for k, (s, n, l) in {"A": (A, 30, 120.0), "B": (B, 40, 120.0)}.items()}
w = A.get_backfill_window(); print("A window: avail=%d shadow=%s ; predictions: %s" % (w.available_nodes, w.shadow_time, ok))
print("=> policy defers the composite; nothing submitted. If we HAD submitted A's leg (30 nodes, 120s): 25+120 < 1000 so EASY would backfill it -> the predicted-now rule can be extended with the backfill clause safely.")

print("\n== same-window sibling exhaustion: two singles then a composite leg in one window on one platform ==")
pa, A = mk()
A.append_job(0.0, 60, QUEUE, 1000.0); A.advance_to(25.0)
w = A.get_backfill_window(); print("free before window:", w.available_nodes)
# the market must validate cumulative demand against the snapshot: 30 + 10 fit in 40, a third 10-node leg would not
A.append_job(25.0, 30, QUEUE, 120.0); A.append_job(25.0, 10, QUEUE, 120.0)
A.advance_to(25.0)
print("after two submissions totalling 40: in_use=%d waiting=%d (both started; cumulative check is the client's job)" % (A.get_nodes_in_use(), A.get_active_job_count()))
