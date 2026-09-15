#!/usr/bin/env python3
################################################################################
#         Copyright 2023 Lawrence Livermore National Security, LLC             #
#         See the top-level LICENSE file for details.                          #
#                                                                              #
#         SPDX-License-Identifier: MIT                                         #
################################################################################

"""
DR_EVT Python API Test Suite

Tests all Python bindings including:
- Configuration parameters
- Streaming API
- Monitoring API
- Statistics
- Different scheduling policies

Note: Scheduler uses time_limit as the best estimator for planning.
run_time_mode is set to LIMIT so jobs run exactly their time_limit.
"""

import csv
import sys
import os
import subprocess
import tempfile

try:
    import dr_evt
except ImportError as e:
    print(f"✗ Failed to import dr_evt module: {e}", file=sys.stderr)
    print("\nBuild Python bindings with:", file=sys.stderr)
    print("  cmake -S . -B build -DDR_EVT_BUILD_PYTHON=ON", file=sys.stderr)
    print("  cmake --build build -j4", file=sys.stderr)
    sys.exit(1)

QUEUE_INPUT = "pbatch" if dr_evt.legacy_queue_input else "1"


class TestResult:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record_pass(self, test_name):
        self.passed += 1
        print(f"  ✓ {test_name}")

    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  ✗ {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{total} passed")
        print(f"{'='*60}")

        if self.errors:
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
            return 1
        else:
            print("\n✅ ALL PYTHON API TESTS PASSED!")
            return 0


def create_test_trace(filename, jobs):
    """Create a test trace file"""
    with open(filename, 'w') as f:
        f.write("job_submit_time,num_nodes,time_limit\n")
        for job in jobs:
            f.write(f"{job[0]},{job[1]},{job[2]}\n")


def find_simulator(repo_root):
    """Find the simulator using the configured or local build prefix."""
    configured_prefix = os.environ.get("CMAKE_INSTALL_PREFIX")
    if configured_prefix:
        candidates = [os.path.join(configured_prefix, "bin", "simulator")]
    else:
        candidates = [
            os.path.join(repo_root, "install", "bin", "simulator"),
            os.path.join(repo_root, "build", "simulator"),
        ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise AssertionError("simulator not found: " + ", ".join(candidates))


def test_module_import(result):
    """Test 1: Module import and version"""
    print("\n1. Module Import")
    try:
        assert hasattr(dr_evt, '__version__')
        result.record_pass(f"Version: {dr_evt.__version__}")
    except Exception as e:
        result.record_fail("Module version", str(e))


def test_streaming_example_from_repo_root(result):
    """The documented invocation must not depend on the caller's cwd."""
    print("\n1b. Streaming Example")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    try:
        completed = subprocess.run(
            [sys.executable, 'python/example_streaming.py'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        assert 'Final Statistics' in completed.stdout
        result.record_pass("Streaming example from repository root")
    except Exception as e:
        result.record_fail("Streaming example from repository root", str(e))


def test_enumerations(result):
    """Test 2: Enumerations"""
    print("\n2. Enumerations")

    # BackfillPolicy
    try:
        assert hasattr(dr_evt, 'BackfillPolicy')
        assert hasattr(dr_evt.BackfillPolicy, 'NONE')
        assert hasattr(dr_evt.BackfillPolicy, 'EASY')
        assert hasattr(dr_evt.BackfillPolicy, 'CONSERVATIVE')
        result.record_pass("BackfillPolicy")
    except Exception as e:
        result.record_fail("BackfillPolicy", str(e))

    # PriorityPolicy
    try:
        assert hasattr(dr_evt, 'PriorityPolicy')
        assert hasattr(dr_evt.PriorityPolicy, 'FCFS')
        assert hasattr(dr_evt.PriorityPolicy, 'FCFS_CONSERVATIVE')
        assert hasattr(dr_evt.PriorityPolicy, 'SJF')
        assert hasattr(dr_evt.PriorityPolicy, 'LJF')
        result.record_pass("PriorityPolicy")
    except Exception as e:
        result.record_fail("PriorityPolicy", str(e))

    # RunTimeMode enum values
    try:
        assert hasattr(dr_evt, 'RunTimeMode')
        assert hasattr(dr_evt.RunTimeMode, 'ACTUAL')
        assert hasattr(dr_evt.RunTimeMode, 'DISTRIBUTION')
        assert hasattr(dr_evt.RunTimeMode, 'LIMIT')
        result.record_pass("RunTimeMode")
    except Exception as e:
        result.record_fail("RunTimeMode", str(e))


def test_sim_params(result):
    """Test 3: SimParams configuration"""
    print("\n3. SimParams Configuration")

    try:
        params = dr_evt.SimParams()

        # Test all exposed parameters
        params.infile = "test.csv"
        params.total_nodes = 100
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT
        params.backfill_policy = dr_evt.BackfillPolicy.EASY
        params.num_max_candidates = 8
        params.priority_policy = dr_evt.PriorityPolicy.FCFS
        params.verbose = False
        params.seed = 42
        params.msec_output = True
        params.run_time_scale = 0.75
        params.run_time_stddev = 0.2
        params.run_time_distribution = dr_evt.DistributionType.LOGNORMAL
        params.outfile = "simulated.csv"
        params.resource_trace = "resource.csv"

        assert params.seed == 42
        assert params.msec_output is True
        assert params.run_time_scale == 0.75
        assert params.run_time_stddev == 0.2
        assert params.run_time_distribution == dr_evt.DistributionType.LOGNORMAL
        assert params.outfile == "simulated.csv"
        assert params.resource_trace == "resource.csv"

        result.record_pass("SimParams creation and configuration")
    except Exception as e:
        result.record_fail("SimParams", str(e))


def test_sim_params_output(result):
    """Test configured simulated-trace filenames and timestamp precision."""
    print("\n3b. SimParams Output Configuration")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        queue_column = "queue" if dr_evt.legacy_queue_input else "q_id"
        with open(trace_file.name, 'w', encoding='utf-8') as output:
            output.write(
                f"job_submit_time,num_nodes,{queue_column},time_limit\n"
            )

        with tempfile.TemporaryDirectory(prefix="dr_evt_sim_params_") as output_dir:
            output_file = os.path.join(output_dir, "simulated.csv")
            params = dr_evt.SimParams()
            params.infile = trace_file.name
            params.total_nodes = 100
            params.trace_format = "simple"
            params.timestamp_format = "epoch"
            params.run_time_mode = dr_evt.RunTimeMode.LIMIT
            params.outfile = output_file
            resource_file = os.path.join(output_dir, "resource.csv")
            params.resource_trace = resource_file

            sim = dr_evt.Simulation(params)
            sim.append_job(0.125, 10, QUEUE_INPUT, 7)
            sim.advance_to(0.125)
            sim.advance_to(7.125)
            sim.write_simulated_trace()
            sim.write_resource_trace(resource_file)

            assert os.path.isfile(output_file)
            with open(output_file, encoding='utf-8') as output:
                header = output.readline().strip().split(',')
            assert header == [
                "job_submit_time", "begin_time", "end_time", "num_nodes",
                "exit_status", queue_column, "time_limit"
            ]
            result.record_pass("Configured outfile writes seven-column trace")

            with open(resource_file, encoding='utf-8') as output:
                resource_rows = [line.strip() for line in output if line.strip()]
            assert "0,90,10" in resource_rows
            assert "7,100,0" in resource_rows
            result.record_pass("Configured resource_trace writes allocation history")

            msec_output_file = os.path.join(output_dir, "simulated_msec.csv")
            msec_params = dr_evt.SimParams()
            msec_params.infile = trace_file.name
            msec_params.total_nodes = 100
            msec_params.trace_format = "simple"
            msec_params.timestamp_format = "epoch"
            msec_params.run_time_mode = dr_evt.RunTimeMode.LIMIT
            msec_params.msec_output = True
            msec_params.outfile = msec_output_file

            msec_sim = dr_evt.Simulation(msec_params)
            msec_sim.append_job(0.125, 10, QUEUE_INPUT, 7)
            msec_sim.advance_to(0.125)
            msec_sim.write_simulated_trace()

            with open(msec_output_file, encoding='utf-8') as output:
                output.readline()
                fields = output.readline().strip().split(',')
            assert fields[:3] == ["0.125", "0.125", "7.125"]
            assert fields[-1] == "7.000"
            result.record_pass("Millisecond output uses three decimals")
    except Exception as e:
        result.record_fail("SimParams output", str(e))
    finally:
        os.unlink(trace_file.name)


def test_run_time_distribution(result):
    """Test seeded runtime sampling in batch mode."""
    print("\n3c. Run-Time Distribution")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        create_test_trace(trace_file.name, [
            (0, 10, 100),
            (0, 10, 100),
            (0, 10, 100),
        ])

        def run_batch(seed, run_time_mode):
            params = dr_evt.SimParams()
            params.infile = trace_file.name
            params.total_nodes = 100
            params.trace_format = "simple"
            params.timestamp_format = "epoch"
            params.seed = seed
            params.run_time_mode = run_time_mode
            params.run_time_distribution = dr_evt.DistributionType.NORMAL
            params.run_time_scale = 0.5
            params.run_time_stddev = 0.2

            sim = dr_evt.Simulation(params)
            sim.initialize_trace()
            sim.run()
            return [
                timing.actual_run_time
                for timing in sim.get_job_timings([0, 1, 2])
            ]

        seed_one_first = run_batch(1, dr_evt.RunTimeMode.DISTRIBUTION)
        seed_one_second = run_batch(1, dr_evt.RunTimeMode.DISTRIBUTION)
        seed_two = run_batch(2, dr_evt.RunTimeMode.DISTRIBUTION)
        limit_times = run_batch(1, dr_evt.RunTimeMode.LIMIT)

        assert seed_one_first == seed_one_second
        assert seed_one_first != seed_two
        assert all(0.0 < run_time <= 100.0
                   for run_time in seed_one_first + seed_two)
        assert limit_times == [100.0, 100.0, 100.0]
        result.record_pass("Seeded run-time distribution and LIMIT mode")
    except Exception as e:
        result.record_fail("Run-time distribution", str(e))
    finally:
        os.unlink(trace_file.name)


def test_streaming_api(result):
    """Test 4: Streaming API"""
    print("\n4. Streaming API")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        # Create test trace
        create_test_trace(trace_file.name, [
            (0, 10, 100),    # Job 0: t=0, 10 nodes, 100s
            (50, 20, 100),   # Job 1: t=50, 20 nodes, 100s
        ])

        # Configure
        params = dr_evt.SimParams()
        params.infile = trace_file.name
        params.total_nodes = 100
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT
        params.backfill_policy = dr_evt.BackfillPolicy.EASY
        params.num_max_candidates = 2
        params.priority_policy = dr_evt.PriorityPolicy.FCFS

        # Create simulation
        sim = dr_evt.Simulation(params)
        # append_job() is the public streaming entry point.
        sim.append_job(0.0, 10, QUEUE_INPUT, 100)
        sim.advance_to(0.0)
        assert sim.get_nodes_in_use() == 10
        result.record_pass("append_job and advance_to")

        # Test run_until_exclusive
        sim.append_job(50.0, 20, QUEUE_INPUT, 100)
        sim.run_until_exclusive(50.0)
        # Job 1 must NOT have started yet - the event at exactly the
        # target time is excluded by run_until_exclusive.
        assert sim.get_nodes_in_use() == 10, \
            f"run_until_exclusive(50.0) should not process the t=50 event yet, but nodes_in_use={sim.get_nodes_in_use()}"
        sim.advance_to(50.0)
        assert sim.get_nodes_in_use() == 30
        result.record_pass("run_until_exclusive")

    except Exception as e:
        result.record_fail("Streaming API", str(e))
    finally:
        os.unlink(trace_file.name)


def test_monitoring_api(result):
    """Test 5: Monitoring API"""
    print("\n5. Monitoring API")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        create_test_trace(trace_file.name, [(0, 30, 100)])

        params = dr_evt.SimParams()
        params.infile = trace_file.name
        params.total_nodes = 100
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT

        sim = dr_evt.Simulation(params)
        # Initial state
        assert sim.get_current_time() == 0.0
        assert sim.get_nodes_in_use() == 0
        assert sim.get_available_nodes() == 100
        try:
            sim.get_resource_area()
            raise AssertionError("standard scheduler exposed live resource area")
        except RuntimeError:
            pass
        result.record_pass("Initial state monitoring")

        # After job starts
        sim.append_job(0.0, 30, QUEUE_INPUT, 100)
        sim.advance_to(0.0)
        assert sim.get_nodes_in_use() == 30
        assert sim.get_available_nodes() == 70
        assert abs(sim.get_current_utilization() - 0.3) < 1e-12
        result.record_pass("Active state monitoring")

        # Queue status
        queue_size = sim.get_active_job_count()
        shadow_time = sim.get_fcfs_head_shadow_time()
        result.record_pass("Queue status API")

    except Exception as e:
        result.record_fail("Monitoring API", str(e))
    finally:
        os.unlink(trace_file.name)


def test_backfill_window_api(result):
    """The in-process Python API exposes the FCFS/EASY reservation snapshot."""
    print("\n5b. Backfill Window API")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        # The 100-node queue head must wait for both running jobs.  Their
        # time limits produce the same releases used for its EASY reservation.
        create_test_trace(trace_file.name, [
            (0, 40, 50),
            (0, 60, 100),
            (0, 100, 10),
        ])

        params = dr_evt.SimParams()
        params.infile = trace_file.name
        params.total_nodes = 100
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT
        params.backfill_policy = dr_evt.BackfillPolicy.EASY
        params.priority_policy = dr_evt.PriorityPolicy.FCFS

        sim = dr_evt.Simulation(
            params,
            lambda job_id, _submit, _runtime, _nodes: job_id,
            lambda candidates: candidates[0][0] if candidates else None,
        )
        for num_nodes, limit_time in [(40, 50), (60, 100), (100, 10)]:
            sim.append_job(0.0, num_nodes, QUEUE_INPUT, limit_time)
            sim.advance_to(0.0)

        window = sim.get_backfill_window()
        assert window.current_time == 0.0
        assert window.available_nodes == 0
        assert window.shadow_time == 100.0
        assert [(release.time, release.nodes_released) for release in window.releases] == [
            (50.0, 40), (100.0, 60)
        ]
        # No running-job completion remains after the shadow event, so the
        # 1000 node-seconds are drained at U * total_nodes = 50 nodes.
        assert abs(sim.get_prediction_horizon(0.5) - 20.0) < 1e-12
        result.record_pass("Backfill window snapshot")
    except Exception as e:
        result.record_fail("Backfill window API", str(e))
    finally:
        os.unlink(trace_file.name)


def test_custom_backfill_api(result):
    """Cost and selection callbacks drive the custom EASY scheduler."""
    print("\n5c. Custom Backfill API")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        create_test_trace(trace_file.name, [])
        params = dr_evt.SimParams()
        params.infile = trace_file.name
        params.total_nodes = 100
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT
        params.backfill_policy = dr_evt.BackfillPolicy.EASY
        params.num_max_candidates = 2

        costed_jobs = []
        candidate_windows = []

        def compute_cost(job_id, submit_time, runtime, nodes):
            costed_jobs.append(job_id)
            return job_id

        def select_lowest_cost(candidates):
            candidate_windows.append(candidates)
            return min(candidates, key=lambda candidate: candidate[1])[0]

        sim = dr_evt.Simulation(params, compute_cost, select_lowest_cost)
        for nodes, runtime in [(70, 100), (50, 200), (20, 50),
                               (10, 20), (10, 30)]:
            sim.append_job(0.0, nodes, QUEUE_INPUT, runtime)
        sim.advance_to(0.0)

        assert costed_jobs == [0, 1, 2, 3, 4]
        assert candidate_windows[0] == [(2, 2), (3, 3)]
        assert select_lowest_cost([(7, 4), (8, 2), (9, 2)]) == 8
        result.record_pass("Custom cost and selection callbacks")
    except Exception as e:
        result.record_fail("Custom backfill API", str(e))
    finally:
        os.unlink(trace_file.name)


def test_job_timing_api(result):
    """Test Python job timing accessors and CLI trace parity."""
    print("\n5c. Job Timing API")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        create_test_trace(trace_file.name, [
            (0, 10, 10),
            (0, 20, 20),
        ])

        params = dr_evt.SimParams()
        params.infile = trace_file.name
        params.total_nodes = 100
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT
        params.backfill_policy = dr_evt.BackfillPolicy.EASY
        params.priority_policy = dr_evt.PriorityPolicy.FCFS

        sim = dr_evt.Simulation(params)
        job0 = sim.append_job(0.0, 10, QUEUE_INPUT, 10)
        job1 = sim.append_job(0.0, 20, QUEUE_INPUT, 20)

        unscheduled = sim.get_job_timing(job0)
        assert isinstance(unscheduled, dr_evt.JobTiming)
        assert repr(unscheduled).startswith("JobTiming(")
        assert unscheduled.job_idx == job0
        assert unscheduled.submit_time == 0.0
        assert unscheduled.begin_time == -1.0
        assert unscheduled.end_time == -1.0
        assert unscheduled.limit_time == 10
        assert unscheduled.actual_run_time == 0.0
        assert unscheduled.num_nodes == 10
        assert unscheduled.scheduled is False
        result.record_pass("Unscheduled JobTiming")

        sim.advance_to(0.0)
        scheduled = sim.get_job_timing(job0)
        assert scheduled.begin_time == 0.0
        assert scheduled.end_time == 10.0
        assert scheduled.actual_run_time == 10.0
        assert scheduled.scheduled is True
        result.record_pass("Scheduled JobTiming")

        timings = sim.get_job_timings([job1, job0])
        assert [timing.job_idx for timing in timings] == [job1, job0]
        assert timings[0].begin_time == 0.0
        assert timings[0].end_time == 20.0
        result.record_pass("Batch JobTiming order")

        try:
            sim.get_job_timing(99)
        except IndexError as error:
            assert "99" in str(error)
        else:
            raise AssertionError("unknown job identifier did not raise IndexError")
        result.record_pass("Unknown JobTiming raises IndexError")

        sim.advance_to(20.0)
        sim.flush_completed_jobs()
        try:
            sim.get_job_timing(job0)
        except IndexError as error:
            assert str(job0) in str(error)
        else:
            raise AssertionError("reclaimed job identifier remained available")
        result.record_pass("Reclaimed JobTiming raises IndexError")

        nodes = [20, 30, 15, 40, 25, 10, 35, 60, 20, 45]
        limits = [200, 150, 300, 100, 250, 80, 180, 220, 90, 160]
        jobs = [(index * 10, nodes[index], limits[index])
                for index in range(10)]
        expected_begin_times = [0, 10, 20, 160, 160, 50, 260, 410, 260, 630]
        create_test_trace(trace_file.name, jobs)
        stream_params = dr_evt.SimParams()
        stream_params.infile = trace_file.name
        stream_params.total_nodes = 100
        stream_params.trace_format = "simple"
        stream_params.timestamp_format = "epoch"
        stream_params.run_time_mode = dr_evt.RunTimeMode.LIMIT
        stream_params.backfill_policy = dr_evt.BackfillPolicy.EASY
        stream_params.priority_policy = dr_evt.PriorityPolicy.FCFS

        stream_sim = dr_evt.Simulation(stream_params)
        job_idxs = []
        for submit_time, num_nodes, limit_time in jobs:
            job_idxs.append(stream_sim.append_job(
                float(submit_time), num_nodes, QUEUE_INPUT, limit_time))
            stream_sim.advance_to(float(submit_time))
        stream_sim.advance_to(1000.0)
        stream_timings = stream_sim.get_job_timings(job_idxs)
        actual_begin_times = [timing.begin_time for timing in stream_timings]
        assert actual_begin_times == [float(begin_time)
                                      for begin_time in expected_begin_times], \
            f"begin times: {actual_begin_times}"

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        simulator = find_simulator(repo_root)
        with tempfile.TemporaryDirectory(prefix="dr_evt_job_timing_") as output_dir:
            simulated_file = os.path.join(output_dir, "simulated.csv")
            command = [
                simulator,
                trace_file.name,
                "--total_nodes", "100",
                "--trace_format", "simple",
                "--timestamp_format", "epoch",
                "--run_time_mode", "limit",
                "--backfill_policy", "easy",
                "--priority_policy", "fcfs",
                "--outfile", simulated_file,
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False)
            assert completed.returncode == 0, \
                completed.stderr or completed.stdout
            with open(simulated_file, newline='', encoding='utf-8') as output:
                output_rows = list(csv.DictReader(output))
            assert len(output_rows) == len(stream_timings)
            for timing, row in zip(stream_timings, output_rows):
                assert timing.scheduled is True
                assert timing.submit_time == float(row["job_submit_time"])
                assert timing.begin_time == float(row["begin_time"])
                assert timing.end_time == float(row["end_time"])
                assert timing.num_nodes == int(row["num_nodes"])
                assert timing.limit_time == int(row["time_limit"])
                assert timing.actual_run_time == float(row["time_limit"])
        result.record_pass("Ten-job timing matches simulator CSV")

    except Exception as e:
        result.record_fail("Job timing API", str(e))
    finally:
        os.unlink(trace_file.name)


def test_statistics(result):
    """Test 6: Statistics"""
    print("\n6. Statistics API")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        create_test_trace(trace_file.name, [
            (0, 10, 50),
            (10, 20, 50),
            (20, 30, 50),
        ])

        params = dr_evt.SimParams()
        params.infile = trace_file.name
        params.total_nodes = 100
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT

        sim = dr_evt.Simulation(
            params,
            lambda job_id, _submit, _runtime, _nodes: job_id,
            lambda candidates: candidates[0][0] if candidates else None,
        )
        # Run complete simulation
        sim.advance_to(0.0)
        sim.append_job(0.0, 10, QUEUE_INPUT, 50)
        sim.advance_to(0.0)

        sim.append_job(10.0, 20, QUEUE_INPUT, 50)
        sim.advance_to(10.0)

        sim.append_job(20.0, 30, QUEUE_INPUT, 50)
        sim.advance_to(20.0)

        sim.advance_to(100.0)

        # Get statistics
        stats = sim.get_statistics()

        # Check all fields exist
        assert hasattr(stats, 'jobs_submitted')
        assert hasattr(stats, 'jobs_completed')
        assert hasattr(stats, 'jobs_running')
        assert hasattr(stats, 'jobs_waiting')
        assert hasattr(stats, 'current_time')
        assert hasattr(stats, 'total_nodes')
        assert hasattr(stats, 'nodes_in_use')
        assert hasattr(stats, 'nodes_available')
        assert hasattr(stats, 'resource_area')
        assert hasattr(stats, 'utilization')
        assert hasattr(stats, 'avg_wait_time')
        assert hasattr(stats, 'avg_turnaround_time')
        assert hasattr(stats, 'makespan')

        # Check values make sense
        assert stats.jobs_completed == 3
        assert stats.total_nodes == 100
        # 10 nodes for 50 s, 20 nodes for 50 s, and 30 nodes for 50 s.
        assert abs(sim.get_resource_area() - 3000.0) < 1e-12
        assert abs(stats.resource_area - 3000.0) < 1e-12
        assert abs(stats.utilization - (3000.0 / (100.0 * 70.0))) < 1e-12

        result.record_pass("Statistics fields and values")

    except Exception as e:
        result.record_fail("Statistics", str(e))
    finally:
        os.unlink(trace_file.name)


def test_backfill_policies(result):
    """Test 7: Different backfill policies"""
    print("\n7. Backfill Policies")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        create_test_trace(trace_file.name, [(0, 50, 100), (10, 30, 50)])

        for policy in [dr_evt.BackfillPolicy.NONE,
                       dr_evt.BackfillPolicy.EASY,
                       dr_evt.BackfillPolicy.CONSERVATIVE]:
            params = dr_evt.SimParams()
            params.infile = trace_file.name
            params.total_nodes = 100
            params.trace_format = "simple"
            params.timestamp_format = "epoch"
            params.run_time_mode = dr_evt.RunTimeMode.LIMIT
            params.backfill_policy = policy

            sim = dr_evt.Simulation(params)
            sim.append_job(0.0, 50, QUEUE_INPUT, 100)
            sim.advance_to(0.0)
            sim.append_job(10.0, 30, QUEUE_INPUT, 50)
            sim.advance_to(200.0)

            stats = sim.get_statistics()
            assert stats.jobs_completed == 2

        result.record_pass("NONE, EASY, CONSERVATIVE policies")

    except Exception as e:
        result.record_fail("Backfill policies", str(e))
    finally:
        os.unlink(trace_file.name)


def test_priority_policies(result):
    """Test 8: Different priority policies"""
    print("\n8. Priority Policies")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        create_test_trace(trace_file.name, [
            (0, 10, 100),   # Long job
            (5, 10, 20),    # Short job
            (10, 10, 50),   # Medium job
        ])

        for policy in [dr_evt.PriorityPolicy.FCFS,
                       dr_evt.PriorityPolicy.SJF,
                       dr_evt.PriorityPolicy.LJF]:
            params = dr_evt.SimParams()
            params.infile = trace_file.name
            params.total_nodes = 100
            params.trace_format = "simple"
            params.timestamp_format = "epoch"
            params.run_time_mode = dr_evt.RunTimeMode.LIMIT
            params.priority_policy = policy

            sim = dr_evt.Simulation(params)
            sim.append_job(0.0, 10, QUEUE_INPUT, 100)
            sim.advance_to(0.0)
            sim.append_job(5.0, 10, QUEUE_INPUT, 20)
            sim.advance_to(5.0)
            sim.append_job(10.0, 10, QUEUE_INPUT, 50)
            sim.advance_to(200.0)

            stats = sim.get_statistics()
            assert stats.jobs_completed == 3

        result.record_pass("FCFS, SJF, LJF policies")

    except Exception as e:
        result.record_fail("Priority policies", str(e))
    finally:
        os.unlink(trace_file.name)


def test_batch_mode(result):
    """Test 9: Batch mode API"""
    print("\n9. Batch Mode")

    trace_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    trace_file.close()

    try:
        create_test_trace(trace_file.name, [(0, 10, 50), (10, 20, 50)])

        params = dr_evt.SimParams()
        params.infile = trace_file.name
        params.total_nodes = 100
        params.trace_format = "simple"
        params.timestamp_format = "epoch"
        params.run_time_mode = dr_evt.RunTimeMode.LIMIT

        sim = dr_evt.Simulation(params)
        sim.initialize_trace()

        # Run entire simulation at once
        sim.run()

        stats = sim.get_statistics()
        assert stats.jobs_completed == 2
        assert stats.nodes_in_use == 0, \
            f"All jobs completed but nodes_in_use={stats.nodes_in_use}, expected 0"

        result.record_pass("Batch mode run()")

    except Exception as e:
        result.record_fail("Batch mode", str(e))
    finally:
        os.unlink(trace_file.name)


def main():
    print("="*60)
    print("DR_EVT Python API Test Suite")
    print("="*60)

    result = TestResult()

    # Run all tests
    test_module_import(result)
    test_streaming_example_from_repo_root(result)
    test_enumerations(result)
    test_sim_params(result)
    test_sim_params_output(result)
    test_run_time_distribution(result)
    test_streaming_api(result)
    test_monitoring_api(result)
    test_backfill_window_api(result)
    test_custom_backfill_api(result)
    test_job_timing_api(result)
    test_statistics(result)
    test_backfill_policies(result)
    test_priority_policies(result)
    test_batch_mode(result)

    # Print summary and exit
    return result.summary()


if __name__ == '__main__':
    sys.exit(main())
