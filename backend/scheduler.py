# scheduler.py
# ─────────────────────────────────────────────────────────────────────────────
#  SMART PROCESS SCHEDULING SYSTEM — Scheduler Module
#
#  Implements 5 algorithms with proper DAA data structures:
#    1. FCFS           — Queue (FIFO)           — O(n)
#    2. SJF            — Min-Heap               — O(n log n)
#    3. Priority       — Max-Heap               — O(n log n)
#    4. Round Robin    — Circular Queue         — O(n * ceil(bt/q))
#    5. Smart (ML)     — ML-scored sort         — O(n log n)
# ─────────────────────────────────────────────────────────────────────────────

import heapq
import copy
from ml_model import get_rank_score, predict_best_algorithm

# ─────────────────────────────────────────────────────────────────────────────
#  DAA COMPLEXITY REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
ALGORITHM_COMPLEXITY = {
    "FCFS": {
        "time":  "O(n)",
        "space": "O(n)",
        "structure": "Queue (FIFO)",
        "preemptive": False,
        "description": "Processes run in arrival order. Simple but may cause convoy effect."
    },
    "SJF": {
        "time":  "O(n log n)",
        "space": "O(n)",
        "structure": "Min-Heap",
        "preemptive": False,
        "description": "Shortest burst runs first using a min-heap. Optimal average waiting time."
    },
    "Priority": {
        "time":  "O(n log n)",
        "space": "O(n)",
        "structure": "Max-Heap (inverted priority)",
        "preemptive": False,
        "description": "Higher priority (lower number) runs first. Risk of starvation for low-priority."
    },
    "Round Robin": {
        "time":  "O(n * ceil(bt/q))",
        "space": "O(n)",
        "structure": "Circular Queue",
        "preemptive": True,
        "description": "Each process gets a fixed time quantum. Fair for interactive systems."
    },
    "Smart (ML)": {
        "time":  "O(n log n)",
        "space": "O(n)",
        "structure": "ML-scored sort",
        "preemptive": False,
        "description": "Random Forest predicts optimal order based on burst, arrival, type, priority."
    }
}

TIME_QUANTUM = 3   # for Round Robin


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: deep-copy processes so originals are untouched
# ─────────────────────────────────────────────────────────────────────────────
def _copy(processes):
    return [dict(p) for p in processes]


# ─────────────────────────────────────────────────────────────────────────────
#  1. FCFS — First Come First Served
#     DAA: sorts by arrival_time → iterates linearly → O(n)
# ─────────────────────────────────────────────────────────────────────────────
def fcfs(processes):
    procs = sorted(_copy(processes), key=lambda x: x["arrival_time"])
    time, order = 0, 1
    for p in procs:
        start              = max(time, p["arrival_time"])
        p["waiting_time"]  = start - p["arrival_time"]
        p["turnaround_time"]= start + p["burst_time"] - p["arrival_time"]
        p["start_time"]    = start
        p["finish_time"]   = start + p["burst_time"]
        p["order"]         = order
        time               = p["finish_time"]
        order             += 1
    return procs


# ─────────────────────────────────────────────────────────────────────────────
#  2. SJF — Shortest Job First (Non-preemptive)
#     DAA: Min-Heap on burst_time → O(n log n)
# ─────────────────────────────────────────────────────────────────────────────
def sjf(processes):
    procs     = sorted(_copy(processes), key=lambda x: x["arrival_time"])
    heap      = []   # (burst_time, arrival_time, process_dict)
    done      = []
    time      = 0
    idx       = 0
    order     = 1
    n         = len(procs)

    while len(done) < n:
        # Push all processes that have arrived
        while idx < n and procs[idx]["arrival_time"] <= time:
            p = procs[idx]
            heapq.heappush(heap, (p["burst_time"], p["arrival_time"], p))
            idx += 1

        if heap:
            _, _, p            = heapq.heappop(heap)
            start              = max(time, p["arrival_time"])
            p["waiting_time"]  = start - p["arrival_time"]
            p["turnaround_time"]= start + p["burst_time"] - p["arrival_time"]
            p["start_time"]    = start
            p["finish_time"]   = start + p["burst_time"]
            p["order"]         = order
            time               = p["finish_time"]
            order             += 1
            done.append(p)
        else:
            # CPU idle — jump to next arrival
            if idx < n:
                time = procs[idx]["arrival_time"]
    return done


# ─────────────────────────────────────────────────────────────────────────────
#  3. Priority Scheduling (Non-preemptive)
#     DAA: Min-Heap on priority value → O(n log n)
#     priority: 1 = highest, 10 = lowest
# ─────────────────────────────────────────────────────────────────────────────
def priority_schedule(processes):
    procs = sorted(_copy(processes), key=lambda x: x["arrival_time"])
    # Assign default priority if missing
    for p in procs:
        if "priority" not in p or p["priority"] is None:
            p["priority"] = 5

    heap  = []   # (priority, arrival_time, process_dict)
    done  = []
    time  = 0
    idx   = 0
    order = 1
    n     = len(procs)

    while len(done) < n:
        while idx < n and procs[idx]["arrival_time"] <= time:
            p = procs[idx]
            heapq.heappush(heap, (p["priority"], p["arrival_time"], p))
            idx += 1

        if heap:
            _, _, p            = heapq.heappop(heap)
            start              = max(time, p["arrival_time"])
            p["waiting_time"]  = start - p["arrival_time"]
            p["turnaround_time"]= start + p["burst_time"] - p["arrival_time"]
            p["start_time"]    = start
            p["finish_time"]   = start + p["burst_time"]
            p["order"]         = order
            time               = p["finish_time"]
            order             += 1
            done.append(p)
        else:
            if idx < n:
                time = procs[idx]["arrival_time"]
    return done


# ─────────────────────────────────────────────────────────────────────────────
#  4. Round Robin
#     DAA: Circular Queue, time quantum Q → O(n * ceil(bt/Q))
# ─────────────────────────────────────────────────────────────────────────────
def round_robin(processes, quantum=TIME_QUANTUM):
    procs     = sorted(_copy(processes), key=lambda x: x["arrival_time"])
    remaining = {p["process_id"]: p["burst_time"] for p in procs}
    queue     = []
    time      = 0
    idx       = 0
    n         = len(procs)
    finish    = {}
    first_run = {}
    order     = 1
    gantt     = []   # for timeline

    # Enqueue first batch
    while idx < n and procs[idx]["arrival_time"] <= time:
        queue.append(procs[idx])
        idx += 1

    if not queue and procs:
        time = procs[0]["arrival_time"]
        queue.append(procs[0])
        idx += 1

    while queue:
        p  = queue.pop(0)
        pid = p["process_id"]

        if pid not in first_run:
            first_run[pid] = time

        run = min(quantum, remaining[pid])
        gantt.append({"pid": pid, "start": time, "end": time + run})
        time           += run
        remaining[pid] -= run

        # Enqueue newly arrived processes
        while idx < n and procs[idx]["arrival_time"] <= time:
            queue.append(procs[idx])
            idx += 1

        if remaining[pid] > 0:
            queue.append(p)   # re-enqueue
        else:
            finish[pid] = time

    result = []
    for i, p in enumerate(procs):
        pid = p["process_id"]
        p["waiting_time"]   = finish[pid] - p["arrival_time"] - p["burst_time"]
        p["turnaround_time"]= finish[pid] - p["arrival_time"]
        p["start_time"]     = first_run.get(pid, 0)
        p["finish_time"]    = finish[pid]
        p["order"]          = i + 1
        result.append(p)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  5. Smart (ML) Schedule
#     ML scores each process, sorts descending, runs non-preemptively
# ─────────────────────────────────────────────────────────────────────────────
def smart_schedule_ml(processes):
    procs = _copy(processes)
    for p in procs:
        p["score"] = get_rank_score(p)
    procs.sort(key=lambda x: x["score"], reverse=True)

    time, order = 0, 1
    for p in procs:
        start              = max(time, p["arrival_time"])
        p["waiting_time"]  = start - p["arrival_time"]
        p["turnaround_time"]= start + p["burst_time"] - p["arrival_time"]
        p["start_time"]    = start
        p["finish_time"]   = start + p["burst_time"]
        p["order"]         = order
        time               = p["finish_time"]
        order             += 1
    return procs


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY — Run all algorithms + ML recommendation
# ─────────────────────────────────────────────────────────────────────────────
def smart_schedule(processes):
    """
    Runs all 5 algorithms, picks the ML-recommended one as primary,
    and returns full comparison data.
    """
    if not processes:
        return {"primary": [], "comparison": [], "recommendation": {}}

    # ML picks best algorithm
    algo_key, algo_name, confidence = predict_best_algorithm(processes)

    algo_map = {
        0: ("FCFS",        fcfs),
        1: ("SJF",         sjf),
        2: ("Priority",    priority_schedule),
        3: ("Round Robin", round_robin),
        4: ("Smart (ML)",  smart_schedule_ml),
    }

    comparison = []
    primary_result = []

    for key, (name, fn) in algo_map.items():
        try:
            result = fn(processes)
            avg_wt  = round(sum(p["waiting_time"]   for p in result) / len(result), 2)
            avg_tat = round(sum(p["turnaround_time"] for p in result) / len(result), 2)
            complexity = ALGORITHM_COMPLEXITY[name]

            comparison.append({
                "algorithm":        name,
                "avg_waiting":      avg_wt,
                "avg_turnaround":   avg_tat,
                "time_complexity":  complexity["time"],
                "space_complexity": complexity["space"],
                "data_structure":   complexity["structure"],
                "preemptive":       complexity["preemptive"],
                "is_recommended":   (name == algo_name),
                "processes":        result
            })

            if name == algo_name:
                primary_result = result

        except Exception as e:
            comparison.append({
                "algorithm": name,
                "error":     str(e),
                "avg_waiting": 0,
                "avg_turnaround": 0,
                "is_recommended": False
            })

    # fallback: if ML picked algo errored, use FCFS
    if not primary_result:
        primary_result = fcfs(processes)

    return {
        "primary":    primary_result,
        "comparison": comparison,
        "recommendation": {
            "algorithm":  algo_name,
            "confidence": confidence,
            "reason":     ALGORITHM_COMPLEXITY.get(algo_name, {}).get("description", "")
        }
    }