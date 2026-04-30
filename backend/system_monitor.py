# system_monitor.py
# ─────────────────────────────────────────────────────────────────────────────
#  SMART PROCESS SCHEDULING SYSTEM — Real-Time System Monitor
#  Uses psutil to expose live OS metrics via Flask endpoints.
# ─────────────────────────────────────────────────────────────────────────────

import time
import collections
import threading

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

# ── In-memory ring buffer: last 60 CPU snapshots (1 per second) ─────────────
_HISTORY_SIZE = 60
_cpu_history  = collections.deque(maxlen=_HISTORY_SIZE)
_lock         = threading.Lock()

def _sampler():
    while True:
        if PSUTIL_OK:
            cpu = psutil.cpu_percent(interval=None)
            with _lock:
                _cpu_history.append({
                    "t":   round(time.time(), 1),
                    "cpu": cpu
                })
        time.sleep(1)

_thread = threading.Thread(target=_sampler, daemon=True)
_thread.start()


def get_stats():
    if not PSUTIL_OK:
        return {"error": "psutil not installed. Run: pip install psutil"}

    cpu_total   = psutil.cpu_percent(interval=0.1)
    cpu_cores   = psutil.cpu_percent(interval=0.1, percpu=True)
    cpu_freq    = psutil.cpu_freq()
    cpu_count_p = psutil.cpu_count(logical=False)
    cpu_count_l = psutil.cpu_count(logical=True)

    mem  = psutil.virtual_memory()
    swap = psutil.swap_memory()

    try:
        disk = psutil.disk_usage("/")
        disk_info = {
            "total_gb": round(disk.total / 1e9, 1),
            "used_gb":  round(disk.used  / 1e9, 1),
            "pct":      disk.percent
        }
    except Exception:
        disk_info = {}

    procs = _top_processes(n=12)

    return {
        "cpu": {
            "total_pct":      round(cpu_total, 1),
            "per_core_pct":   [round(c, 1) for c in cpu_cores],
            "physical_cores": cpu_count_p,
            "logical_cores":  cpu_count_l,
            "freq_mhz":       round(cpu_freq.current, 0) if cpu_freq else None,
            "freq_max_mhz":   round(cpu_freq.max,     0) if cpu_freq else None,
        },
        "memory": {
            "total_gb":     round(mem.total     / 1e9, 2),
            "used_gb":      round(mem.used      / 1e9, 2),
            "available_gb": round(mem.available / 1e9, 2),
            "pct":          mem.percent,
            "swap_used_gb": round(swap.used     / 1e9, 2),
            "swap_pct":     swap.percent,
        },
        "disk": disk_info,
        "processes": procs,
        "timestamp": round(time.time(), 1),
        "psutil_ok": True,
    }


def _top_processes(n=12):
    rows = []
    try:
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent", "status", "num_threads"]
        ):
            try:
                info = proc.info
                rows.append({
                    "pid":     info["pid"],
                    "name":    (info["name"] or "?")[:25],
                    "cpu_pct": round(info["cpu_percent"] or 0, 1),
                    "mem_pct": round(info["memory_percent"] or 0, 2),
                    "status":  info["status"],
                    "threads": info["num_threads"],
                    "type":    "CPU" if (info["cpu_percent"] or 0) > 5 else "IO",
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        rows.sort(key=lambda x: x["cpu_pct"], reverse=True)
    except Exception:
        pass
    return rows[:n]


def get_cpu_history():
    with _lock:
        return list(_cpu_history)