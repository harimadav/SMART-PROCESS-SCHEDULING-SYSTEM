# ml_model.py
# ─────────────────────────────────────────────────────────────────────────────
#  SMART PROCESS SCHEDULING SYSTEM — ML Module
#  Uses a trained Random Forest Classifier to predict the best scheduling
#  algorithm for a given workload, AND scores individual processes.
# ─────────────────────────────────────────────────────────────────────────────

import random

# ── Try to import sklearn; gracefully degrade if not installed ────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  SYNTHETIC TRAINING DATA
#  Each sample: [avg_burst, avg_arrival_spread, num_processes, io_ratio]
#  Label: best algorithm (0=FCFS, 1=SJF, 2=Priority, 3=RoundRobin, 4=Smart)
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_DATA = [
    # [avg_burst, arrival_spread, num_proc, io_ratio] → label
    # FCFS works well when all arrive at 0 and bursts are similar
    [5,  0, 3, 0.0],  [6,  0, 4, 0.1],  [4,  1, 3, 0.0],
    [7,  0, 5, 0.0],  [5,  0, 6, 0.1],  [3,  1, 4, 0.0],
    # SJF works best when there's high burst variance
    [2,  3, 5, 0.2],  [1,  4, 6, 0.1],  [3,  5, 7, 0.2],
    [10, 3, 4, 0.0],  [8,  4, 5, 0.1],  [2,  6, 8, 0.2],
    # Priority works best with mixed IO/CPU processes
    [5,  2, 4, 0.5],  [6,  3, 5, 0.6],  [4,  2, 6, 0.7],
    [7,  1, 3, 0.8],  [5,  2, 5, 0.5],  [6,  3, 4, 0.6],
    # Round Robin works best with many IO processes
    [4,  4, 8, 0.8],  [3,  5, 9, 0.9],  [5,  3, 7, 0.7],
    [4,  4, 10,0.8],  [3,  5, 8, 0.9],  [5,  3, 9, 0.7],
    # Smart ML schedule for complex mixed workloads
    [8,  5, 6, 0.4],  [7,  6, 7, 0.5],  [9,  4, 8, 0.4],
    [6,  7, 9, 0.3],  [10, 5, 5, 0.5],  [8,  6, 6, 0.4],
]
TRAINING_LABELS = (
    [0]*6 +   # FCFS
    [1]*6 +   # SJF
    [2]*6 +   # Priority
    [3]*6 +   # Round Robin
    [4]*6     # Smart
)

ALGORITHM_NAMES = {
    0: "FCFS",
    1: "SJF",
    2: "Priority",
    3: "Round Robin",
    4: "Smart (ML)"
}

# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN THE MODEL AT IMPORT TIME
# ─────────────────────────────────────────────────────────────────────────────
_model = None

def _train_model():
    global _model
    if not SKLEARN_AVAILABLE:
        return
    X = np.array(TRAINING_DATA, dtype=float)
    y = np.array(TRAINING_LABELS, dtype=int)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    _model = clf

_train_model()


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API 1: Predict best algorithm for a workload
# ─────────────────────────────────────────────────────────────────────────────
def predict_best_algorithm(processes):
    """
    Given a list of process dicts, predict the best scheduling algorithm.
    Returns (algorithm_key: int, algorithm_name: str, confidence: float)
    """
    if not processes:
        return 0, "FCFS", 1.0

    bursts   = [p["burst_time"]   for p in processes]
    arrivals = [p["arrival_time"] for p in processes]
    io_count = sum(1 for p in processes if p.get("process_type", "CPU") == "IO")

    avg_burst       = sum(bursts) / len(bursts)
    arrival_spread  = max(arrivals) - min(arrivals) if len(arrivals) > 1 else 0
    num_proc        = len(processes)
    io_ratio        = io_count / num_proc

    features = [avg_burst, arrival_spread, num_proc, io_ratio]

    if SKLEARN_AVAILABLE and _model is not None:
        import numpy as np
        X = np.array([features])
        pred  = int(_model.predict(X)[0])
        proba = float(_model.predict_proba(X).max())
        return pred, ALGORITHM_NAMES[pred], round(proba * 100, 1)
    else:
        # Fallback heuristic
        if io_ratio > 0.6:
            return 3, "Round Robin", 70.0
        elif arrival_spread > 4:
            return 1, "SJF", 65.0
        elif io_ratio > 0.4:
            return 2, "Priority", 60.0
        else:
            return 0, "FCFS", 55.0


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API 2: Score an individual process (used by Smart scheduler)
# ─────────────────────────────────────────────────────────────────────────────
def get_rank_score(process):
    """
    Compute a priority score for a single process.
    Higher score = should be scheduled earlier.
    Features: burst_time, arrival_time, process_type, priority
    """
    bt       = max(1, process.get("burst_time", 1))
    at       = process.get("arrival_time", 0)
    ptype    = process.get("process_type", "CPU")
    priority = process.get("priority", 5)   # 1=highest, 10=lowest

    # Weights — tuned for balanced performance
    w_burst    = 0.40   # shorter burst → higher score
    w_arrival  = 0.25   # earlier arrival → higher score
    w_type     = 0.20   # IO bound gets slight boost (prevents IO starvation)
    w_priority = 0.15   # lower priority number → higher score

    type_score     = 1.0 if ptype == "IO" else 0.6
    priority_score = 1.0 / priority          # priority=1 → score=1.0

    score = (
        w_burst    * (1.0 / bt)           +
        w_arrival  * (1.0 / (at + 1))     +
        w_type     * type_score           +
        w_priority * priority_score
    )
    return round(score, 6)


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API 3: Model info for display
# ─────────────────────────────────────────────────────────────────────────────
def get_model_info():
    return {
        "model_type":        "Random Forest Classifier" if SKLEARN_AVAILABLE else "Heuristic (sklearn not found)",
        "sklearn_available": SKLEARN_AVAILABLE,
        "training_samples":  len(TRAINING_DATA),
        "features":          ["avg_burst_time", "arrival_spread", "num_processes", "io_ratio"],
        "classes":           list(ALGORITHM_NAMES.values()),
        "n_estimators":      50 if SKLEARN_AVAILABLE else 0,
    }