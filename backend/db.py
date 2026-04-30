# db.py
# ─────────────────────────────────────────────────────────────────────────────
#  SMART PROCESS SCHEDULING SYSTEM — Database Module
#
#  Tables:
#    processes        — input process queue
#    schedule_results — current scheduling output
#    run_history      — every past scheduling run (audit log)
#    algorithm_stats  — aggregated performance per algorithm
# ─────────────────────────────────────────────────────────────────────────────

import mysql.connector
from config import DB_CONFIG


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMA SETUP — call once on startup
# ─────────────────────────────────────────────────────────────────────────────
def setup_schema():
    """
    Creates all tables if they don't exist.
    Safe to call on every startup.
    """
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS processes (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            process_id   VARCHAR(20)  NOT NULL,
            burst_time   INT          NOT NULL,
            arrival_time INT          NOT NULL DEFAULT 0,
            process_type ENUM('CPU','IO') NOT NULL DEFAULT 'CPU',
            priority     INT          NOT NULL DEFAULT 5,
            created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS schedule_results (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            process_id      VARCHAR(20)  NOT NULL,
            algorithm       VARCHAR(30)  NOT NULL DEFAULT 'Smart (ML)',
            rank_score      FLOAT,
            execution_order INT,
            waiting_time    FLOAT,
            turnaround_time FLOAT,
            start_time      INT,
            finish_time     INT,
            run_id          INT,
            created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS run_history (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            algorithm_used  VARCHAR(30)  NOT NULL,
            num_processes   INT,
            avg_waiting     FLOAT,
            avg_turnaround  FLOAT,
            ml_confidence   FLOAT,
            ran_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS algorithm_stats (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            algorithm        VARCHAR(30) UNIQUE NOT NULL,
            total_runs       INT     NOT NULL DEFAULT 0,
            total_avg_wait   FLOAT   NOT NULL DEFAULT 0,
            total_avg_tat    FLOAT   NOT NULL DEFAULT 0,
            best_wait        FLOAT,
            best_tat         FLOAT,
            last_used        TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESSES TABLE
# ─────────────────────────────────────────────────────────────────────────────
def insert_process(pid, bt, at, ptype, priority=5):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO processes
           (process_id, burst_time, arrival_time, process_type, priority)
           VALUES (%s, %s, %s, %s, %s)""",
        (pid, bt, at, ptype, priority)
    )
    conn.commit()
    conn.close()


def get_all_processes():
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM processes ORDER BY created_at ASC")
    data = cur.fetchall()
    conn.close()
    return data


def clear_processes():
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM processes")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULE RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def clear_results():
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM schedule_results")
    conn.commit()
    conn.close()


def insert_result(pid, score, order_no, wt, tat, algorithm="Smart (ML)",
                  start_time=0, finish_time=0, run_id=None):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO schedule_results
           (process_id, algorithm, rank_score, execution_order,
            waiting_time, turnaround_time, start_time, finish_time, run_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (pid, algorithm, score, order_no, wt, tat, start_time, finish_time, run_id)
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  RUN HISTORY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def log_run(algorithm_used, num_processes, avg_waiting, avg_turnaround, ml_confidence=0.0):
    """Insert a run record and return the new run ID."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO run_history
           (algorithm_used, num_processes, avg_waiting, avg_turnaround, ml_confidence)
           VALUES (%s, %s, %s, %s, %s)""",
        (algorithm_used, num_processes, avg_waiting, avg_turnaround, ml_confidence)
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def get_run_history(limit=10):
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM run_history ORDER BY ran_at DESC LIMIT %s", (limit,)
    )
    data = cur.fetchall()
    conn.close()
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  ALGORITHM STATS TABLE (DBMS Aggregation Demo)
# ─────────────────────────────────────────────────────────────────────────────
def update_algorithm_stats(algorithm, avg_wait, avg_tat):
    """
    Upsert algorithm performance stats.
    Uses INSERT ... ON DUPLICATE KEY UPDATE for atomic upsert.
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO algorithm_stats
               (algorithm, total_runs, total_avg_wait, total_avg_tat, best_wait, best_tat, last_used)
           VALUES (%s, 1, %s, %s, %s, %s, NOW())
           ON DUPLICATE KEY UPDATE
               total_runs     = total_runs + 1,
               total_avg_wait = total_avg_wait + VALUES(total_avg_wait),
               total_avg_tat  = total_avg_tat  + VALUES(total_avg_tat),
               best_wait      = LEAST(COALESCE(best_wait, 9999), VALUES(best_wait)),
               best_tat       = LEAST(COALESCE(best_tat,  9999), VALUES(best_tat)),
               last_used      = NOW()
        """,
        (algorithm, avg_wait, avg_tat, avg_wait, avg_tat)
    )
    conn.commit()
    conn.close()


def get_algorithm_stats():
    """
    Returns aggregated stats per algorithm (DBMS analytics demo).
    Uses computed avg over all runs.
    """
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT
            algorithm,
            total_runs,
            ROUND(total_avg_wait  / total_runs, 2) AS overall_avg_wait,
            ROUND(total_avg_tat   / total_runs, 2) AS overall_avg_tat,
            best_wait,
            best_tat,
            last_used
        FROM algorithm_stats
        ORDER BY overall_avg_wait ASC
    """)
    data = cur.fetchall()
    conn.close()
    return data