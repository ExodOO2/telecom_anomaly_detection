"""
Live Tower Simulator Script
Owned by: Person 2 (Backend Developer & API Engineer)

Simulates continuous 15-minute telemetry streams hitting the running FastAPI
service, demonstrating real-time anomaly detection alerts in a colorized
Network Operations Center (NOC) terminal dashboard.
"""

import argparse
import collections
import datetime
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Enable ANSI escape sequences on Windows console
if os.name == "nt":
    os.system("")

# ---------------------------------------------------------------------------
# ANSI Color Palette for NOC Terminal
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Foreground Colors
FG_RED = "\033[91m"
FG_GREEN = "\033[92m"
FG_YELLOW = "\033[93m"
FG_BLUE = "\033[94m"
FG_MAGENTA = "\033[95m"
FG_CYAN = "\033[96m"
FG_WHITE = "\033[97m"

# Background Colors
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"


# ---------------------------------------------------------------------------
# Reference Telemetry Banks (Derived from actual 4G dataset for cell 3BLTE)
# ---------------------------------------------------------------------------
NORMAL_TELEMETRY_BANK: List[Dict[str, float]] = [
    {"PRBUsageDL": 3.133, "meanThr_DL": 0.875, "meanUE_DL": 1.172},
    {"PRBUsageDL": 1.213, "meanThr_DL": 0.324, "meanUE_DL": 1.132},
    {"PRBUsageDL": 0.606, "meanThr_DL": 0.102, "meanUE_DL": 1.031},
    {"PRBUsageDL": 0.707, "meanThr_DL": 0.066, "meanUE_DL": 1.061},
    {"PRBUsageDL": 1.584, "meanThr_DL": 0.562, "meanUE_DL": 1.122},
    {"PRBUsageDL": 4.345, "meanThr_DL": 1.688, "meanUE_DL": 1.091},
    {"PRBUsageDL": 0.909, "meanThr_DL": 0.273, "meanUE_DL": 1.101},
    {"PRBUsageDL": 0.707, "meanThr_DL": 0.120, "meanUE_DL": 1.051},
    {"PRBUsageDL": 0.505, "meanThr_DL": 0.070, "meanUE_DL": 1.041},
    {"PRBUsageDL": 0.505, "meanThr_DL": 0.071, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.034, "meanUE_DL": 1.011},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.027, "meanUE_DL": 1.031},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.019, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.072, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.606, "meanThr_DL": 0.197, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.606, "meanThr_DL": 0.205, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.505, "meanThr_DL": 0.150, "meanUE_DL": 1.031},
    {"PRBUsageDL": 0.707, "meanThr_DL": 0.210, "meanUE_DL": 1.041},
    {"PRBUsageDL": 0.808, "meanThr_DL": 0.250, "meanUE_DL": 1.051},
    {"PRBUsageDL": 1.111, "meanThr_DL": 0.350, "meanUE_DL": 1.081},
    {"PRBUsageDL": 1.515, "meanThr_DL": 0.510, "meanUE_DL": 1.111},
    {"PRBUsageDL": 2.020, "meanThr_DL": 0.720, "meanUE_DL": 1.141},
    {"PRBUsageDL": 2.525, "meanThr_DL": 0.890, "meanUE_DL": 1.171},
    {"PRBUsageDL": 2.929, "meanThr_DL": 1.020, "meanUE_DL": 1.201},
    {"PRBUsageDL": 3.434, "meanThr_DL": 1.250, "meanUE_DL": 1.231},
]

ANOMALY_TELEMETRY_BANK: List[Dict[str, float]] = [
    {"PRBUsageDL": 0.606, "meanThr_DL": 0.205, "meanUE_DL": 1.021},
    {"PRBUsageDL": 1.139, "meanThr_DL": 0.575, "meanUE_DL": 1.062},
    {"PRBUsageDL": 0.961, "meanThr_DL": 0.308, "meanUE_DL": 1.054},
    {"PRBUsageDL": 0.505, "meanThr_DL": 0.012, "meanUE_DL": 1.031},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.061, "meanUE_DL": 1.071},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.035, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.016, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.020, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.505, "meanThr_DL": 0.070, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.011, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.505, "meanThr_DL": 0.020, "meanUE_DL": 1.031},
    {"PRBUsageDL": 1.213, "meanThr_DL": 0.401, "meanUE_DL": 1.071},
    {"PRBUsageDL": 0.404, "meanThr_DL": 0.031, "meanUE_DL": 1.021},
    {"PRBUsageDL": 0.505, "meanThr_DL": 0.121, "meanUE_DL": 1.041},
    {"PRBUsageDL": 1.843, "meanThr_DL": 0.884, "meanUE_DL": 1.003},
    {"PRBUsageDL": 0.424, "meanThr_DL": 0.023, "meanUE_DL": 1.063},
]


# ---------------------------------------------------------------------------
# Synthetic Telemetry Generation
# ---------------------------------------------------------------------------
def generate_telemetry_step(
    phase: str,
    step_idx: int,
    base_time: datetime.datetime,
) -> Tuple[Dict[str, float], datetime.datetime]:
    """
    Generates a realistic 15-minute cellular KPI reading based on current phase.
    
    Phases:
      - 'normal': Healthy operational traffic with natural diurnal fluctuation
      - 'anomaly': Outage / throughput collapse (users connected, PRB used, but 0 throughput)
      - 'congestion': Severe radio saturation (PRB near 100%, high latency)
    """
    sim_time = base_time + datetime.timedelta(minutes=15 * step_idx)

    if phase == "normal":
        # Cycle through realistic normal sequence with subtle realistic noise (+- 2%)
        base = NORMAL_TELEMETRY_BANK[step_idx % len(NORMAL_TELEMETRY_BANK)]
        jitter = random.uniform(-0.015, 0.015)
        prb = max(0.1, round(base["PRBUsageDL"] * (1.0 + jitter), 3))
        thr = max(0.01, round(base["meanThr_DL"] * (1.0 + jitter), 3))
        ue = max(1.0, round(base["meanUE_DL"] * (1.0 + jitter * 0.5), 3))
    elif phase == "anomaly":
        # Outage / Service Degradation: severe throughput collapse
        base = ANOMALY_TELEMETRY_BANK[step_idx % len(ANOMALY_TELEMETRY_BANK)]
        jitter = random.uniform(-0.01, 0.01)
        prb = max(0.1, round(base["PRBUsageDL"] * (1.0 + jitter), 3))
        thr = max(0.005, round(base["meanThr_DL"] * (1.0 + jitter), 4))
        ue = max(1.0, round(base["meanUE_DL"] * (1.0 + jitter * 0.5), 3))
    elif phase == "congestion":
        # Heavy PRB saturation
        prb = min(100.0, round(85.0 + random.uniform(0.0, 14.0), 2))
        thr = round(random.uniform(0.02, 0.08), 3)
        ue = round(random.uniform(25.0, 45.0), 2)
    else:
        prb, thr, ue = 1.0, 0.5, 1.1

    reading = {
        "PRBUsageDL": prb,
        "meanThr_DL": thr,
        "meanUE_DL": ue,
    }
    return reading, sim_time


def load_csv_telemetry(csv_path: Path, cell_name: str) -> List[Dict[str, float]]:
    """Loads KPI sequences for a specific cell from a raw CSV file if available."""
    import csv

    readings = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("CellName", "").strip() == cell_name or not cell_name:
                try:
                    readings.append({
                        "PRBUsageDL": float(row["PRBUsageDL"]),
                        "meanThr_DL": float(row["meanThr_DL"]),
                        "meanUE_DL": float(row["meanUE_DL"]),
                    })
                except (KeyError, ValueError):
                    continue
    return readings


# ---------------------------------------------------------------------------
# Terminal Visual Components
# ---------------------------------------------------------------------------
def print_banner(
    cell_name: str,
    endpoint: str,
    threshold: float,
    mode: str,
    speed: float,
    steps: int,
):
    """Renders the executive NOC Dashboard Header."""
    thr_label = "0.50 (Safety Net Mode)" if threshold == 0.50 else f"{threshold:.2f} (Custom Mode)"
    if threshold == 0.66:
        thr_label = "0.66 (Balanced / Low False Alarm Mode)"

    steps_label = f"{steps} ticks" if steps > 0 else "Continuous (Infinite Loop)"

    print(f"\n{BOLD}{FG_CYAN}" + "=" * 80 + f"{RESET}")
    print(f"{BOLD}{FG_WHITE}   TELECOM NETWORK OPERATIONS CENTER (NOC) — LIVE ANOMALY MONITOR{RESET}")
    print(f"{BOLD}{FG_CYAN}" + "=" * 80 + f"{RESET}")
    print(f"  {FG_WHITE}Target Base Station :{RESET} {BOLD}{FG_YELLOW}{cell_name}{RESET}")
    print(f"  {FG_WHITE}FastAPI Endpoint    :{RESET} {FG_CYAN}{endpoint}{RESET}")
    print(f"  {FG_WHITE}Decision Threshold  :{RESET} {BOLD}{FG_MAGENTA}{thr_label}{RESET}")
    print(f"  {FG_WHITE}Simulation Profile  :{RESET} {BOLD}{FG_WHITE}{mode.upper()}{RESET}")
    print(f"  {FG_WHITE}Tick Pacing (Speed) :{RESET} {speed}s per 15-minute telemetry step")
    print(f"  {FG_WHITE}Simulation Scope    :{RESET} {steps_label}")
    print(f"{BOLD}{FG_CYAN}" + "-" * 80 + f"{RESET}")
    print(
        f" {BOLD}{FG_WHITE}{'Tick':<5} {'Time':<8} {'PRB %':<8} {'Throughput':<12} "
        f"{'Users':<7} {'Status':<18} {'Confidence':<14} {'Latency':<8}{RESET}"
    )
    print(f"{BOLD}{FG_CYAN}" + "-" * 80 + f"{RESET}")


def render_progress_bar(confidence: float, is_anomaly: bool, width: int = 8) -> str:
    """Renders a compact colored confidence progress bar."""
    filled = int(round(confidence * width))
    empty = width - filled
    bar_str = "█" * filled + "░" * empty
    color = FG_RED if is_anomaly else FG_GREEN
    return f"{color}{bar_str}{RESET}"


def check_api_health(base_url: str) -> bool:
    """Verifies that the target FastAPI service is online and model is loaded."""
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded", False):
                return True
        return False
    except requests.exceptions.RequestException:
        return False


def print_summary_card(
    total_ticks: int,
    normal_count: int,
    anomaly_count: int,
    latencies: List[float],
    cell_name: str,
):
    """Renders session completion statistics."""
    if total_ticks == 0:
        print("\n[NOC] Simulation terminated before any telemetry was streamed.")
        return

    anomaly_pct = (anomaly_count / total_ticks) * 100.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    max_lat = max(latencies) if latencies else 0.0

    print(f"\n{BOLD}{FG_CYAN}" + "=" * 80 + f"{RESET}")
    print(f"{BOLD}{FG_WHITE}   NOC TELEMETRY STREAM SESSION SUMMARY — CELL: {cell_name}{RESET}")
    print(f"{BOLD}{FG_CYAN}" + "=" * 80 + f"{RESET}")
    print(f"  Total Windows Evaluated : {BOLD}{total_ticks}{RESET}")
    print(f"  Healthy Windows (NORMAL): {FG_GREEN}{normal_count}{RESET} ({100.0 - anomaly_pct:.1f}%)")
    print(f"  Anomalies Flagged       : {FG_RED}{BOLD}{anomaly_count}{RESET} ({anomaly_pct:.1f}%)")
    print(f"  Average Model Latency   : {avg_lat:.2f} ms")
    print(f"  Peak Model Latency      : {max_lat:.2f} ms")

    # Overall Tower Assessment
    if anomaly_pct > 25.0:
        health_status = f"{BG_RED}{BOLD}{FG_WHITE} CRITICAL OUTAGE / DEGRADATION DETECTED {RESET}"
    elif anomaly_pct > 5.0:
        health_status = f"{BG_YELLOW}{BOLD}{FG_WHITE} MINOR QOS INSTABILITY DETECTED {RESET}"
    else:
        health_status = f"{BG_GREEN}{BOLD}{FG_WHITE} NOMINAL / TOWER OPERATING HEALTHY {RESET}"

    print(f"  Overall Sector State    : {health_status}")
    print(f"{BOLD}{FG_CYAN}" + "=" * 80 + f"{RESET}\n")


# ---------------------------------------------------------------------------
# Main Streaming Engine
# ---------------------------------------------------------------------------
def run_simulation(args: argparse.Namespace):
    base_url = args.url.rstrip("/")
    predict_url = f"{base_url}/predict"
    cell_name = args.cell
    speed = args.speed
    threshold = args.threshold
    mode = args.mode
    max_steps = args.steps

    # 1. Verify API Connectivity
    print(f"{DIM}[NOC Initiator] Checking API status at {base_url}...{RESET}", end="", flush=True)
    if not check_api_health(base_url):
        print(f"\r{BOLD}{FG_RED}[ERROR] Cannot connect to API or model is not loaded at {base_url}.{RESET}")
        print(f"{FG_YELLOW}Ensure your FastAPI server is active in another terminal:{RESET}")
        print(f"  {BOLD}uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload{RESET}\n")
        sys.exit(1)
    print(f"\r{BOLD}{FG_GREEN}[NOC Online] Connected to FastAPI service. ResNet-18 model ready.{RESET}")

    # 2. Render Header
    print_banner(cell_name, predict_url, threshold, mode, speed, max_steps)

    # 3. Initialize Sliding Window (deque of 16 items)
    # Pre-seed with normal reference telemetry so scoring begins immediately on tick 1
    sliding_window = collections.deque(NORMAL_TELEMETRY_BANK[:16], maxlen=16)

    # Optional CSV data source
    csv_readings = []
    if mode == "csv" and args.csv:
        csv_file = Path(args.csv)
        if csv_file.exists():
            csv_readings = load_csv_telemetry(csv_file, cell_name)
            print(f"{DIM}[NOC CSV] Loaded {len(csv_readings)} historical rows for {cell_name}.{RESET}")
        else:
            print(f"{FG_YELLOW}[WARN] CSV '{args.csv}' not found. Falling back to synthetic telemetry.{RESET}")
            mode = "mixed"

    base_time = datetime.datetime.now() - datetime.timedelta(hours=4)

    total_evaluated = 0
    normal_count = 0
    anomaly_count = 0
    model_latencies: List[float] = []

    try:
        step = 0
        while True:
            step += 1
            if max_steps > 0 and step > max_steps:
                break

            # Determine simulation phase for mixed mode
            if mode == "mixed":
                # Ticks 1-7: Normal
                # Ticks 8-15: Outage / Anomaly Injection
                # Ticks 16+: Recovery back to Normal
                if 8 <= step <= 15:
                    curr_phase = "anomaly"
                else:
                    curr_phase = "normal"
            elif mode == "anomaly":
                curr_phase = "anomaly"
            elif mode == "normal":
                curr_phase = "normal"
            else:
                curr_phase = "normal"

            # Generate or fetch next reading
            if mode == "csv" and csv_readings:
                reading = csv_readings[(step - 1) % len(csv_readings)]
                sim_time = base_time + datetime.timedelta(minutes=15 * step)
            else:
                reading, sim_time = generate_telemetry_step(curr_phase, step, base_time)

            # Push reading to sliding window (drops oldest 16th step)
            sliding_window.append(reading)

            # Build PredictRequest payload conforming to api/schemas.py
            payload = {
                "cell_name": cell_name,
                "threshold": threshold,
                "window": list(sliding_window),
            }

            # Send HTTP request
            t0 = time.perf_counter()
            try:
                response = requests.post(predict_url, json=payload, timeout=5.0)
                round_trip_ms = (time.perf_counter() - t0) * 1000.0

                if response.status_code == 200:
                    data = response.json()
                    status_text = data.get("status", "UNKNOWN")
                    confidence = float(data.get("confidence", 0.0))
                    inference_ms = float(data.get("inference_time_ms", 0.0))
                    is_anomaly = (status_text == "ANOMALY_DETECTED")

                    total_evaluated += 1
                    model_latencies.append(inference_ms)

                    if is_anomaly:
                        anomaly_count += 1
                        status_badge = f"{BG_RED}{BOLD}{FG_WHITE} 🚨 ANOMALY ALERT {RESET}"
                    else:
                        normal_count += 1
                        status_badge = f"{FG_GREEN}{BOLD}   ✅ NORMAL    {RESET}"

                    bar = render_progress_bar(confidence, is_anomaly)
                    time_str = sim_time.strftime("%H:%M")

                    print(
                        f" {FG_WHITE}{step:<5}{RESET} "
                        f"{DIM}{time_str:<8}{RESET} "
                        f"{reading['PRBUsageDL']:>6.1f}%  "
                        f"{reading['meanThr_DL']:>8.3f} Mb/s  "
                        f"{reading['meanUE_DL']:>5.2f}  "
                        f"{status_badge}  "
                        f"{confidence * 100:>5.1f}% {bar}  "
                        f"{DIM}{inference_ms:>5.1f}ms{RESET}"
                    )

                    # Highlight anomalous incident transition
                    if is_anomaly and (step == 8 or curr_phase == "anomaly" and anomaly_count == 1):
                        print(
                            f"   {FG_RED}{BOLD}└─▶ [INCIDENT TRIGGERED] QoS Collapse Detected! "
                            f"Throughput={reading['meanThr_DL']} Mbps (Active Users={reading['meanUE_DL']}). "
                            f"NOC ticket auto-created.{RESET}"
                        )
                    elif not is_anomaly and step == 16:
                        print(
                            f"   {FG_GREEN}{BOLD}└─▶ [RECOVERY CONFIRMED] Downlink telemetry restored to nominal operating range.{RESET}"
                        )

                else:
                    print(
                        f" {FG_RED}{step:<5} HTTP {response.status_code}: {response.text[:60]}{RESET}"
                    )

            except requests.exceptions.RequestException as e:
                print(f" {FG_RED}{step:<5} Connection Error: {e}{RESET}")

            # Sleep between ticks to simulate pacing
            if speed > 0:
                time.sleep(speed)

    except KeyboardInterrupt:
        print(f"\n{FG_YELLOW}[NOC] Telemetry streaming interrupted by user.{RESET}")

    # 4. Display Session Summary
    print_summary_card(total_evaluated, normal_count, anomaly_count, model_latencies, cell_name)


def main():
    parser = argparse.ArgumentParser(
        description="Network Operations Center (NOC) Antenna Telemetry Streaming Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running FastAPI anomaly detection service",
    )
    parser.add_argument(
        "--cell",
        type=str,
        default="3BLTE",
        help="Base station cell identifier (e.g., 3BLTE, 1BLTE, 2BLTE)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.4,
        help="Simulated seconds between 15-minute telemetry ticks (e.g. 0.3 for fast demo)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Decision threshold: 0.50 (Safety Net / High Recall) or 0.66 (Balanced)",
    )
    parser.add_argument(
        "--mode",
        choices=["mixed", "normal", "anomaly", "csv"],
        default="mixed",
        help="Traffic profile: 'mixed' (normal -> outage -> recovery), 'normal', 'anomaly', or 'csv'",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of 15-minute ticks to stream (0 for continuous infinite stream)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to raw CSV dataset containing historical telemetry",
    )

    args = parser.parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
