#!/usr/bin/env python3
"""
Fixed-action open-loop test for the bottom-wall 5-segment bubble-rising pybind case.

Expected C++/pybind side:
    PYBIND11_MODULE(br_2d_bubble_rising_bottom_python, m)
    class name: bubble_rising_heat_from_sph_cpp

Expected bottom-wall control methods:
    set_bottom_wall_segment_actions(actions, amplitude, mean_temperature)
    set_bottom_wall_segment_temperatures(temperatures, enforce_mean, mean_temperature)
    get_bottom_wall_segment_temperatures()

The C++ case is assumed to use DL=2, DH=2, and reached_target_height = (center_y >= 1.8).
"""
  
import os
import sys
import csv
import glob
import argparse
import importlib.util
from typing import Optional


# =============================================================================
# Configure your pybind module.
# This name must match:
#     PYBIND11_MODULE(br_2d_bubble_rising_bottom_python, m)
# =============================================================================
MODULE_NAME = "br_2d_bubble_rising_bottom_python"
CLASS_NAME = "bubble_rising_heat_from_sph_cpp"

# Optional: override search directory by environment variable.
ENV_DIR = os.environ.get("SPH_PYBIND_LIB_DIR", "").strip()

# Bottom wall is divided into 5 segments.
N_SEG = 5


# =============================================================================
# Fixed open-loop actions for bottom-wall 5-segment control-authority test.
# C++ maps actions to segment temperatures by:
#     T_i = mean_temperature + amplitude * (a_i - mean(a))
# =============================================================================
FIXED_ACTIONS = [
    ("zero_baseline", [0.0, 0.0, 0.0, 0.0, 0.0]),

    # Left half bottom hotter / right half bottom colder.
    ("left_hot_right_cold", [1.0, 1.0, 0.0, -1.0, -1.0]),

    # Left half bottom colder / right half bottom hotter.
    ("left_cold_right_hot", [-1.0, -1.0, 0.0, 1.0, 1.0]),

    # Center plume test.
    ("center_hot_edges_cold", [-1.0, 0.0, 1.0, 0.0, -1.0]),

    # Edge-heating test.
    ("edges_hot_center_cold", [1.0, 0.0, -1.0, 0.0, 1.0]),

    # One-sided weak bias tests.
    ("left_hot_only", [1.0, 0.0, 0.0, 0.0, 0.0]),
    ("right_hot_only", [0.0, 0.0, 0.0, 0.0, 1.0]),
]


# =============================================================================
# Loader helpers.
# =============================================================================
def _candidate_dirs() -> list[str]:
    """
    Search order:
      1) SPH_PYBIND_LIB_DIR, if provided
      2) ../../lib/Release
      3) ../../lib/RelWithDebInfo
      4) ../../lib/Debug
      5) ../../lib
    """
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    lib = os.path.join(base, "lib")

    dirs = []

    if ENV_DIR:
        dirs.append(os.path.abspath(ENV_DIR))

    for cfg in ("Release", "RelWithDebInfo", "Debug", ""):
        d = os.path.join(lib, cfg) if cfg else lib
        dirs.append(d)

    seen = set()
    unique_dirs = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)

    return unique_dirs


def _candidate_patterns() -> list[str]:
    pyver = f"{sys.version_info.major}{sys.version_info.minor}"

    if os.name == "nt":
        return [
            f"{MODULE_NAME}.cp{pyver}-win_amd64.pyd",
            f"{MODULE_NAME}.pyd",
        ]

    return [
        f"{MODULE_NAME}.cpython-{pyver}*.so",
        f"{MODULE_NAME}.abi3*.so",
        f"{MODULE_NAME}.so",
    ]


def locate_extension() -> Optional[str]:
    for d in _candidate_dirs():
        if not os.path.isdir(d):
            continue

        for pat in _candidate_patterns():
            matches = glob.glob(os.path.join(d, pat))
            if matches:
                matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return os.path.abspath(matches[0])

    return None


def load_extension(path: str):
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)

    if not spec or not spec.loader:
        raise ImportError(f"Cannot create spec/loader for {MODULE_NAME} at: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = mod
    spec.loader.exec_module(mod)

    return mod


def ensure_module():
    ext = locate_extension()

    if ext and os.path.exists(ext):
        print(f"[Loader] Using compiled extension: {ext}")
        return load_extension(ext)

    for d in _candidate_dirs():
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    try:
        __import__(MODULE_NAME)
        mod = sys.modules[MODULE_NAME]
        print(f"[Loader] Imported '{MODULE_NAME}' via sys.path: {mod.__file__}")
        return mod

    except Exception as e:
        checked_dirs = [d for d in _candidate_dirs() if os.path.isdir(d)]

        msg = [
            f"Failed to import '{MODULE_NAME}'.",
            f"Checked directories: {', '.join(checked_dirs) if checked_dirs else '(none found)'},",
            "Tips:",
            "  - Make sure PYBIND11_MODULE name is br_2d_bubble_rising_bottom_python.",
            "  - Make sure the generated .pyd/.so is in lib/Release, lib/Debug, or SPH_PYBIND_LIB_DIR.",
            "  - Make sure Python version/architecture matches the compiled extension.",
            "  - On Windows, expected file looks like br_2d_bubble_rising_bottom_python.cp310-win_amd64.pyd.",
        ]

        raise ImportError("\n".join(msg)) from e


# =============================================================================
# Utility.
# =============================================================================
def mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def prepare_pybind_test_dir(clear: bool = False) -> str:
    """
    Create output root for this open-loop test:
        <case>/bin/bind/pybind_test_bottom/

    All generated files will be written under this folder because we chdir into it
    before constructing the C++ solver.
    """
    bind_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(bind_dir, "pybind_test_bottom")

    if clear and os.path.isdir(test_dir):
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    mkdir(test_dir)

    for name in ("input", "output", "reload", "restart"):
        mkdir(os.path.join(test_dir, name))

    return test_dir


def append_csv_row(path: str, fieldnames: list[str], row: dict):
    mkdir(os.path.dirname(path))
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)


def safe_float(metrics: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(metrics[key])
    except Exception:
        return float(default)


def safe_int(metrics: dict, key: str, default: int = 0) -> int:
    try:
        return int(metrics[key])
    except Exception:
        return int(default)


def action_temp_fields(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(N_SEG)]


METRICS_FIELDS = [
    "case_name", "rl_step", "time", "number_of_iterations",
    *action_temp_fields("action"),
    *action_temp_fields("temp"),
    "center_x", "center_y", "center_u", "center_v",
    "x_min", "x_max", "y_min", "y_max",
    "bubble_width", "bubble_height", "deformation_index", "aspect_ratio",
    "bubble_area", "area_ratio", "area_rel", "area_error",
    "centroid_in_target", "reached_target_height", "all_extreme_particles_in_target",
    "left_particle_in_target", "right_particle_in_target",
    "bottom_particle_in_target", "top_particle_in_target",
]


SUMMARY_FIELDS = [
    "case_name",
    "parallel_env",
    "episode_env",
    *action_temp_fields("action"),
    *action_temp_fields("temp"),
    "amplitude",
    "mean_temperature",
    "warmup_time",
    "control_horizon",
    "sample_dt",
    "final_time",
    "final_iterations",
    "target_entry_time",
    "centroid_entry_time",
    "time_centroid_in_target",
    "time_whole_bubble_in_target",
    "final_center_x",
    "final_center_y",
    "final_center_u",
    "final_center_v",
    "final_deformation_index",
    "max_deformation_index",
    "final_area_ratio",
    "final_area_rel",
    "final_area_error",
    "max_area_error",
    "final_centroid_in_target",
    "final_reached_target_height",
    "final_all_extreme_particles_in_target",
    "has_reached_target_height",
    "is_bubble_in_target_region",
    "is_whole_bubble_in_target_region",
]


def validate_vector(name: str, values: list[float], expected: int = N_SEG):
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(values)}: {values}")


def fill_action_temp(row: dict, action: list[float], seg_temps: list[float]):
    validate_vector("action", action)
    validate_vector("seg_temps", seg_temps)

    for i in range(N_SEG):
        row[f"action_{i}"] = float(action[i])
        row[f"temp_{i}"] = float(seg_temps[i])


def metrics_row(
    case_name: str,
    action: list[float],
    seg_temps: list[float],
    metrics: dict,
    ref_area_ratio: float,
    rl_step: int,
    time_value: float,
    number_of_iterations: int,
) -> dict:
    area_ratio = safe_float(metrics, "area_ratio")
    area_rel = area_ratio / (ref_area_ratio + 1.0e-12)
    area_error = abs(area_rel - 1.0)

    row = {
        "case_name": case_name,
        "rl_step": int(rl_step),
        "time": float(time_value),
        "number_of_iterations": int(number_of_iterations),

        "center_x": safe_float(metrics, "center_x"),
        "center_y": safe_float(metrics, "center_y"),
        "center_u": safe_float(metrics, "center_u"),
        "center_v": safe_float(metrics, "center_v"),

        "x_min": safe_float(metrics, "x_min"),
        "x_max": safe_float(metrics, "x_max"),
        "y_min": safe_float(metrics, "y_min"),
        "y_max": safe_float(metrics, "y_max"),

        "bubble_width": safe_float(metrics, "bubble_width"),
        "bubble_height": safe_float(metrics, "bubble_height"),
        "deformation_index": safe_float(metrics, "deformation_index"),
        "aspect_ratio": safe_float(metrics, "aspect_ratio"),

        "bubble_area": safe_float(metrics, "bubble_area"),
        "area_ratio": area_ratio,
        "area_rel": area_rel,
        "area_error": area_error,

        "centroid_in_target": safe_int(metrics, "centroid_in_target"),
        "reached_target_height": safe_int(metrics, "reached_target_height"),
        "all_extreme_particles_in_target": safe_int(metrics, "all_extreme_particles_in_target"),

        "left_particle_in_target": safe_int(metrics, "left_particle_in_target"),
        "right_particle_in_target": safe_int(metrics, "right_particle_in_target"),
        "bottom_particle_in_target": safe_int(metrics, "bottom_particle_in_target"),
        "top_particle_in_target": safe_int(metrics, "top_particle_in_target"),
    }

    fill_action_temp(row, action, seg_temps)
    return row


def run_one_fixed_action(
    Solver,
    case_name: str,
    action: list[float],
    parallel_env: int,
    episode_env: int,
    reload_particles: bool,
    write_output: bool,
    warmup_time: float,
    control_horizon: float,
    sample_dt: float,
    amplitude: float,
    mean_temperature: float,
    output_dir: str,
) -> dict:
    validate_vector("action", action)

    print("\n" + "=" * 78)
    print(f"[Case] {case_name}")
    print(f"[Case] parallel_env={parallel_env}, episode_env={episode_env}")
    print(f"[Case] action={action}")

    sim = Solver(
        int(parallel_env),
        int(episode_env),
        bool(reload_particles),
        bool(write_output),
    )

    # Match the Gym environment reset: start from uniform bottom-wall temperature.
    sim.set_bottom_wall_segment_temperatures(
        [mean_temperature] * N_SEG,
        True,
        mean_temperature,
    )

    if warmup_time > 0.0:
        target_time = sim.get_physical_time() + warmup_time
        print(f"[Warmup] Running to t={target_time:.6f}")
        sim.run_case(target_time)

    metrics_warm = sim.get_bubble_metrics_dict()
    ref_area_ratio = max(safe_float(metrics_warm, "area_ratio", 1.0), 1.0e-12)

    sim.set_bottom_wall_segment_actions(
        [float(x) for x in action],
        float(amplitude),
        float(mean_temperature),
    )
    seg_temps = list(sim.get_bottom_wall_segment_temperatures())
    validate_vector("seg_temps returned by C++", seg_temps)

    print(f"[Control] bottom segment temperatures={seg_temps}")
    print(f"[Control] mean segment temperature={sum(seg_temps) / len(seg_temps):.12f}")

    metrics_csv = os.path.join(output_dir, f"metrics_{case_name}.csv")
    if os.path.exists(metrics_csv):
        os.remove(metrics_csv)

    start_time = float(sim.get_physical_time())
    final_target_time = start_time + float(control_horizon)
    current_time = start_time

    target_entry_time = None
    centroid_entry_time = None
    time_centroid_in_target = 0.0
    time_whole_bubble_in_target = 0.0
    max_deformation_index = 0.0
    max_area_error = 0.0

    step = 0
    previous_time = current_time

    # Record warmup state as step 0.
    row0 = metrics_row(
        case_name=case_name,
        action=action,
        seg_temps=seg_temps,
        metrics=metrics_warm,
        ref_area_ratio=ref_area_ratio,
        rl_step=step,
        time_value=current_time,
        number_of_iterations=sim.get_number_of_iterations(),
    )
    append_csv_row(metrics_csv, METRICS_FIELDS, row0)

    while current_time < final_target_time - 1.0e-12:
        next_time = min(current_time + float(sample_dt), final_target_time)
        sim.run_case(next_time)

        step += 1
        current_time = float(sim.get_physical_time())
        metrics = sim.get_bubble_metrics_dict()

        dt_actual = max(current_time - previous_time, 0.0)
        previous_time = current_time

        row = metrics_row(
            case_name=case_name,
            action=action,
            seg_temps=seg_temps,
            metrics=metrics,
            ref_area_ratio=ref_area_ratio,
            rl_step=step,
            time_value=current_time,
            number_of_iterations=sim.get_number_of_iterations(),
        )
        append_csv_row(metrics_csv, METRICS_FIELDS, row)

        if row["reached_target_height"] and target_entry_time is None:
            target_entry_time = current_time

        if row["centroid_in_target"] and centroid_entry_time is None:
            centroid_entry_time = current_time

        if row["centroid_in_target"]:
            time_centroid_in_target += dt_actual

        if row["all_extreme_particles_in_target"]:
            time_whole_bubble_in_target += dt_actual

        max_deformation_index = max(max_deformation_index, row["deformation_index"])
        max_area_error = max(max_area_error, row["area_error"])

    final_metrics = sim.get_bubble_metrics_dict()
    final_area_ratio = safe_float(final_metrics, "area_ratio")
    final_area_rel = final_area_ratio / (ref_area_ratio + 1.0e-12)
    final_area_error = abs(final_area_rel - 1.0)

    summary = {
        "case_name": case_name,
        "parallel_env": int(parallel_env),
        "episode_env": int(episode_env),
        "amplitude": float(amplitude),
        "mean_temperature": float(mean_temperature),
        "warmup_time": float(warmup_time),
        "control_horizon": float(control_horizon),
        "sample_dt": float(sample_dt),
        "final_time": float(sim.get_physical_time()),
        "final_iterations": int(sim.get_number_of_iterations()),
        "target_entry_time": "" if target_entry_time is None else float(target_entry_time),
        "centroid_entry_time": "" if centroid_entry_time is None else float(centroid_entry_time),
        "time_centroid_in_target": float(time_centroid_in_target),
        "time_whole_bubble_in_target": float(time_whole_bubble_in_target),
        "final_center_x": safe_float(final_metrics, "center_x"),
        "final_center_y": safe_float(final_metrics, "center_y"),
        "final_center_u": safe_float(final_metrics, "center_u"),
        "final_center_v": safe_float(final_metrics, "center_v"),
        "final_deformation_index": safe_float(final_metrics, "deformation_index"),
        "max_deformation_index": float(max_deformation_index),
        "final_area_ratio": final_area_ratio,
        "final_area_rel": final_area_rel,
        "final_area_error": final_area_error,
        "max_area_error": float(max_area_error),
        "final_centroid_in_target": safe_int(final_metrics, "centroid_in_target"),
        "final_reached_target_height": safe_int(final_metrics, "reached_target_height"),
        "final_all_extreme_particles_in_target": safe_int(final_metrics, "all_extreme_particles_in_target"),
        "has_reached_target_height": bool(sim.has_reached_target_height()),
        "is_bubble_in_target_region": bool(sim.is_bubble_in_target_region()),
        "is_whole_bubble_in_target_region": bool(sim.is_whole_bubble_in_target_region()),
    }
    fill_action_temp(summary, action, seg_temps)

    print("[Summary]")
    print(f"  target_entry_time              = {summary['target_entry_time']}")
    print(f"  centroid_entry_time            = {summary['centroid_entry_time']}")
    print(f"  time_centroid_in_target        = {summary['time_centroid_in_target']:.6f}")
    print(f"  time_whole_bubble_in_target    = {summary['time_whole_bubble_in_target']:.6f}")
    print(f"  final_center                   = ({summary['final_center_x']:.6f}, {summary['final_center_y']:.6f})")
    print(f"  max_deformation_index          = {summary['max_deformation_index']:.6f}")
    print(f"  max_area_error                 = {summary['max_area_error']:.6f}")
    print(f"  per-step metrics CSV           = {metrics_csv}")

    return summary


# =============================================================================
# Bottom-wall bubble rising fixed-action open-loop test.
# =============================================================================
def run_case():
    parser = argparse.ArgumentParser()

    parser.add_argument("--parallel_env_base", default=0, type=int)
    parser.add_argument("--episode_env_base", default=0, type=int)

    parser.add_argument("--reload_particles", action="store_true")

    # For this open-loop test, output is ON by default.
    parser.add_argument("--no_write_output", action="store_true")
    parser.add_argument("--clear_pybind_test", action="store_true")

    parser.add_argument("--warmup_time", default=0.02, type=float)
    parser.add_argument("--control_horizon", default=4.5, type=float)
    parser.add_argument("--sample_dt", default=0.02, type=float)

    parser.add_argument("--amplitude", default=0.3, type=float)
    parser.add_argument("--mean_temperature", default=1.0, type=float)

    parser.add_argument(
        "--skip_zero_baseline",
        action="store_true",
        help="Skip the zero-action baseline case and run only the nonzero fixed actions.",
    )

    args = parser.parse_args()

    if args.sample_dt <= 0.0:
        raise ValueError("--sample_dt must be positive.")

    mod = ensure_module()

    if not hasattr(mod, CLASS_NAME):
        raise AttributeError(
            f"Module '{MODULE_NAME}' does not expose class '{CLASS_NAME}'. "
            f"Available attributes include: {dir(mod)[:30]} ..."
        )

    Solver = getattr(mod, CLASS_NAME)

    pybind_test_dir = prepare_pybind_test_dir(clear=args.clear_pybind_test)
    os.chdir(pybind_test_dir)

    print(f"[Info] Working directory changed to: {os.getcwd()}")

    write_output = not args.no_write_output
    print(f"[Info] module = {MODULE_NAME}")
    print(f"[Info] class = {CLASS_NAME}")
    print(f"[Info] bottom wall segments = {N_SEG}")
    print(f"[Info] write_output = {write_output}")
    print(f"[Info] warmup_time = {args.warmup_time}")
    print(f"[Info] control_horizon = {args.control_horizon}")
    print(f"[Info] sample_dt = {args.sample_dt}")
    print(f"[Info] amplitude = {args.amplitude}")
    print(f"[Info] mean_temperature = {args.mean_temperature}")

    output_dir = mkdir(os.path.join(pybind_test_dir, "fixed_action_results"))
    summary_csv = os.path.join(output_dir, "open_loop_summary.csv")
    if os.path.exists(summary_csv):
        os.remove(summary_csv)

    cases = FIXED_ACTIONS if not args.skip_zero_baseline else FIXED_ACTIONS[1:]

    summaries = []
    for i, (case_name, action) in enumerate(cases):
        parallel_env = args.parallel_env_base + i
        episode_env = args.episode_env_base + i

        summary = run_one_fixed_action(
            Solver=Solver,
            case_name=case_name,
            action=action,
            parallel_env=parallel_env,
            episode_env=episode_env,
            reload_particles=args.reload_particles,
            write_output=write_output,
            warmup_time=args.warmup_time,
            control_horizon=args.control_horizon,
            sample_dt=args.sample_dt,
            amplitude=args.amplitude,
            mean_temperature=args.mean_temperature,
            output_dir=output_dir,
        )
        summaries.append(summary)
        append_csv_row(summary_csv, SUMMARY_FIELDS, summary)

    print("\n" + "=" * 78)
    print("[Open-loop fixed-action test complete]")
    print(f"[Info] Summary CSV: {summary_csv}")
    print(f"[Info] Per-case metrics CSVs are under: {output_dir}")

    print("\n[Compact summary]")
    for s in summaries:
        print(
            f"{s['case_name']:>24s} | "
            f"entry={s['target_entry_time']} | "
            f"inside_time={s['time_centroid_in_target']:.4f} | "
            f"final_center=({s['final_center_x']:.4f}, {s['final_center_y']:.4f}) | "
            f"max_def={s['max_deformation_index']:.4f} | "
            f"max_area_err={s['max_area_error']:.4f} "
        )


if __name__ == "__main__":
    run_case()
