#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import importlib.util
from typing import Optional


# =============================================================================
# Configure your pybind module.
# This name must match PYBIND11_MODULE(br_2d_bubble_rising_heat_python, m)
# =============================================================================
MODULE_NAME = "br_2d_bubble_rising_heat_python"
CLASS_NAME = "bubble_rising_heat_from_sph_cpp"

# Optional: override search directory by environment variable.
ENV_DIR = os.environ.get("SPH_PYBIND_LIB_DIR", "").strip()


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
        dirs.append(ENV_DIR)

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
            f"Checked directories: {', '.join(checked_dirs) if checked_dirs else '(none found)'}",
            "Tips:",
            "  - Make sure PYBIND11_MODULE name is br_2d_bubble_rising_heat_python.",
            "  - Make sure the generated file is in lib/Release, lib/Debug, or SPH_PYBIND_LIB_DIR.",
            "  - Make sure Python version/architecture matches the compiled extension.",
            "  - On Windows, expected file looks like br_2d_bubble_rising_heat_python.cp310-win_amd64.pyd.",
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
    Create output root for this smoke test:
        <case>/bin/bind/pybind_test/

    All SPHinXsys-generated files will be written under this folder because
    we chdir into it before constructing the C++ solver.
    """
    bind_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(bind_dir, "pybind_test")

    if clear and os.path.isdir(test_dir):
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    mkdir(test_dir)

    # Optional skeleton. CustomIOEnvironment may create its own subfolders,
    # but these are useful for consistency.
    for name in ("input", "output", "reload", "restart"):
        mkdir(os.path.join(test_dir, name))

    return test_dir


def parse_float_list(s: str) -> list[float]:
    if not s.strip():
        return []

    return [float(x.strip()) for x in s.split(",") if x.strip()]


def summarize_observation(name: str, obs: list[float], max_items: int = 10):
    print(f"{name}: size = {len(obs)}")
    if obs:
        head = obs[:max_items]
        print(f"{name}: first {len(head)} values = {head}")


# =============================================================================
# Bubble rising heat pybind smoke test.
# =============================================================================
def run_case():
    parser = argparse.ArgumentParser()

    parser.add_argument("--parallel_env", default=0, type=int)
    parser.add_argument("--episode_env", default=0, type=int)

    parser.add_argument("--reload_particles", action="store_true")

    # For this smoke test, output is ON by default.
    parser.add_argument("--no_write_output", action="store_true")

    parser.add_argument("--clear_pybind_test", action="store_true")

    parser.add_argument("--warmup_time", default=0.02, type=float)
    parser.add_argument("--control_horizon", default=5.0, type=float)

    parser.add_argument(
        "--actions",
        default="1.0,0.2,-0.4,-0.8",
        type=str,
        help="Left-wall segment actions, comma separated.",
    )

    parser.add_argument(
        "--temperatures",
        default="",
        type=str,
        help="Direct left-wall segment temperatures, comma separated. If provided, overrides actions.",
    )

    parser.add_argument("--amplitude", default=0.3, type=float)
    parser.add_argument("--mean_temperature", default=1.0, type=float)

    args = parser.parse_args()

    mod = ensure_module()

    if not hasattr(mod, CLASS_NAME):
        raise AttributeError(
            f"Module '{MODULE_NAME}' does not expose class '{CLASS_NAME}'. "
            f"Available attributes include: {dir(mod)[:30]} ..."
        )

    Solver = getattr(mod, CLASS_NAME)

    # -------------------------------------------------------------------------
    # Important:
    # Change working directory BEFORE constructing the SPHinXsys solver.
    # All generated files will go into ./pybind_test/.
    # -------------------------------------------------------------------------
    pybind_test_dir = prepare_pybind_test_dir(clear=args.clear_pybind_test)
    os.chdir(pybind_test_dir)

    print(f"[Info] Working directory changed to: {os.getcwd()}")

    write_output = not args.no_write_output

    print("\n=== Stage A: construct bubble rising heat solver ===")
    sim = Solver(
        args.parallel_env,
        args.episode_env,
        args.reload_particles,
        write_output,
    )

    print("[Info] Solver constructed.")
    print(f"[Info] write_output   = {write_output}")
    print(f"[Info] physical_time  = {sim.get_physical_time()}")
    print(f"[Info] iteration      = {sim.get_number_of_iterations()}")

    print("\n=== Stage B: initial observations ===")
    bubble_metrics_0 = sim.get_bubble_metrics_dict()
    bubble_obs_0 = sim.get_bubble_observation()
    flow_obs_0 = sim.get_flow_observation()

    print("[Initial bubble metrics]")
    for k, v in bubble_metrics_0.items():
        print(f"  {k}: {v}")

    summarize_observation("bubble_obs_0", bubble_obs_0)
    summarize_observation("flow_obs_0", flow_obs_0)

    print("\n=== Stage C: warm-up run ===")
    if args.warmup_time > 0.0:
        target_time = sim.get_physical_time() + args.warmup_time
        print(f"[Info] Running to t = {target_time}")
        sim.run_case(target_time)

    print(f"[After warmup] physical_time = {sim.get_physical_time()}")
    print(f"[After warmup] iteration     = {sim.get_number_of_iterations()}")

    bubble_metrics_warm = sim.get_bubble_metrics_dict()
    print("[Warmup bubble metrics]")
    for k, v in bubble_metrics_warm.items():
        print(f"  {k}: {v}")

    print("\n=== Stage D: apply left-wall control ===")

    temperatures = parse_float_list(args.temperatures)
    actions = parse_float_list(args.actions)

    if temperatures:
        print(f"[Info] Setting direct left-wall segment temperatures = {temperatures}")
        sim.set_left_wall_segment_temperatures(
            temperatures,
            True,
            args.mean_temperature,
        )
    else:
        print(f"[Info] Setting left-wall segment actions = {actions}")
        print(f"[Info] amplitude = {args.amplitude}, mean_temperature = {args.mean_temperature}")
        sim.set_left_wall_segment_actions(
            actions,
            args.amplitude,
            args.mean_temperature,
        )

    current_temps = sim.get_left_wall_segment_temperatures()
    print(f"[Info] Current left-wall segment temperatures = {current_temps}")
    print(f"[Info] Mean segment temperature = {sum(current_temps) / len(current_temps)}")

    print("\n=== Stage E: controlled run ===")
    target_time = sim.get_physical_time() + args.control_horizon
    print(f"[Info] Running to t = {target_time}")
    sim.run_case(target_time)

    print(f"[After control] physical_time = {sim.get_physical_time()}")
    print(f"[After control] iteration     = {sim.get_number_of_iterations()}")

    print("\n=== Stage F: final observations ===")
    bubble_metrics = sim.get_bubble_metrics_dict()
    bubble_obs = sim.get_bubble_observation()
    flow_obs = sim.get_flow_observation()

    print("[Final bubble metrics]")
    for k, v in bubble_metrics.items():
        print(f"  {k}: {v}")

    summarize_observation("bubble_obs", bubble_obs)
    summarize_observation("flow_obs", flow_obs)

    print("\n=== Stage G: termination checks ===")
    print(f"has_reached_target_height        = {sim.has_reached_target_height()}")
    print(f"is_bubble_in_target_region       = {sim.is_bubble_in_target_region()}")
    print(f"is_whole_bubble_in_target_region = {sim.is_whole_bubble_in_target_region()}")
    print(f"is_bubble_broken                 = {sim.is_bubble_broken()}")

    print("\n[Info] Bubble rising heat pybind smoke run complete.")
    print(f"[Info] Generated files are under: {pybind_test_dir}")


if __name__ == "__main__":
    run_case()