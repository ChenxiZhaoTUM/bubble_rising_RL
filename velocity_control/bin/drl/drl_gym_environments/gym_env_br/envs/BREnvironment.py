#!/usr/bin/env python3
"""
Gymnasium environment for the bottom-wall five-segment bubble-rising task.

Task:
    Single objective: reach center_y >= target_height as fast as possible.

C++/pybind expected:
    PYBIND11_MODULE(br_2d_bubble_rising_bottom_python, m)
    class name: bubble_rising_heat_from_sph_cpp

Required bottom-wall control methods:
    set_bottom_wall_segment_actions(actions, amplitude, mean_temperature)
    set_bottom_wall_segment_temperatures(temperatures, enforce_mean, mean_temperature)
    get_bottom_wall_segment_temperatures()

Important:
    The bottom-wall average temperature is still fixed by C++ because actions are
    mapped as:
        T_i = mean_temperature + amplitude * (a_i - mean(a))
    Therefore increasing action_amplitude strengthens the spatial temperature
    contrast without increasing the mean bottom-wall temperature.
"""

import os
import sys
import csv
import glob
import argparse
import importlib.util
from typing import Optional
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# Pybind module configuration.
# =============================================================================
MODULE_NAME = "br_2d_bubble_rising_bottom_python"
CLASS_NAME = "bubble_rising_heat_from_sph_cpp"

ENV_DIR = os.environ.get("SPH_PYBIND_LIB_DIR", "").strip()


# =============================================================================
# Loader helpers.
# =============================================================================
def _candidate_dirs() -> list[str]:
    """
    Robustly search upward from this file until a project root containing lib/
    is found. This works when this script is inside:
        bin/drl/drl_gym_environments/gym_env_br/envs/
    """
    dirs: list[str] = []

    if ENV_DIR:
        dirs.append(os.path.abspath(ENV_DIR))

    here = os.path.abspath(os.path.dirname(__file__))

    cur = here
    for _ in range(12):
        lib = os.path.join(cur, "lib")

        for cfg in ("Release", "RelWithDebInfo", "Debug", ""):
            d = os.path.join(lib, cfg) if cfg else lib
            dirs.append(d)

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    seen = set()
    uniq: list[str] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)

    return uniq


def _find_project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))

    cur = here
    for _ in range(12):
        if os.path.isdir(os.path.join(cur, "lib")):
            return cur

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return os.getcwd()


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
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]

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
            "  - Make sure PYBIND11_MODULE name equals br_2d_bubble_rising_bottom_python.",
            "  - Make sure the generated .pyd/.so is under lib/Release, lib/Debug, or SPH_PYBIND_LIB_DIR.",
            "  - Make sure Python version and architecture match the compiled extension.",
            "  - On Windows, expected file looks like br_2d_bubble_rising_bottom_python.cp310-win_amd64.pyd.",
        ]

        raise ImportError("\n".join(msg)) from e


# =============================================================================
# Utility.
# =============================================================================
def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _episode_output_dir(training_root: str, parallel_envs: int, episode: int) -> str:
    path = os.path.join(
        os.path.abspath(training_root),
        f"output_env_{parallel_envs}_episode_{episode}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def _append_csv_row(path: str, fieldnames: list[str], row: dict, reset_file: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if reset_file and os.path.exists(path):
        os.remove(path)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()

        writer.writerow(row)


def _read_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _read_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


# ------------------------------------------------------------------
# Pure physical metrics output.
# Keep the fields identical to the reference environment.
# ------------------------------------------------------------------
BUBBLE_METRICS_FIELDS = [
    "time",
    "rl_step",
    "number_of_iterations",

    "center_x",
    "center_y",
    "center_u",
    "center_v",

    "x_min",
    "x_max",
    "y_min",
    "y_max",

    "bubble_width",
    "bubble_height",
    "deformation_index",
    "aspect_ratio",

    "bubble_area",
    "area_ratio",

    "centroid_in_target",
    "reached_target_height",

    "left_particle_in_target",
    "right_particle_in_target",
    "bottom_particle_in_target",
    "top_particle_in_target",

    "all_extreme_particles_in_target",
]


def _write_bubble_metrics_csv(
    output_dir: str,
    time_value: float,
    rl_step: int,
    number_of_iterations: int,
    metrics: dict,
    reset_file: bool = False,
):
    """
    Write physical bubble metrics to:
        output_dir/bubble_control_metrics.csv

    This intentionally does not include reward/action-specific diagnostics.
    """
    row = {
        "time": float(time_value),
        "rl_step": int(rl_step),
        "number_of_iterations": int(number_of_iterations),

        "center_x": float(metrics["center_x"]),
        "center_y": float(metrics["center_y"]),
        "center_u": float(metrics["center_u"]),
        "center_v": float(metrics["center_v"]),

        "x_min": float(metrics["x_min"]),
        "x_max": float(metrics["x_max"]),
        "y_min": float(metrics["y_min"]),
        "y_max": float(metrics["y_max"]),

        "bubble_width": float(metrics["bubble_width"]),
        "bubble_height": float(metrics["bubble_height"]),
        "deformation_index": float(metrics["deformation_index"]),
        "aspect_ratio": float(metrics["aspect_ratio"]),

        "bubble_area": float(metrics["bubble_area"]),
        "area_ratio": float(metrics["area_ratio"]),

        "centroid_in_target": int(metrics["centroid_in_target"]),
        "reached_target_height": int(metrics["reached_target_height"]),

        "left_particle_in_target": int(metrics["left_particle_in_target"]),
        "right_particle_in_target": int(metrics["right_particle_in_target"]),
        "bottom_particle_in_target": int(metrics["bottom_particle_in_target"]),
        "top_particle_in_target": int(metrics["top_particle_in_target"]),

        "all_extreme_particles_in_target": int(
            metrics["all_extreme_particles_in_target"]
        ),
    }

    path = os.path.join(output_dir, "bubble_control_metrics.csv")

    _append_csv_row(
        path=path,
        fieldnames=BUBBLE_METRICS_FIELDS,
        row=row,
        reset_file=reset_file,
    )


# =============================================================================
# Single-agent fast-rising environment.
# =============================================================================
class BubbleRisingPositionEnv(gym.Env):
    """
    Single-agent RL environment for the fast-rising bottom-wall task.

    Action:
        R^5 in [-1, 1].
        C++ converts action to five bottom-wall segment temperatures while
        enforcing fixed mean temperature.

    Observation:
        Flow field only:
            [u0, v0, T0, u1, v1, T1, ...]

    Reward:
        Single objective:
            reach center_y >= target_height as early as possible.

        The reward compares height and vertical velocity against a passive
        zero-action baseline trajectory generated outside the replay buffer.
    """

    metadata = {}

    def __init__(
        self,
        render_mode=None,
        parallel_envs: int = 0,
        n_seg: int = 5,
        training_root: str | None = None,
        reload_particles: bool = False,
        write_output: bool = False,
        warmup_time: float = 0.0,
        delta_time: float = 0.02,
        max_steps_per_episode: int = 225,
        episode_max_time: float = 4.5,
        n_probe_points: int = 400,
        action_amplitude: float = 0.7,
        mean_temperature: float = 1.0,
        target_height: float = 1.5,
        baseline_metrics_csv: str | None = None,
        baseline_entry_time: float | None = None,
        terminate_on_success: bool = True,
    ):
        super().__init__()

        self.parallel_envs = int(parallel_envs)
        self.episode = 1

        self.n_seg = int(n_seg)
        if self.n_seg != 5:
            raise ValueError(
                f"This bottom-wall task expects n_seg=5, got n_seg={self.n_seg}."
            )

        self.reload_particles = bool(reload_particles)
        self.write_output = bool(write_output)

        self.warmup_time = float(warmup_time)
        self.delta_time = float(delta_time)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.max_steps_per_episode_eval = 4 * self.max_steps_per_episode
        self.episode_max_time = float(episode_max_time)
        self.terminate_on_success = bool(terminate_on_success)
        self.deterministic = False

        self.n_probe_points = int(n_probe_points)
        self.flow_obs_len = 3 * self.n_probe_points

        self.action_amplitude = float(action_amplitude)
        self.mean_temperature = float(mean_temperature)
        self.target_height = float(target_height)

        # Domain constants. Must match C++ case.
        self.DL = 2.0
        self.DH = 2.0
        self.gravity_g = 0.98
        self.U_f = float(np.sqrt(self.gravity_g * self.DH))

        # ------------------------------------------------------------------
        # Folders.
        # ------------------------------------------------------------------
        if training_root is None:
            project_root = _find_project_root()
            training_root = os.path.join(
                project_root,
                "training_process",
                "bubble_rising_fast_y15_bottom5",
            )

        self.training_root = _mkdir(os.path.abspath(training_root))

        for name in ("input", "output", "reload", "restart"):
            _mkdir(os.path.join(self.training_root, name))

        self.log_dir = _mkdir(
            os.path.join(self.training_root, f"logs_env_{self.parallel_envs}")
        )

        # ------------------------------------------------------------------
        # Gym spaces.
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_seg,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0e6,
            high=1.0e6,
            shape=(self.flow_obs_len,),
            dtype=np.float32,
        )

        # Moving-average buffer for flow observation only.
        self._obs_hist = deque(maxlen=4)

        # ------------------------------------------------------------------
        # Reward weights.
        # Keep this reward intentionally simple for the single target.
        # ------------------------------------------------------------------
        self.w_time = 0.02
        self.w_height_advantage = 20.0
        self.w_vertical_velocity_advantage = 3.0

        # If a controlled episode reaches the target earlier by 0.04 s, this
        # contributes roughly +4 reward.
        self.w_fast_time = 100.0
        self.w_success = 5.0

        # Very small action regularization. Do not suppress control authority.
        self.w_action = 0.001
        self.w_smooth = 0.002

        # Safety terms. These are deliberately weak unless the bubble is close
        # to breakdown.
        self.deformation_warning = 0.40
        self.area_warning = 0.08
        self.w_deformation_excess = 0.0
        self.w_area_excess = 0.0

        # Hard termination thresholds.
        self.deformation_break = 0.60
        self.area_break = 0.35
        self.w_break = 8.0

        # ------------------------------------------------------------------
        # Runtime state.
        # ------------------------------------------------------------------
        self.mod = ensure_module()
        if not hasattr(self.mod, CLASS_NAME):
            raise AttributeError(
                f"Module '{MODULE_NAME}' does not expose class '{CLASS_NAME}'."
            )

        self.Solver = getattr(self.mod, CLASS_NAME)
        self.sim = None

        self.sim_time = 0.0
        self.step_count = 0
        self.total_reward_per_episode = 0.0
        self.has_reached_task_height = False
        self.first_reach_time: float | None = None

        self.last_action = np.zeros(self.n_seg, dtype=np.float32)

        self.ref_center_y = 0.0
        self.prev_center_y = 0.0
        self.ref_area_ratio = 1.0

        self.last_reset_info = {}

        # Passive baseline loaded from a separate zero-action rollout.
        self.baseline_metrics_csv = baseline_metrics_csv
        self.baseline_metrics_by_step: dict[int, dict] = {}
        self.baseline_entry_time: float | None = (
            None if baseline_entry_time is None else float(baseline_entry_time)
        )

        self._load_baseline_metrics_if_available()

    # ------------------------------------------------------------------
    # Baseline handling.
    # ------------------------------------------------------------------
    def _load_baseline_metrics_if_available(self):
        self.baseline_metrics_by_step = {}

        if not self.baseline_metrics_csv:
            return

        path = os.path.abspath(self.baseline_metrics_csv)
        if not os.path.isfile(path):
            print(f"[Baseline] File not found, dense baseline disabled: {path}")
            return

        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    step = _read_int(row, "rl_step", default=-1)
                    if step < 0:
                        continue

                    parsed = {
                        "time": _read_float(row, "time"),
                        "center_y": _read_float(row, "center_y"),
                        "center_v": _read_float(row, "center_v"),
                        "deformation_index": _read_float(row, "deformation_index"),
                        "area_ratio": _read_float(row, "area_ratio", default=1.0),
                    }

                    self.baseline_metrics_by_step[step] = parsed

                    if (
                        self.baseline_entry_time is None
                        and parsed["center_y"] >= self.target_height
                    ):
                        self.baseline_entry_time = parsed["time"]

            print(f"[Baseline] Loaded {len(self.baseline_metrics_by_step)} rows from: {path}")
            print(f"[Baseline] baseline_entry_time = {self.baseline_entry_time}")

        except Exception as e:
            print(f"[Baseline] Failed to load baseline CSV: {path}")
            print(f"[Baseline] Error: {e}")
            self.baseline_metrics_by_step = {}

    def _baseline_metrics_for_step(self, step_count: int) -> dict | None:
        if step_count in self.baseline_metrics_by_step:
            return self.baseline_metrics_by_step[step_count]

        if not self.baseline_metrics_by_step:
            return None

        # Fallback to the nearest earlier baseline row.
        available = [k for k in self.baseline_metrics_by_step.keys() if k <= step_count]
        if not available:
            return None

        return self.baseline_metrics_by_step[max(available)]

    # ------------------------------------------------------------------
    # Action handling.
    # ------------------------------------------------------------------
    def _sanitize_action(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)

        if a.size != self.n_seg:
            raise ValueError(
                f"Action must have length {self.n_seg}, "
                f"got shape={np.asarray(action).shape}"
            )

        return np.clip(a, -1.0, 1.0)

    def _send_action_to_cpp(self, action: np.ndarray) -> list[float]:
        """
        Send raw action to C++ bottom-wall controller.

        C++ performs the fixed-mean transformation:
            T_i = mean_temperature + amplitude * (a_i - mean(a)).
        """
        self.sim.set_bottom_wall_segment_actions(
            action.astype(float).tolist(),
            self.action_amplitude,
            self.mean_temperature,
        )

        return list(self.sim.get_bottom_wall_segment_temperatures())

    # ------------------------------------------------------------------
    # Observation: flow only.
    # ------------------------------------------------------------------
    def _instant_observation(self) -> np.ndarray:
        obs = self.sim.get_flow_observation()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        expected = self.observation_space.shape[0]
        if obs.size != expected:
            raise RuntimeError(
                f"Observation length mismatch. C++ returned {obs.size}, "
                f"but observation_space expects {expected}. "
                f"Check n_probe_points or C++ createObservationPoints()."
            )

        return obs

    def _read_observation(self) -> np.ndarray:
        snap = self._instant_observation()
        self._obs_hist.append(snap)

        obs = np.mean(list(self._obs_hist), axis=0).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Reward.
    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray, metrics: dict) -> tuple[float, dict]:
        center_x = float(metrics["center_x"])
        center_y = float(metrics["center_y"])
        center_u = float(metrics["center_u"])
        center_v = float(metrics["center_v"])

        area_ratio_raw = float(metrics["area_ratio"])
        deformation_index = float(metrics["deformation_index"])

        area_rel = area_ratio_raw / (self.ref_area_ratio + 1.0e-12)
        area_error = abs(area_rel - 1.0)

        task_reached = bool(center_y >= self.target_height)

        if task_reached and not self.has_reached_task_height:
            self.has_reached_task_height = True
            self.first_reach_time = float(self.sim_time)

        deformation_break = bool(deformation_index >= self.deformation_break)
        area_break = bool(area_error >= self.area_break)
        bubble_broken = bool(deformation_break or area_break)

        baseline_metrics = self._baseline_metrics_for_step(self.step_count)
        baseline_available = baseline_metrics is not None

        baseline_center_y = self.ref_center_y
        baseline_center_v = 0.0
        baseline_deformation = 0.0
        baseline_area_error = 0.0

        if baseline_metrics is not None:
            baseline_center_y = float(baseline_metrics["center_y"])
            baseline_center_v = float(baseline_metrics["center_v"])
            baseline_deformation = float(baseline_metrics["deformation_index"])
            baseline_area_rel = (
                float(baseline_metrics["area_ratio"])
                / (self.ref_area_ratio + 1.0e-12)
            )
            baseline_area_error = abs(baseline_area_rel - 1.0)

        height_advantage = center_y - baseline_center_y
        velocity_advantage = (center_v - baseline_center_v) / (self.U_f + 1.0e-12)

        height_to_target = max(self.target_height - self.ref_center_y, 1.0e-12)
        height_progress = float(np.clip((center_y - self.ref_center_y) / height_to_target, 0.0, 1.5))

        dy_step = center_y - self.prev_center_y
        progress_norm = float(
            np.clip(
                dy_step / (self.U_f * self.delta_time + 1.0e-12),
                -1.0,
                1.0,
            )
        )

        deformation_excess = max(0.0, deformation_index - self.deformation_warning)
        area_excess = max(0.0, area_error - self.area_warning)

        reward = 0.0

        # Small time pressure: every extra step is bad.
        time_reward = -self.w_time
        reward += time_reward

        # Dense task signal: be higher and faster than the passive trajectory
        # at the same physical step.
        height_reward = self.w_height_advantage * height_advantage
        vertical_velocity_reward = self.w_vertical_velocity_advantage * velocity_advantage
        reward += height_reward
        reward += vertical_velocity_reward

        # Safety: only penalize excessive deformation/area drift.
        deform_reward = -self.w_deformation_excess * deformation_excess
        area_reward = -self.w_area_excess * area_excess
        reward += deform_reward
        reward += area_reward

        # Minimal regularization so SAC does not saturate gratuitously.
        action_cost = self.w_action * float(np.mean(action * action))
        smooth_cost = self.w_smooth * float(np.mean((action - self.last_action) ** 2))
        reward -= action_cost
        reward -= smooth_cost

        entry_time_gain = 0.0
        success_reward = 0.0

        if task_reached:
            if self.baseline_entry_time is not None:
                entry_time_gain = float(self.baseline_entry_time - self.sim_time)
            success_reward = self.w_success + self.w_fast_time * entry_time_gain
            reward += success_reward

        if bubble_broken:
            reward -= self.w_break

        self.prev_center_y = center_y

        info = {
            "center_x": center_x,
            "center_y": center_y,
            "center_u": center_u,
            "center_v": center_v,

            "target_height": self.target_height,
            "task_reached_target_height": int(task_reached),
            "raw_cpp_reached_target_height": int(metrics["reached_target_height"]),

            "area_ratio": area_ratio_raw,
            "area_rel": area_rel,
            "area_error": area_error,
            "deformation_index": deformation_index,

            "centroid_in_target": int(metrics["centroid_in_target"]),
            "reached_target_height": int(task_reached),
            "all_extreme_particles_in_target": int(metrics["all_extreme_particles_in_target"]),

            "deformation_break": deformation_break,
            "area_break": area_break,
            "bubble_broken": bubble_broken,

            "progress_norm": progress_norm,
            "height_progress": height_progress,
            "height_advantage": height_advantage,
            "velocity_advantage": velocity_advantage,
            "entry_time_gain": entry_time_gain,

            "baseline_available": baseline_available,
            "baseline_center_y": baseline_center_y,
            "baseline_center_v": baseline_center_v,
            "baseline_entry_time": self.baseline_entry_time,
            "baseline_deformation": baseline_deformation,
            "baseline_area_error": baseline_area_error,

            "time_reward": time_reward,
            "height_reward": height_reward,
            "vertical_velocity_reward": vertical_velocity_reward,
            "success_reward": success_reward,
            "area_reward": area_reward,
            "deform_reward": deform_reward,
            "action_cost": action_cost,
            "smooth_cost": smooth_cost,

            "raw_reward": float(reward),
        }

        return float(reward), info

    # ------------------------------------------------------------------
    # Logging.
    # ------------------------------------------------------------------
    def _open_episode_logs(self):
        _mkdir(self.log_dir)

        episode_msg = f"[env {self.parallel_envs}] ===== Episode {self.episode} start ====="
        print(episode_msg)

        with open(os.path.join(self.log_dir, "episodes.txt"), "a", encoding="utf-8") as f:
            f.write(episode_msg + "\n")

        open(
            os.path.join(
                self.log_dir,
                f"action_env{self.parallel_envs}_epi{self.episode}.txt",
            ),
            "w",
        ).close()

        open(
            os.path.join(
                self.log_dir,
                f"reward_env{self.parallel_envs}_epi{self.episode}.txt",
            ),
            "w",
        ).close()

    def _log_step(self, action: np.ndarray, seg_temps: list[float], reward: float, info: dict):
        action_log = os.path.join(
            self.log_dir,
            f"action_env{self.parallel_envs}_epi{self.episode}.txt",
        )

        reward_log = os.path.join(
            self.log_dir,
            f"reward_env{self.parallel_envs}_epi{self.episode}.txt",
        )

        with open(action_log, "a", encoding="utf-8") as f:
            f.write(
                f"clock: {self.sim_time:.6f} "
                f"raw_action: {action.tolist()} "
                f"bottom_seg_temps: {seg_temps}\n"
            )

        with open(reward_log, "a", encoding="utf-8") as f:
            f.write(
                f"clock: {self.sim_time:.6f} "
                f"reward: {reward:.6f} "
                f"center: ({info['center_x']:.6f}, {info['center_y']:.6f}) "
                f"vel: ({info['center_u']:.6f}, {info['center_v']:.6f}) "
                f"target_height: {info['target_height']:.6f} "
                f"reached: {info['task_reached_target_height']} "
                f"baseline_center_y: {info['baseline_center_y']:.6f} "
                f"height_advantage: {info['height_advantage']:.6f} "
                f"entry_time_gain: {info['entry_time_gain']:.6f} "
                f"deformation: {info['deformation_index']:.6f} "
                f"area_ratio: {info['area_ratio']:.6f} "
                f"area_rel: {info['area_rel']:.6f} "
                f"area_error: {info['area_error']:.6f} "
                f"height_reward: {info['height_reward']:.6f} "
                f"vertical_velocity_reward: {info['vertical_velocity_reward']:.6f} "
                f"success_reward: {info['success_reward']:.6f} "
                f"deformation_break: {info['deformation_break']} "
                f"area_break: {info['area_break']} "
                f"broken: {info['bubble_broken']} "
                f"raw_reward: {info['raw_reward']:.6f}\n"
            )

    def _log_episode_end(self):
        summary_path = os.path.join(
            self.log_dir,
            f"reward_env{self.parallel_envs}.txt",
        )

        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"episode: {self.episode} "
                f"total_reward: {self.total_reward_per_episode:.6f} "
                f"first_reach_time: {self.first_reach_time}\n"
            )

    # ------------------------------------------------------------------
    # Gym API: reset.
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._open_episode_logs()

        os.chdir(self.training_root)

        self.sim = self.Solver(
            self.parallel_envs,
            self.episode,
            self.reload_particles,
            self.write_output,
        )

        # Uniform bottom-wall temperature at reset.
        self.sim.set_bottom_wall_segment_temperatures(
            [self.mean_temperature] * self.n_seg,
            True,
            self.mean_temperature,
        )

        self.sim_time = 0.0
        self.step_count = 0
        self.total_reward_per_episode = 0.0
        self.has_reached_task_height = False
        self.first_reach_time = None
        self.last_action = np.zeros(self.n_seg, dtype=np.float32)

        self._obs_hist.clear()

        if self.warmup_time > 0.0:
            self.sim.run_case(self.warmup_time)
            self.sim_time = float(self.sim.get_physical_time())

        metrics0 = self.sim.get_bubble_metrics_dict()

        output_dir = _episode_output_dir(
            self.training_root,
            self.parallel_envs,
            self.episode,
        )

        _write_bubble_metrics_csv(
            output_dir=output_dir,
            time_value=self.sim_time,
            rl_step=0,
            number_of_iterations=self.sim.get_number_of_iterations(),
            metrics=metrics0,
            reset_file=True,
        )

        self.ref_center_y = float(metrics0["center_y"])
        self.prev_center_y = float(metrics0["center_y"])
        self.ref_area_ratio = max(float(metrics0["area_ratio"]), 1.0e-12)

        self.has_reached_task_height = bool(self.ref_center_y >= self.target_height)
        if self.has_reached_task_height:
            self.first_reach_time = self.sim_time

        snap0 = self._instant_observation()
        for _ in range(4):
            self._obs_hist.append(snap0.copy())

        obs0 = self._read_observation()

        self.last_reset_info = {
            "episode": self.episode,
            "physical_time": self.sim_time,
            "metrics": metrics0,
            "ref_center_y": self.ref_center_y,
            "ref_area_ratio": self.ref_area_ratio,
            "bottom_wall_segment_temperatures": list(
                self.sim.get_bottom_wall_segment_temperatures()
            ),
            "target_height": self.target_height,
            "baseline_metrics_csv": self.baseline_metrics_csv,
            "baseline_entry_time": self.baseline_entry_time,
            "action_amplitude": self.action_amplitude,
        }

        # Keep reset info empty for Tianshou compatibility.
        return obs0, {}

    # ------------------------------------------------------------------
    # Gym API: step.
    # ------------------------------------------------------------------
    def step(self, action):
        applied_action = self._sanitize_action(action)
        seg_temps = self._send_action_to_cpp(applied_action)

        end_time = min(
            self.sim_time + self.delta_time,
            self.episode_max_time,
        )

        self.sim.run_case(end_time)
        self.step_count += 1
        self.sim_time = float(self.sim.get_physical_time())

        output_dir = _episode_output_dir(
            self.training_root,
            self.parallel_envs,
            self.episode,
        )

        metrics = self.sim.get_bubble_metrics_dict()

        _write_bubble_metrics_csv(
            output_dir=output_dir,
            time_value=self.sim_time,
            rl_step=self.step_count,
            number_of_iterations=self.sim.get_number_of_iterations(),
            metrics=metrics,
            reset_file=False,
        )

        obs = self._read_observation()

        reward, info = self._compute_reward(applied_action, metrics)
        self.total_reward_per_episode += reward

        self._log_step(applied_action, seg_temps, reward, info)

        terminated = bool(info["bubble_broken"])
        if self.terminate_on_success:
            terminated = bool(terminated or info["task_reached_target_height"])

        episode_limit = (
            self.max_steps_per_episode_eval
            if self.deterministic
            else self.max_steps_per_episode
        )

        truncated = bool(
            self.step_count >= episode_limit
            or self.sim_time >= self.episode_max_time - 1.0e-12
        )

        info.update(
            {
                "episode": self.episode,
                "step_count": self.step_count,
                "physical_time": float(self.sim.get_physical_time()),
                "segment_temperatures": seg_temps,
                "total_reward": self.total_reward_per_episode,
                "applied_action": applied_action.copy(),
                "policy_action": applied_action.copy(),
                "first_reach_time": self.first_reach_time,
            }
        )

        if terminated or truncated:
            self._log_episode_end()
            self.episode += 1

        self.last_action = applied_action.copy()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return 0

    def _render_frame(self):
        return 0

    def close(self):
        self.sim = None
        return 0


# =============================================================================
# Optional smoke test.
# =============================================================================
def _smoke_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel_env", default=0, type=int)
    parser.add_argument("--n_steps", default=3, type=int)
    parser.add_argument("--write_output", action="store_true")
    parser.add_argument("--reload_particles", action="store_true")
    parser.add_argument("--training_root", default=None, type=str)
    parser.add_argument("--action_amplitude", default=0.7, type=float)
    parser.add_argument("--target_height", default=1.5, type=float)
    args = parser.parse_args()

    env = BubbleRisingPositionEnv(
        parallel_envs=args.parallel_env,
        training_root=args.training_root,
        reload_particles=args.reload_particles,
        write_output=args.write_output,
        warmup_time=0.0,
        delta_time=0.02,
        max_steps_per_episode=225,
        episode_max_time=4.5,
        n_probe_points=400,
        action_amplitude=args.action_amplitude,
        target_height=args.target_height,
    )

    obs, info = env.reset()
    print("[reset] obs shape:", obs.shape)
    print("[reset] info:", info)
    print("[reset] last_reset_info:", env.last_reset_info)

    for k in range(args.n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"[step {k}] reward={reward:.6f}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"center=({info['center_x']:.4f}, {info['center_y']:.4f}), "
            f"height_advantage={info['height_advantage']:.4f}, "
            f"target_reached={info['task_reached_target_height']}, "
            f"deform={info['deformation_index']:.4f}, "
            f"area_rel={info['area_rel']:.4f}"
        )

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    _smoke_test()
