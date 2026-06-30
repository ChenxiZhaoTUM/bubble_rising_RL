#!/usr/bin/env python3
import os
import sys
import csv
import glob
import argparse
import importlib.util
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# =============================================================================
# Pybind module configuration.
# Must match:
#     PYBIND11_MODULE(br_2d_bubble_rising_heat_python, m)
# and the compiled .pyd/.so basename.
# =============================================================================
MODULE_NAME = "br_2d_bubble_rising_heat_python"
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
    dirs = []

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
    uniq = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)

    return uniq


def _find_project_root() -> str:
    """
    Find project root by searching upward for a directory containing lib/.
    For your case this should be:
        D:/bubble_rising_RL/position_control
    """
    here = os.path.abspath(os.path.dirname(__file__))

    cur = here
    for _ in range(12):
        if os.path.isdir(os.path.join(cur, "lib")):
            return cur

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Fallback: current working directory.
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
            "  - Make sure PYBIND11_MODULE name equals MODULE_NAME.",
            "  - Make sure the generated .pyd/.so is under lib/Release, lib/Debug, or SPH_PYBIND_LIB_DIR.",
            "  - Make sure Python version and architecture match the compiled extension.",
            "  - On Windows, expected file looks like br_2d_bubble_rising_heat_python.cp310-win_amd64.pyd.",
        ]

        raise ImportError("\n".join(msg)) from e


# =============================================================================
# Utility.
# =============================================================================
def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


# ------------------------------------------------------------------
# CSV / metrics output helpers.
# Keep these outside the Gym environment class.
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


def _episode_output_dir(training_root: str, parallel_envs: int, episode: int) -> str:
    """
    Return the same episode output folder used by the C++ IO environment.
    Example:
        training_root/output_env_0_episode_1
    """
    path = os.path.join(
        os.path.abspath(training_root),
        f"output_env_{parallel_envs}_episode_{episode}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def _append_csv_row(path: str, fieldnames: list[str], row: dict, reset_file: bool = False):
    """
    Append one row to a CSV file.
    If reset_file=True, remove the old file first.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if reset_file and os.path.exists(path):
        os.remove(path)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()

        writer.writerow(row)


def _write_bubble_metrics_csv(
    output_dir: str,
    time_value: float,
    rl_step: int,
    number_of_iterations: int,
    metrics: dict,
    reset_file: bool = False,
):
    """
    Write pure physical bubble metrics to:
        output_dir/bubble_control_metrics.csv

    This file deliberately excludes:
        reward,
        action,
        seg_temps,
        progress_norm,
        height_progress,
        penalties,
        bubble_broken.
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
# Single-agent bubble rising position-control environment.
# =============================================================================
class BubbleRisingPositionEnv(gym.Env):
    """
    Single-agent RL environment for bubble rising position control.

    Action:
        R^4 in [-1, 1].
        C++ converts action to four left-wall segment temperatures while
        enforcing fixed mean temperature.

    Observation:
        Flow field only:
            [u0, v0, T0, u1, v1, T1, ...]

    Reward:
        Python-side reward computed only from bubble metrics:
            center position / velocity,
            area ratio,
            deformation index,
            target-region flags.
    """

    metadata = {}

    def __init__(
        self,
        render_mode=None,
        parallel_envs: int = 0,
        n_seg: int = 4,
        training_root: str | None = None,
        reload_particles: bool = True,
        write_output: bool = False,
        warmup_time: float = 0.0,
        delta_time: float = 0.02,
        max_steps_per_episode: int = 250,
        n_probe_points: int = 400,
        action_amplitude: float = 0.3,
        mean_temperature: float = 1.0,
        use_first_episode_as_baseline: bool = True,
        baseline_parallel_env: int | None = None,
        passive_target_time: float = 0.835632,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Basic settings.
        # ------------------------------------------------------------------
        self.parallel_envs = int(parallel_envs)
        self.episode = 1

        self.n_seg = int(n_seg)
        self.reload_particles = bool(reload_particles)
        self.write_output = bool(write_output)

        self.warmup_time = float(warmup_time)
        self.delta_time = float(delta_time)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.max_steps_per_episode_eval = 4 * self.max_steps_per_episode
        self.deterministic = False

        self.n_probe_points = int(n_probe_points)
        self.flow_obs_len = 3 * self.n_probe_points

        self.action_amplitude = float(action_amplitude)
        self.mean_temperature = float(mean_temperature)

        # ------------------------------------------------------------------
        # Domain constants. Must match C++ case.
        # ------------------------------------------------------------------
        self.DL = 2.0
        self.DH = 2.0
        self.gravity_g = 0.98
        self.U_f = float(np.sqrt(self.gravity_g * self.DH))

        # Target region:
        # x / DL in [1/3, 2/3], y / DH in [1/3, 2/3].
        self.target_x_min = self.DL / 3.0
        self.target_x_max = 2.0 * self.DL / 3.0
        self.target_y_min = self.DH / 3.0
        self.target_y_max = 2.0 * self.DH / 3.0

        self.target_x_center = 0.5 * self.DL
        self.target_y_center = 0.5 * self.DH

        # ------------------------------------------------------------------
        # Folders.
        # ------------------------------------------------------------------
        if training_root is None:
            project_root = _find_project_root()
            training_root = os.path.join(
                project_root,
                "training_process",
                "bubble_rising_position_single",
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

        # ------------------------------------------------------------------
        # Moving-average buffer for flow observation only.
        # Reward is not averaged.
        # ------------------------------------------------------------------
        self._obs_hist = deque(maxlen=4)

        # ------------------------------------------------------------------
        # Reward weights.
        # Objective:
        #   1. Episode 1 of env0 is a passive no-control baseline.
        #   2. The passive baseline should receive reward exactly 0.
        #   3. Later controlled episodes are rewarded only for doing better
        #      than the passive baseline: faster entry, better holding near the
        #      target center, lower velocity, smaller deformation/area error,
        #      and less time outside the target rectangle.
        # ------------------------------------------------------------------

        # Baseline control.
        # By default, every environment uses its own episode 1 as a passive
        # baseline. Set baseline_parallel_env=0 if only env0 should do this.
        self.use_first_episode_as_baseline = bool(use_first_episode_as_baseline)
        self.baseline_parallel_env = baseline_parallel_env
        self.passive_target_time = float(passive_target_time)

        # Entry reward: only rewards faster-than-passive arrival at target height.
        self.w_fast_entry = 4.0

        # Holding reward: positive only when the bubble is close to the target
        # center and slow. The passive trajectory passes through the target, but
        # it should not earn high reward just for natural rising.
        self.w_hold = 1.0
        self.center_radius_for_hold = 0.45
        self.speed_norm_for_hold = 0.18

        # Penalties/improvements relative to passive baseline.
        self.w_outside_target = 1.0
        self.w_velocity = 0.4
        self.w_area = 0.6
        self.w_deform = 1.2

        # Absolute control regularization.
        self.w_action = 0.01
        self.w_smooth = 0.02

        # Strong penalty once the bubble is judged broken.
        self.w_break = 8.0

        # Break thresholds. These are used for termination.
        self.deformation_break = 0.60
        self.area_break = 0.35

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
        self.has_entered_target_height = False

        self.last_action = np.zeros(self.n_seg, dtype=np.float32)

        # Reward references initialized in reset().
        self.ref_center_y = 0.0
        self.prev_center_y = 0.0
        self.ref_area_ratio = 1.0

        self.last_reset_info = {}

        # Passive baseline recorded from env0 episode 1.
        self.current_episode_is_baseline = False
        self.baseline_metrics_by_step: list[dict] = []
        self.baseline_entry_time: float | None = None

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
        Send raw action to C++.
        C++ set_left_wall_segment_actions performs zero-mean transformation:
            T_i = mean_temperature + amplitude * (a_i - mean(a)).
        """
        self.sim.set_left_wall_segment_actions(
            action.astype(float).tolist(),
            self.action_amplitude,
            self.mean_temperature,
        )

        return list(self.sim.get_left_wall_segment_temperatures())

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
    # Reward helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _ramp01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _target_distance_penalty(self, center_x: float, center_y: float) -> float:
        """
        Bounded target-center penalty in [0, 1].

        0 means the bubble center is exactly at the target center.
        Values near the target boundary are approximately 0.5 to 0.7.
        Far-away states are capped at 1.
        """
        half_w = 0.5 * (self.target_x_max - self.target_x_min)
        half_h = 0.5 * (self.target_y_max - self.target_y_min)

        dx = (center_x - self.target_x_center) / (half_w + 1.0e-12)
        dy = (center_y - self.target_y_center) / (half_h + 1.0e-12)

        dist = float(np.sqrt(dx * dx + dy * dy))
        return float(np.clip(dist, 0.0, 2.0) / 2.0)

    def _outside_target_penalty(self, center_x: float, center_y: float) -> float:
        """
        Bounded outside-target penalty in [0, 1].

        0 means the bubble center is inside the target rectangle.
        It increases smoothly once the center leaves the rectangle.
        """
        half_w = 0.5 * (self.target_x_max - self.target_x_min)
        half_h = 0.5 * (self.target_y_max - self.target_y_min)

        ox = max(self.target_x_min - center_x, 0.0, center_x - self.target_x_max)
        oy = max(self.target_y_min - center_y, 0.0, center_y - self.target_y_max)

        ox /= half_w + 1.0e-12
        oy /= half_h + 1.0e-12

        return float(np.clip(np.sqrt(ox * ox + oy * oy), 0.0, 1.0))

    def _target_center_distance_raw(self, center_x: float, center_y: float) -> float:
        """
        Normalized distance from the target center.

        0 means exactly at target center. A value near 1 means near the target
        rectangle boundary along x or y.
        """
        half_w = 0.5 * (self.target_x_max - self.target_x_min)
        half_h = 0.5 * (self.target_y_max - self.target_y_min)

        dx = (center_x - self.target_x_center) / (half_w + 1.0e-12)
        dy = (center_y - self.target_y_center) / (half_h + 1.0e-12)

        return float(np.sqrt(dx * dx + dy * dy))

    def _hold_score(
        self,
        center_x: float,
        center_y: float,
        center_u: float,
        center_v: float,
    ) -> tuple[float, float, float, float, float]:
        """
        Holding score in [0, 1].

        A high score requires both:
            1. center close to target center,
            2. small centroid velocity.

        Natural passive rising usually passes through the target, but its speed
        is not small, so this score prevents passive crossing from receiving
        a large positive reward.
        """
        center_dist = self._target_center_distance_raw(center_x, center_y)
        speed_norm = float(
            np.sqrt(center_u * center_u + center_v * center_v)
            / (self.U_f + 1.0e-12)
        )

        center_score = float(
            np.clip(1.0 - center_dist / (self.center_radius_for_hold + 1.0e-12), 0.0, 1.0)
        )
        speed_score = float(
            np.clip(1.0 - speed_norm / (self.speed_norm_for_hold + 1.0e-12), 0.0, 1.0)
        )
        hold_score = center_score * speed_score

        return hold_score, center_dist, speed_norm, center_score, speed_score

    def _baseline_metrics_for_step(self, step_count: int) -> dict | None:
        idx = int(step_count) - 1
        if 0 <= idx < len(self.baseline_metrics_by_step):
            return self.baseline_metrics_by_step[idx]
        return None

    # ------------------------------------------------------------------
    # Reward: computed only from bubble metrics.
    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray, metrics: dict) -> tuple[float, dict]:
        center_x = float(metrics["center_x"])
        center_y = float(metrics["center_y"])
        center_u = float(metrics["center_u"])
        center_v = float(metrics["center_v"])

        area_ratio_raw = float(metrics["area_ratio"])
        deformation_index = float(metrics["deformation_index"])

        centroid_in_target = int(metrics["centroid_in_target"])
        reached_target_height = int(metrics["reached_target_height"])
        all_extreme_in_target = int(metrics["all_extreme_particles_in_target"])

        # Area error relative to reset-time value.
        area_rel = area_ratio_raw / (self.ref_area_ratio + 1.0e-12)
        area_error = abs(area_rel - 1.0)

        deformation_break = bool(deformation_index >= self.deformation_break)
        area_break = bool(area_error >= self.area_break)
        bubble_broken = bool(deformation_break or area_break)

        # Stage transition. Only the first time reaching target height can get
        # an entry reward; passive baseline receives exactly zero reward.
        enter_bonus = 0.0
        entry_time_gain = 0.0

        if reached_target_height and not self.has_entered_target_height:
            self.has_entered_target_height = True

            if not self.current_episode_is_baseline:
                ref_entry_time = (
                    self.baseline_entry_time
                    if self.baseline_entry_time is not None
                    else self.passive_target_time
                )
                entry_time_gain = float(
                    np.clip(
                        (ref_entry_time - self.sim_time) / (ref_entry_time + 1.0e-12),
                        -1.0,
                        1.0,
                    )
                )
                enter_bonus = self.w_fast_entry * entry_time_gain

        # Vertical progress is diagnostic only. It is not rewarded directly,
        # because passive rising would otherwise receive high reward.
        dy_step = center_y - self.prev_center_y
        progress_norm = float(
            np.clip(
                dy_step / (self.U_f * self.delta_time + 1.0e-12),
                -1.0,
                1.0,
            )
        )

        height_denom = self.target_y_min - self.ref_center_y
        if abs(height_denom) < 1.0e-12:
            height_progress = 1.0
        else:
            height_progress = self._ramp01(
                (center_y - self.ref_center_y) / height_denom
            )

        outside_target_penalty = self._outside_target_penalty(center_x, center_y)
        hold_score, center_dist, speed_norm, center_score, speed_score = self._hold_score(
            center_x, center_y, center_u, center_v
        )
        speed_norm2 = speed_norm * speed_norm

        # Baseline diagnostics/defaults.
        baseline_metrics = self._baseline_metrics_for_step(self.step_count)
        baseline_available = baseline_metrics is not None

        baseline_hold_score = 0.0
        baseline_center_dist = 0.0
        baseline_speed_norm = 0.0
        baseline_speed_norm2 = 0.0
        baseline_outside_penalty = 0.0
        baseline_area_error = 0.0
        baseline_deformation = 0.0

        if baseline_metrics is not None:
            baseline_area_rel = float(baseline_metrics["area_ratio"]) / (self.ref_area_ratio + 1.0e-12)
            baseline_area_error = abs(baseline_area_rel - 1.0)
            baseline_deformation = float(baseline_metrics["deformation_index"])
            baseline_outside_penalty = self._outside_target_penalty(
                float(baseline_metrics["center_x"]),
                float(baseline_metrics["center_y"]),
            )
            (
                baseline_hold_score,
                baseline_center_dist,
                baseline_speed_norm,
                _,
                _,
            ) = self._hold_score(
                float(baseline_metrics["center_x"]),
                float(baseline_metrics["center_y"]),
                float(baseline_metrics["center_u"]),
                float(baseline_metrics["center_v"]),
            )
            baseline_speed_norm2 = baseline_speed_norm * baseline_speed_norm

        # --------------------------------------------------------------
        # Reward.
        # --------------------------------------------------------------
        if self.current_episode_is_baseline:
            # Episode 1 is a pure calibration/passive baseline. It is forced to
            # zero action in step(), and its reward is exactly zero so that the
            # baseline total reward is 0 by definition.
            reward = 0.0
            hold_reward = 0.0
            outside_reward = 0.0
            velocity_reward = 0.0
            area_reward = 0.0
            deform_reward = 0.0
            action_cost = 0.0
            smooth_cost = 0.0

        else:
            reward = 0.0

            # Faster-than-passive entry. Passive entry gives 0. Slower entry is
            # negative; faster entry is positive.
            reward += enter_bonus

            # Holding quality relative to passive trajectory at the same step.
            # If the controlled trajectory is not better than passive, this term
            # is near zero or negative.
            hold_reward = self.w_hold * (hold_score - baseline_hold_score)
            reward += hold_reward

            # Leaving the target rectangle is judged relative to passive. If the
            # controlled bubble stays in the target while passive has left, this
            # becomes positive. If it leaves earlier/farther, it becomes negative.
            outside_reward = self.w_outside_target * (
                baseline_outside_penalty - outside_target_penalty
            )
            reward += outside_reward

            # Prefer smaller velocity than passive after target height is reached.
            # Before target height, this term is weak because the main objective
            # is fast entry.
            velocity_weight = self.w_velocity if self.has_entered_target_height else 0.25 * self.w_velocity
            velocity_reward = velocity_weight * (baseline_speed_norm2 - speed_norm2)
            reward += velocity_reward

            # Prefer smaller area error and deformation than passive.
            area_reward = self.w_area * (baseline_area_error - area_error)
            deform_reward = self.w_deform * (baseline_deformation - deformation_index)
            reward += area_reward
            reward += deform_reward

            action_cost = self.w_action * float(np.mean(action * action))
            smooth_cost = self.w_smooth * float(np.mean((action - self.last_action) ** 2))
            reward -= action_cost
            reward -= smooth_cost

            if bubble_broken:
                reward -= self.w_break

        self.prev_center_y = center_y

        info = {
            "center_x": center_x,
            "center_y": center_y,
            "center_u": center_u,
            "center_v": center_v,

            "area_ratio": area_ratio_raw,
            "area_rel": area_rel,
            "area_error": area_error,
            "deformation_index": deformation_index,

            "centroid_in_target": centroid_in_target,
            "reached_target_height": reached_target_height,
            "all_extreme_particles_in_target": all_extreme_in_target,

            "deformation_break": deformation_break,
            "area_break": area_break,
            "bubble_broken": bubble_broken,

            "progress_norm": progress_norm,
            "height_progress": height_progress,
            "enter_bonus": enter_bonus,
            "entry_time_gain": entry_time_gain,

            "outside_target_penalty": outside_target_penalty,
            "hold_score": hold_score,
            "center_dist": center_dist,
            "center_score": center_score,
            "speed_score": speed_score,
            "speed_norm": speed_norm,
            "speed_norm2": speed_norm2,

            "baseline_available": baseline_available,
            "baseline_hold_score": baseline_hold_score,
            "baseline_center_dist": baseline_center_dist,
            "baseline_speed_norm": baseline_speed_norm,
            "baseline_outside_penalty": baseline_outside_penalty,
            "baseline_area_error": baseline_area_error,
            "baseline_deformation": baseline_deformation,

            "hold_reward": hold_reward,
            "outside_reward": outside_reward,
            "velocity_reward": velocity_reward,
            "area_reward": area_reward,
            "deform_reward": deform_reward,
            "action_cost": action_cost,
            "smooth_cost": smooth_cost,

            # Backward-compatible fields used by old log analysis.
            "target_center_penalty": float(np.clip(center_dist, 0.0, 2.0) / 2.0),
            "target_center_score": float(np.clip(1.0 - center_dist, 0.0, 1.0)),
            "area_penalty": -area_reward,
            "deform_penalty": -deform_reward,

            "baseline_episode": self.current_episode_is_baseline,
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
                f"seg_temps: {seg_temps}\n"
            )

        with open(reward_log, "a", encoding="utf-8") as f:
            f.write(
                f"clock: {self.sim_time:.6f} "
                f"reward: {reward:.6f} "
                f"center: ({info['center_x']:.6f}, {info['center_y']:.6f}) "
                f"vel: ({info['center_u']:.6f}, {info['center_v']:.6f}) "
                f"deformation: {info['deformation_index']:.6f} "
                f"area_ratio: {info['area_ratio']:.6f} "
                f"area_rel: {info['area_rel']:.6f} "
                f"area_error: {info['area_error']:.6f} "
                f"progress: {info['progress_norm']:.6f} "
                f"height_progress: {info['height_progress']:.6f} "
                f"deformation_break: {info.get('deformation_break', False)} "
                f"area_break: {info.get('area_break', False)} "
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
                f"total_reward: {self.total_reward_per_episode:.6f}\n"
            )

    # ------------------------------------------------------------------
    # Gym API: reset.
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_episode_is_baseline = bool(
            self.use_first_episode_as_baseline
            and self.episode == 1
            and (
                self.baseline_parallel_env is None
                or self.parallel_envs == self.baseline_parallel_env
            )
        )

        self._open_episode_logs()

        os.chdir(self.training_root)

        self.sim = self.Solver(
            self.parallel_envs,
            self.episode,
            self.reload_particles,
            self.write_output,
        )

        # Baseline left wall: four controlled segments with mean T = 1.
        self.sim.set_left_wall_segment_temperatures(
            [self.mean_temperature] * self.n_seg,
            True,
            self.mean_temperature,
        )

        self.sim_time = 0.0
        self.step_count = 0
        self.total_reward_per_episode = 0.0
        self.has_entered_target_height = False
        self.last_action = np.zeros(self.n_seg, dtype=np.float32)

        if self.current_episode_is_baseline:
            self.baseline_metrics_by_step = []
            self.baseline_entry_time = None

        self._obs_hist.clear()

        if self.warmup_time > 0.0:
            self.sim.run_case(self.warmup_time)
            self.sim_time = self.warmup_time

        # --------------------------------------------------------------
        # Reward references from initial / warm-up state.
        # --------------------------------------------------------------
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

        self.has_entered_target_height = bool(
            int(metrics0["reached_target_height"])
        )

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
            "left_wall_segment_temperatures": list(
                self.sim.get_left_wall_segment_temperatures()
            ),
            "baseline_episode": self.current_episode_is_baseline,
        }

        # Keep reset info empty for Tianshou compatibility.
        return obs0, {}

    # ------------------------------------------------------------------
    # Gym API: step.
    # ------------------------------------------------------------------
    def step(self, action):
        policy_action = self._sanitize_action(action)

        # Episode 1 is the passive no-control baseline. Ignore whatever
        # action Tianshou provides and apply zero action to the C++ solver.
        if self.current_episode_is_baseline:
            applied_action = np.zeros(self.n_seg, dtype=np.float32)
        else:
            applied_action = policy_action

        seg_temps = self._send_action_to_cpp(applied_action)

        end_time = self.sim_time + self.delta_time
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

        if self.current_episode_is_baseline:
            # Store the passive trajectory for later controlled episodes.
            self.baseline_metrics_by_step.append(dict(metrics))
            if self.baseline_entry_time is None and int(metrics["reached_target_height"]):
                self.baseline_entry_time = float(self.sim_time)

        # Observation: flow field only.
        obs = self._read_observation()

        # Reward: bubble metrics only. Use applied_action because that is the
        # actual action sent to C++.
        reward, info = self._compute_reward(applied_action, metrics)
        self.total_reward_per_episode += reward

        self._log_step(applied_action, seg_temps, reward, info)

        terminated = bool(info["bubble_broken"])

        episode_limit = (
            self.max_steps_per_episode_eval
            if self.deterministic
            else self.max_steps_per_episode
        )

        truncated = bool(self.step_count >= episode_limit)

        info.update(
            {
                "episode": self.episode,
                "step_count": self.step_count,
                "physical_time": float(self.sim.get_physical_time()),
                "segment_temperatures": seg_temps,
                "total_reward": self.total_reward_per_episode,
                "baseline_episode": self.current_episode_is_baseline,
                "policy_action": policy_action.copy(),
                "applied_action": applied_action.copy(),
                "baseline_entry_time": self.baseline_entry_time,
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
    args = parser.parse_args()

    env = BubbleRisingPositionEnv(
        parallel_envs=args.parallel_env,
        training_root=args.training_root,
        reload_particles=args.reload_particles,
        write_output=args.write_output,
        warmup_time=0.0,
        delta_time=0.02,
        max_steps_per_episode=10,
        n_probe_points=400,
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
            f"progress={info['progress_norm']:.4f}, "
            f"height_progress={info['height_progress']:.4f}, "
            f"deform={info['deformation_index']:.4f}, "
            f"area_rel={info['area_rel']:.4f}, "
            f"area_error={info['area_error']:.4f}"
        )

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    _smoke_test()