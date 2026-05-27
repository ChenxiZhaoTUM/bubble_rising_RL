#!/usr/bin/env python3
import os
import sys
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

    Bubble metrics are NOT part of the agent observation.
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
        obs_mode: str = "flow",
        n_probe_points: int = 400,
        action_amplitude: float = 0.3,
        mean_temperature: float = 1.0,
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

        # Force flow-only observation.
        self.obs_mode = "flow"

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
        # Bounded reward weights.
        # Per-step reward is kept O(1), avoiding -1000 episode returns.
        # ------------------------------------------------------------------
        self.w_alive = 0.05

        # Before reaching target height.
        self.w_rise_progress = 0.6
        self.w_height_progress = 0.4
        self.w_time = 0.005
        self.w_enter_bonus = 2.0

        # After reaching target height.
        self.w_inside_target = 1.0
        self.w_whole_inside_target = 0.5
        self.w_outside_distance = 0.5
        self.w_velocity_hold = 0.10

        # Bubble integrity.
        self.w_area = 1.0
        self.w_deform = 1.5
        self.w_break = 5.0

        # Control regularization.
        self.w_action = 0.02
        self.w_smooth = 0.05

        # Soft / hard limits.
        self.deformation_soft = 0.55
        self.deformation_hard = 0.90

        # Area is judged relative to reset-time area ratio.
        self.area_soft = 0.25
        self.area_hard = 0.65

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

    @staticmethod
    def _soft_excess(value: float, soft: float, hard: float) -> float:
        """
        Returns 0 below soft limit and 1 above hard limit.
        """
        if hard <= soft:
            return float(value > hard)

        return float(np.clip((value - soft) / (hard - soft), 0.0, 1.0))

    def _target_distance_penalty(self, center_x: float, center_y: float) -> float:
        """
        Bounded target-distance penalty in [0, 1].
        """
        half_w = 0.5 * (self.target_x_max - self.target_x_min)
        half_h = 0.5 * (self.target_y_max - self.target_y_min)

        dx = (center_x - self.target_x_center) / (half_w + 1.0e-12)
        dy = (center_y - self.target_y_center) / (half_h + 1.0e-12)

        dist = float(np.sqrt(dx * dx + dy * dy))

        # 0 at target center, roughly 1 near target boundary.
        return float(np.clip(dist, 0.0, 2.0) / 2.0)

    # ------------------------------------------------------------------
    # Reward: computed only from bubble metrics.
    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        metrics = self.sim.get_bubble_metrics_dict()

        center_x = float(metrics["center_x"])
        center_y = float(metrics["center_y"])
        center_u = float(metrics["center_u"])
        center_v = float(metrics["center_v"])

        area_ratio_raw = float(metrics["area_ratio"])
        deformation_index = float(metrics["deformation_index"])

        centroid_in_target = int(metrics["centroid_in_target"])
        reached_target_height = int(metrics["reached_target_height"])
        all_extreme_in_target = int(metrics["all_extreme_particles_in_target"])

        # ------------------------------------------------------------------
        # Relative area error based on reset value.
        # This avoids large reward offsets if C++ area_ratio is not exactly 1.0.
        # ------------------------------------------------------------------
        area_rel = area_ratio_raw / (self.ref_area_ratio + 1.0e-12)
        area_error = abs(area_rel - 1.0)

        deformation_excess = self._soft_excess(
            deformation_index,
            self.deformation_soft,
            self.deformation_hard,
        )

        area_excess = self._soft_excess(
            area_error,
            self.area_soft,
            self.area_hard,
        )

        bubble_broken = bool(
            deformation_index >= self.deformation_hard
            or area_error >= self.area_hard
        )

        # ------------------------------------------------------------------
        # Reward components.
        # ------------------------------------------------------------------
        reward = 0.0
        reward += self.w_alive

        enter_bonus = 0.0

        if reached_target_height and not self.has_entered_target_height:
            self.has_entered_target_height = True
            enter_bonus = self.w_enter_bonus

        # Vertical progress normalized by expected velocity scale.
        dy_step = center_y - self.prev_center_y
        progress_norm = float(
            np.clip(
                dy_step / (self.U_f * self.delta_time + 1.0e-12),
                -1.0,
                1.0,
            )
        )

        # Progress from reset height to target lower boundary.
        height_denom = self.target_y_min - self.ref_center_y
        if abs(height_denom) < 1.0e-12:
            height_progress = 1.0
        else:
            height_progress = self._ramp01(
                (center_y - self.ref_center_y) / height_denom
            )

        if not self.has_entered_target_height:
            # Stage 1: move upward quickly to y = DH / 3.
            reward += self.w_rise_progress * progress_norm
            reward += self.w_height_progress * height_progress
            reward -= self.w_time

        else:
            # Stage 2: stay inside target region.
            if centroid_in_target:
                reward += self.w_inside_target
            else:
                dist_penalty = self._target_distance_penalty(center_x, center_y)
                reward -= self.w_outside_distance * dist_penalty

            if all_extreme_in_target:
                reward += self.w_whole_inside_target

            # Once target height is reached, slow motion is preferred.
            speed_norm2 = (
                center_u * center_u + center_v * center_v
            ) / (self.U_f * self.U_f + 1.0e-12)

            reward -= self.w_velocity_hold * float(np.clip(speed_norm2, 0.0, 4.0))

        reward += enter_bonus

        # ------------------------------------------------------------------
        # Bubble integrity penalties, bounded.
        # ------------------------------------------------------------------
        area_penalty = self.w_area * area_excess
        deform_penalty = self.w_deform * deformation_excess

        reward -= area_penalty
        reward -= deform_penalty

        if bubble_broken:
            reward -= self.w_break

        # ------------------------------------------------------------------
        # Control cost.
        # ------------------------------------------------------------------
        action_cost = self.w_action * float(np.mean(action * action))
        smooth_cost = self.w_smooth * float(np.mean((action - self.last_action) ** 2))

        reward -= action_cost
        reward -= smooth_cost

        # Update previous position after using it.
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
            "bubble_broken": bubble_broken,

            "progress_norm": progress_norm,
            "height_progress": height_progress,
            "enter_bonus": enter_bonus,

            "area_penalty": area_penalty,
            "deform_penalty": deform_penalty,
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

        self._obs_hist.clear()

        if self.warmup_time > 0.0:
            self.sim.run_case(self.warmup_time)
            self.sim_time = self.warmup_time

        # --------------------------------------------------------------
        # Reward references from initial / warm-up state.
        # --------------------------------------------------------------
        metrics0 = self.sim.get_bubble_metrics_dict()

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
            "physical_time": float(self.sim.get_physical_time()),
            "metrics": metrics0,
            "ref_center_y": self.ref_center_y,
            "ref_area_ratio": self.ref_area_ratio,
            "left_wall_segment_temperatures": list(
                self.sim.get_left_wall_segment_temperatures()
            ),
        }

        # Keep reset info empty for Tianshou compatibility.
        return obs0, {}

    # ------------------------------------------------------------------
    # Gym API: step.
    # ------------------------------------------------------------------
    def step(self, action):
        action = self._sanitize_action(action)

        seg_temps = self._send_action_to_cpp(action)

        end_time = self.sim_time + self.delta_time
        self.sim.run_case(end_time)

        self.step_count += 1
        self.sim_time = end_time

        # Observation: flow field only.
        obs = self._read_observation()

        # Reward: bubble metrics only.
        reward, info = self._compute_reward(action)
        self.total_reward_per_episode += reward

        self._log_step(action, seg_temps, reward, info)

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
            }
        )

        if terminated or truncated:
            self._log_episode_end()
            self.episode += 1

        self.last_action = action.copy()

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
        obs_mode="flow",
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