#!/usr/bin/env python3
"""
Final open-loop comparison script for this folder layout:

velocity_control/
    bin/drl/drl_tianshou_training/
        sac_single.py
        open_loop_compare_best_fixed.py   <- put this file here
    training_process/
        bubble_rising_fast_y15_bottom5/
            log/BR-v0/sac/0/<timestamp>/policy.pth

What this script does:
    1. Automatically finds the latest policy.pth unless --policy-path is given.
    2. Loads the trained SAC policy.
    3. Gets the policy action at the initial state.
    4. Runs two open-loop cases only:
         - zero_baseline: [0, 0, 0, 0, 0]
         - best_fixed_first: fixed first action from best policy
    5. Writes open_loop_fixed_action_summary.csv.

No mean-action, no median-action, no action-sequence replay.
This script is deliberately fixed to the simple diagnostic requested.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Batch
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


# -----------------------------------------------------------------------------
# Path helpers for the shown project structure.
# -----------------------------------------------------------------------------
def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def velocity_root_from_script() -> str:
    # current file is expected at:
    # velocity_control/bin/drl/drl_tianshou_training/open_loop_compare_best_fixed.py
    return os.path.abspath(os.path.join(script_dir(), "..", "..", ".."))


def default_training_script() -> str:
    return os.path.join(script_dir(), "sac_single.py")


def default_case_training_root() -> str:
    return os.path.join(
        velocity_root_from_script(),
        "training_process",
        "bubble_rising_fast_y15_bottom5",
    )


def default_output_root() -> str:
    return os.path.join(
        velocity_root_from_script(),
        "training_process",
        "open_loop_best_fixed_compare",
    )


def find_latest_policy(training_root: str, task: str, seed: int) -> Optional[str]:
    """Find latest policy.pth under the actual log folder used by this case."""
    candidates: list[str] = []

    search_roots = [
        os.path.join(training_root, "log", task, "sac", str(seed)),
        os.path.join(training_root, "log"),
        os.path.join(script_dir(), "log", task, "sac", str(seed)),
        os.path.join(script_dir(), "log"),
    ]

    for root in search_roots:
        candidates.extend(glob.glob(os.path.join(root, "**", "policy.pth"), recursive=True))

    candidates = [os.path.abspath(p) for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def import_training_script(path: str):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"training script not found: {path}")

    module_dir = os.path.dirname(path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("_br_sac_single_for_open_loop", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import training script: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# SAC policy builder. Must match sac_single.py architecture.
# -----------------------------------------------------------------------------
def build_sac_policy(obs_shape, action_shape, max_action: float, args) -> SACPolicy:
    net_a = Net(
        obs_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )

    actor = ActorProb(
        net_a,
        action_shape,
        device=args.device,
        max_action=max_action,
        conditioned_sigma=True,
    ).to(args.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(
        obs_shape,
        action_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    net_c2 = Net(
        obs_shape,
        action_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        concat=True,
        device=args.device,
    )
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    alpha = float(args.alpha)
    if args.auto_alpha:
        prod = np.prod(action_shape)
        target_entropy = -float(prod) if np.isscalar(prod) else -float(prod.item())
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        estimation_step=args.n_step,
        action_space=args.action_space,
    )

    return policy


def policy_action(policy: SACPolicy, obs: np.ndarray, action_space, device: str) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    batch = Batch(obs=obs, info={})

    with torch.no_grad():
        out = policy(batch)

    act = out.act
    if isinstance(act, torch.Tensor):
        act = act.detach().cpu().numpy()

    try:
        act = policy.map_action(act)
    except Exception:
        pass

    if isinstance(act, torch.Tensor):
        act = act.detach().cpu().numpy()

    act = np.asarray(act, dtype=np.float32)
    if act.ndim >= 2:
        act = act[0]

    return np.clip(act, action_space.low, action_space.high).astype(np.float32)


# -----------------------------------------------------------------------------
# Open-loop rollout.
# -----------------------------------------------------------------------------
@dataclass
class RolloutSummary:
    case: str
    parallel_envs: int
    fixed_action: list[float]
    first_segment_temperatures: list[float] | None
    first_reach_time: float | None
    steps: int
    final_time: float
    final_center_x: float
    final_center_y: float
    max_center_y: float
    max_deformation_index: float
    max_area_error: float
    total_reward: float
    output_dir: str


def make_env(training_mod, run_args, parallel_envs: int, output_root: str, write_output: bool):
    return training_mod.make_br_env(
        parallel_envs=parallel_envs,
        training_root=output_root,
        args=run_args,
        write_output_override=write_output,
        terminate_on_success=True,
        baseline_metrics_csv=None,
        baseline_entry_time=None,
    )


def run_reset_env_fixed_action(
    env,
    run_args,
    output_root: str,
    case_name: str,
    parallel_envs: int,
    fixed_action: np.ndarray,
) -> RolloutSummary:
    first_segment_temperatures = None
    first_reach_time = None
    final_info = {}
    total_reward = 0.0
    max_center_y = -1.0e30
    max_deformation_index = -1.0e30
    max_area_error = -1.0e30
    steps = 0

    for _ in range(int(run_args.max_steps_per_episode)):
        _obs, reward, terminated, truncated, info = env.step(fixed_action)
        steps += 1
        total_reward += float(reward)
        final_info = info

        if first_segment_temperatures is None:
            first_segment_temperatures = [float(x) for x in info.get("segment_temperatures", [])]

        center_y = float(info.get("center_y", np.nan))
        deformation_index = float(info.get("deformation_index", np.nan))
        area_error = float(info.get("area_error", np.nan))

        max_center_y = max(max_center_y, center_y)
        max_deformation_index = max(max_deformation_index, deformation_index)
        max_area_error = max(max_area_error, area_error)

        if first_reach_time is None and center_y >= float(run_args.target_height):
            first_reach_time = float(info.get("physical_time", np.nan))

        if terminated or truncated:
            break

    output_dir = os.path.join(os.path.abspath(output_root), f"output_env_{parallel_envs}_episode_1")

    return RolloutSummary(
        case=case_name,
        parallel_envs=parallel_envs,
        fixed_action=[float(x) for x in fixed_action.tolist()],
        first_segment_temperatures=first_segment_temperatures,
        first_reach_time=first_reach_time,
        steps=steps,
        final_time=float(final_info.get("physical_time", np.nan)),
        final_center_x=float(final_info.get("center_x", np.nan)),
        final_center_y=float(final_info.get("center_y", np.nan)),
        max_center_y=float(max_center_y),
        max_deformation_index=float(max_deformation_index),
        max_area_error=float(max_area_error),
        total_reward=float(total_reward),
        output_dir=output_dir,
    )


def run_fixed_action_case(
    training_mod,
    run_args,
    output_root: str,
    case_name: str,
    parallel_envs: int,
    fixed_action: np.ndarray,
    write_output: bool,
) -> RolloutSummary:
    env = make_env(training_mod, run_args, parallel_envs, output_root, write_output)
    env.reset()
    summary = run_reset_env_fixed_action(
        env=env,
        run_args=run_args,
        output_root=output_root,
        case_name=case_name,
        parallel_envs=parallel_envs,
        fixed_action=fixed_action,
    )
    env.close()
    return summary


def write_summary_csv(path: str, rows: list[RolloutSummary]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "case",
        "parallel_envs",
        "fixed_action",
        "first_segment_temperatures",
        "first_reach_time",
        "steps",
        "final_time",
        "final_center_x",
        "final_center_y",
        "max_center_y",
        "max_deformation_index",
        "max_area_error",
        "total_reward",
        "output_dir",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "case": r.case,
                "parallel_envs": r.parallel_envs,
                "fixed_action": r.fixed_action,
                "first_segment_temperatures": r.first_segment_temperatures,
                "first_reach_time": r.first_reach_time,
                "steps": r.steps,
                "final_time": r.final_time,
                "final_center_x": r.final_center_x,
                "final_center_y": r.final_center_y,
                "max_center_y": r.max_center_y,
                "max_deformation_index": r.max_deformation_index,
                "max_area_error": r.max_area_error,
                "total_reward": r.total_reward,
                "output_dir": r.output_dir,
            })


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-script", type=str, default=default_training_script())
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--case-training-root", type=str, default=default_case_training_root())
    parser.add_argument("--output-root", type=str, default=default_output_root())
    parser.add_argument("--clear-output", action="store_true")

    parser.add_argument("--task", type=str, default="BR-v0")
    parser.add_argument("--n-seg", type=int, default=5)
    parser.add_argument("--n-probe-points", type=int, default=400)
    parser.add_argument("--reload-particles", action="store_true")
    parser.add_argument("--write-output", action="store_true", default=True)
    parser.add_argument("--no-write-output", dest="write_output", action="store_false")

    parser.add_argument("--warmup-time", type=float, default=0.0)
    parser.add_argument("--delta-time", type=float, default=0.02)
    parser.add_argument("--episode-max-time", type=float, default=4.5)
    parser.add_argument("--max-steps-per-episode", type=int, default=225)
    parser.add_argument("--target-height", type=float, default=1.5)
    parser.add_argument("--action-amplitude", type=float, default=0.7)
    parser.add_argument("--mean-temperature", type=float, default=1.0)

    parser.add_argument("--zero-parallel-env", type=int, default=0)
    parser.add_argument("--fixed-parallel-env", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])
    parser.add_argument("--actor-lr", type=float, default=1.0e-4)
    parser.add_argument("--critic-lr", type=float, default=1.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--auto-alpha", action="store_true", default=True)
    parser.add_argument("--no-auto-alpha", dest="auto_alpha", action="store_false")
    parser.add_argument("--alpha-lr", type=float, default=5.0e-4)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    training_script = os.path.abspath(args.training_script)
    case_training_root = os.path.abspath(args.case_training_root)
    output_root = os.path.abspath(args.output_root)

    if args.policy_path is None:
        args.policy_path = find_latest_policy(case_training_root, args.task, args.seed)
        if args.policy_path is None:
            raise FileNotFoundError(
                "Could not automatically find policy.pth.\n"
                f"Searched under:\n"
                f"  {os.path.join(case_training_root, 'log')}\n"
                f"  {os.path.join(script_dir(), 'log')}\n"
                "Please pass --policy-path explicitly."
            )
        print(f"[Auto] Using latest policy: {args.policy_path}")
    else:
        args.policy_path = os.path.abspath(args.policy_path)

    if not os.path.isfile(args.policy_path):
        raise FileNotFoundError(f"policy file not found: {args.policy_path}")

    if args.clear_output and os.path.isdir(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    print(f"[Path] training_script   = {training_script}")
    print(f"[Path] policy_path       = {args.policy_path}")
    print(f"[Path] output_root       = {output_root}")
    print(f"[Task] target_height     = {args.target_height}")
    print(f"[Task] action_amplitude  = {args.action_amplitude}")
    print(f"[Task] mean_temperature  = {args.mean_temperature}")

    training_mod = import_training_script(training_script)

    # Args object expected by sac_single.make_br_env.
    run_args = SimpleNamespace(
        task=args.task,
        n_seg=args.n_seg,
        n_probe_points=args.n_probe_points,
        reload_particles=args.reload_particles,
        write_output=args.write_output,
        warmup_time=args.warmup_time,
        delta_time=args.delta_time,
        episode_max_time=args.episode_max_time,
        max_steps_per_episode=args.max_steps_per_episode,
        target_height=args.target_height,
        action_amplitude=args.action_amplitude,
        mean_temperature=args.mean_temperature,
        baseline_metrics_csv=None,
        baseline_entry_time=None,
    )

    # Create zero-baseline env first. Its initial observation is also used
    # to extract the first best-policy action. This keeps the output to only
    # two physical cases: output_env_0 and output_env_1.
    zero_env = make_env(
        training_mod=training_mod,
        run_args=run_args,
        parallel_envs=args.zero_parallel_env,
        output_root=output_root,
        write_output=args.write_output,
    )

    obs0, _ = zero_env.reset()

    args.action_space = zero_env.action_space
    state_shape = zero_env.observation_space.shape or zero_env.observation_space.n
    action_shape = zero_env.action_space.shape or zero_env.action_space.n
    max_action = float(zero_env.action_space.high[0])

    policy = build_sac_policy(state_shape, action_shape, max_action, args)
    policy.load_state_dict(torch.load(args.policy_path, map_location=args.device))
    policy.eval()

    best_first_action = policy_action(policy, obs0, zero_env.action_space, args.device)
    zero_action = np.zeros((args.n_seg,), dtype=np.float32)

    with open(os.path.join(output_root, "fixed_action.txt"), "w", encoding="utf-8") as f:
        f.write(f"policy_path: {args.policy_path}\n")
        f.write(f"fixed_action_source: first_policy_action_at_reset\n")
        f.write(f"zero_action: {zero_action.tolist()}\n")
        f.write(f"best_fixed_first_action: {best_first_action.tolist()}\n")

    print("[Action] zero_baseline:", zero_action.tolist())
    print("[Action] best_fixed_first:", best_first_action.tolist())

    rows: list[RolloutSummary] = []

    rows.append(run_reset_env_fixed_action(
        env=zero_env,
        run_args=run_args,
        output_root=output_root,
        case_name="zero_baseline",
        parallel_envs=args.zero_parallel_env,
        fixed_action=zero_action,
    ))
    zero_env.close()

    rows.append(run_fixed_action_case(
        training_mod=training_mod,
        run_args=run_args,
        output_root=output_root,
        case_name="best_fixed_first",
        parallel_envs=args.fixed_parallel_env,
        fixed_action=best_first_action,
        write_output=args.write_output,
    ))

    summary_csv = os.path.join(output_root, "open_loop_fixed_action_summary.csv")
    write_summary_csv(summary_csv, rows)

    print("\n[Done] Summary:")
    for r in rows:
        print(
            f"  {r.case}: first_reach_time={r.first_reach_time}, "
            f"steps={r.steps}, final_center_y={r.final_center_y:.6f}, "
            f"max_center_y={r.max_center_y:.6f}"
        )

    z = rows[0].first_reach_time
    b = rows[1].first_reach_time
    if z is not None and b is not None:
        print(f"[Compare] zero - best_fixed_first = {z - b:.6f} s")
    else:
        print("[Compare] Cannot compute time improvement because one case did not reach target.")

    print(f"[Done] CSV = {summary_csv}")
    print(f"[Done] fixed_action.txt = {os.path.join(output_root, 'fixed_action.txt')}")


if __name__ == "__main__":
    main()
