#!/usr/bin/env python3
import os
import sys
import argparse
import datetime
import pprint
import shutil
import stat
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# Add local Gym environment package.
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../bin/drl/drl_tianshou_training
DRL_DIR = os.path.dirname(CURRENT_DIR)                    # .../bin/drl
GYM_ROOT = os.path.join(DRL_DIR, "drl_gym_environments")  # .../bin/drl/drl_gym_environments

if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)

# This package should register BR-v0.
import gym_env_br

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, Batch
from tianshou.policy import SACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.env import SubprocVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


# =============================================================================
# File utilities.
# =============================================================================
def _rmtree_onerror(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass

    try:
        func(path)
    except Exception:
        pass


def rebuild_training_root(training_root: str, clear: bool = True) -> str:
    """
    Rebuild training root.

    Directory layout:
        training_root/
            input/
            output/
            reload/
            restart/
            logs_env_*/
    """
    training_root = os.path.abspath(training_root)

    base = os.path.basename(training_root.rstrip("\\/"))
    if base != "bubble_rising_position_single":
        raise RuntimeError(
            f"Refuse to clear unexpected directory:\n"
            f"  {training_root}\n"
            f"Expected basename == 'bubble_rising_position_single'."
        )

    if clear and os.path.isdir(training_root):
        for _ in range(5):
            try:
                shutil.rmtree(training_root, onerror=_rmtree_onerror)
                break
            except Exception:
                time.sleep(0.2)

        if os.path.isdir(training_root):
            shutil.rmtree(training_root, onerror=_rmtree_onerror)

    os.makedirs(training_root, exist_ok=True)

    for name in ("input", "output", "reload", "restart"):
        os.makedirs(os.path.join(training_root, name), exist_ok=True)

    return training_root


def prepare_case_root(case_root: str, clear: bool = True) -> str:
    case_root = os.path.abspath(case_root)

    if clear and os.path.isdir(case_root):
        shutil.rmtree(case_root, onerror=_rmtree_onerror)

    os.makedirs(case_root, exist_ok=True)

    for name in ("input", "output", "reload", "restart"):
        os.makedirs(os.path.join(case_root, name), exist_ok=True)

    return case_root


# =============================================================================
# Arguments.
# =============================================================================
def get_args():
    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------------------
    # Bubble rising environment.
    # -------------------------------------------------------------------------
    parser.add_argument("--task", type=str, default="BR-v0")
    parser.add_argument("--n-seg", type=int, default=4)

    # Current environment uses flow observation only:
    # obs = [u0, v0, T0, u1, v1, T1, ...]
    parser.add_argument("--obs-mode", type=str, default="flow", choices=["flow"])
    parser.add_argument(
        "--n-probe-points",
        type=int,
        default=400,
        help="Must match C++ createObservationPoints(). 20x20 -> 400.",
    )

    parser.add_argument("--reload-particles", default=False, action="store_true")
    parser.add_argument("--write-output", default=False, action="store_true")

    parser.add_argument("--warmup-time", type=float, default=0.0)
    parser.add_argument("--delta-time", type=float, default=0.02)
    parser.add_argument("--max-steps-per-episode", type=int, default=250)

    parser.add_argument("--action-amplitude", type=float, default=0.3)
    parser.add_argument("--mean-temperature", type=float, default=1.0)

    # -------------------------------------------------------------------------
    # SAC parameters.
    # -------------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)

    parser.add_argument("--episodes-per-epoch", type=int, default=5)
    parser.add_argument("--buffer-size", type=int, default=int(1e5))
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])

    parser.add_argument("--actor-lr", type=float, default=1.0e-4)
    parser.add_argument("--critic-lr", type=float, default=1.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--auto-alpha", default=True, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=5.0e-4)

    parser.add_argument("--start-episodes", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--update-per-step", type=float, default=1.0)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--watch", default=False, action="store_true")

    # -------------------------------------------------------------------------
    # Generate output after training using the best policy.
    # -------------------------------------------------------------------------
    parser.add_argument("--generate-best-output", default=True, action="store_true")
    parser.add_argument(
        "--no-generate-best-output",
        dest="generate_best_output",
        action="store_false",
    )
    parser.add_argument("--best-output-steps", type=int, default=250)
    parser.add_argument("--best-output-parallel-env", type=int, default=888)

    # -------------------------------------------------------------------------
    # Training root behavior.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--no-clear-training-root",
        dest="clear_training_root",
        action="store_false",
    )
    parser.set_defaults(clear_training_root=True)

    return parser.parse_args()


# =============================================================================
# Environment builders.
# =============================================================================
def make_br_env(
    parallel_envs: int,
    training_root: str,
    args,
    write_output_override: Optional[bool] = None,
):
    use_write_output = args.write_output if write_output_override is None else bool(write_output_override)

    return gym.make(
        args.task,
        parallel_envs=parallel_envs,
        n_seg=args.n_seg,
        training_root=training_root,
        reload_particles=args.reload_particles,
        write_output=use_write_output,
        warmup_time=args.warmup_time,
        delta_time=args.delta_time,
        max_steps_per_episode=args.max_steps_per_episode,
        n_probe_points=args.n_probe_points,
        action_amplitude=args.action_amplitude,
        mean_temperature=args.mean_temperature,
    )


def make_br_env_fn(
    parallel_envs: int,
    training_root: str,
    args,
    write_output_override: Optional[bool] = None,
):
    def _thunk():
        return make_br_env(
            parallel_envs=parallel_envs,
            training_root=training_root,
            args=args,
            write_output_override=write_output_override,
        )

    return _thunk


# =============================================================================
# Best-policy rollout.
# =============================================================================
def _policy_action(policy, obs: np.ndarray, action_space, device: str) -> np.ndarray:
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

    act = np.clip(act, action_space.low, action_space.high)

    return act.astype(np.float32)


def generate_output_from_best_policy(
    policy,
    args,
    training_root: str,
    log_path: str,
):
    """
    After training:
      1. Load log_path/policy.pth.
      2. Create a new environment with write_output=True.
      3. Roll out the best policy.
      4. Save VTP/output files.
    """
    policy_path = os.path.join(log_path, "policy.pth")

    if not os.path.isfile(policy_path):
        print(f"[BestOutput] No best policy found at: {policy_path}")
        print("[BestOutput] Save current policy as fallback.")
        torch.save(policy.state_dict(), policy_path)

    print(f"[BestOutput] Loading best policy from: {policy_path}")
    policy.load_state_dict(torch.load(policy_path, map_location=args.device))
    policy.eval()

    best_output_root = os.path.join(training_root, "best_policy_output")
    best_output_root = prepare_case_root(best_output_root, clear=True)

    print(f"[BestOutput] Output root: {best_output_root}")

    env = make_br_env(
        parallel_envs=args.best_output_parallel_env,
        training_root=best_output_root,
        args=args,
        write_output_override=True,
    )

    obs, _ = env.reset()

    total_reward = 0.0
    final_info = {}
    last_action = None

    n_steps = int(args.best_output_steps)
    if n_steps <= 0:
        n_steps = int(args.max_steps_per_episode)

    for i in range(n_steps):
        action = _policy_action(policy, obs, env.action_space, args.device)
        last_action = action.copy()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        final_info = info

        if terminated or truncated:
            break

    env.close()

    summary_path = os.path.join(best_output_root, "best_policy_rollout_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"policy_path: {policy_path}\n")
        f.write(f"steps: {i + 1}\n")
        f.write(f"total_reward: {total_reward:.9f}\n")
        f.write(f"last_action: {None if last_action is None else last_action.tolist()}\n")
        f.write(f"final_info: {final_info}\n")

    print("[BestOutput] Done.")
    print(f"[BestOutput] steps = {i + 1}")
    print(f"[BestOutput] total_reward = {total_reward:.6f}")
    print(f"[BestOutput] summary = {summary_path}")


# =============================================================================
# SAC training.
# =============================================================================
def training_sac(args=None):
    if args is None:
        args = get_args()

    proj_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "../training_process")
    )

    training_root = os.path.join(proj_root, "bubble_rising_position_single")
    training_root = rebuild_training_root(
        training_root,
        clear=args.clear_training_root,
    )

    print(f"[BR] training_root: {training_root}")
    print(f"[BR] task: {args.task}")
    print(f"[BR] n_seg: {args.n_seg}")
    print(f"[BR] obs_mode: {args.obs_mode}")
    print(f"[BR] n_probe_points: {args.n_probe_points}")
    print(f"[BR] reload_particles: {args.reload_particles}")
    print(f"[BR] write_output during training: False")
    print(f"[BR] generate_best_output: {args.generate_best_output}")

    # -------------------------------------------------------------------------
    # Create environments.
    # Training output is forced off. Best-policy output is generated after training.
    # -------------------------------------------------------------------------
    if args.training_num > 1:
        train_env = SubprocVectorEnv(
            [
                make_br_env_fn(
                    parallel_envs=i,
                    training_root=training_root,
                    args=args,
                    write_output_override=False,
                )
                for i in range(args.training_num)
            ]
        )
    else:
        train_env = make_br_env(
            parallel_envs=0,
            training_root=training_root,
            args=args,
            write_output_override=False,
        )

    if args.test_num > 1:
        test_env = SubprocVectorEnv(
            [
                make_br_env_fn(
                    parallel_envs=900 + i,
                    training_root=training_root,
                    args=args,
                    write_output_override=False,
                )
                for i in range(args.test_num)
            ]
        )
    else:
        test_env = make_br_env(
            parallel_envs=999,
            training_root=training_root,
            args=args,
            write_output_override=False,
        )

    # -------------------------------------------------------------------------
    # Spaces.
    # -------------------------------------------------------------------------
    args.state_shape = train_env.observation_space.shape or train_env.observation_space.n
    args.action_shape = train_env.action_space.shape or train_env.action_space.n
    args.max_action = train_env.action_space.high[0]

    print(f"[BR] state_shape: {args.state_shape}")
    print(f"[BR] action_shape: {args.action_shape}")
    print(f"[BR] max_action: {args.max_action}")

    # -------------------------------------------------------------------------
    # Seed.
    # Do not manually reset env here; Tianshou Collector will reset environments.
    # -------------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # Actor.
    # -------------------------------------------------------------------------
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )

    actor = ActorProb(
        net_a,
        args.action_shape,
        device=args.device,
        max_action=args.max_action,
        conditioned_sigma=True,
    ).to(args.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # -------------------------------------------------------------------------
    # Critic 1.
    # -------------------------------------------------------------------------
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        concat=True,
        device=args.device,
    )

    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    # -------------------------------------------------------------------------
    # Critic 2.
    # -------------------------------------------------------------------------
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        concat=True,
        device=args.device,
    )

    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # -------------------------------------------------------------------------
    # SAC alpha.
    # -------------------------------------------------------------------------
    alpha = float(args.alpha)

    if args.auto_alpha:
        prod = np.prod(train_env.action_space.shape)
        target_entropy = -float(prod) if np.isscalar(prod) else -float(prod.item())

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

        alpha = (target_entropy, log_alpha, alpha_optim)

    # -------------------------------------------------------------------------
    # Policy.
    # -------------------------------------------------------------------------
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
        action_space=train_env.action_space,
    )

    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print(f"[BR] Loaded agent from: {args.resume_path}")

    # -------------------------------------------------------------------------
    # Watch mode.
    # -------------------------------------------------------------------------
    if args.watch:
        policy.eval()

        test_collector = Collector(policy, test_env)
        test_collector.reset()

        result = test_collector.collect(n_episode=args.test_num)

        print(f"Final reward: {result['rews'].mean()}, length: {result['lens'].mean()}")

        if args.generate_best_output and args.resume_path:
            now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            log_path = os.path.join(
                args.logdir,
                args.task,
                "sac",
                str(args.seed),
                f"watch-{now}",
            )
            os.makedirs(log_path, exist_ok=True)

            shutil.copy(args.resume_path, os.path.join(log_path, "policy.pth"))

            generate_output_from_best_policy(
                policy=policy,
                args=args,
                training_root=training_root,
                log_path=log_path,
            )

        return

    # -------------------------------------------------------------------------
    # Replay buffer and collectors.
    # -------------------------------------------------------------------------
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    else:
        buffer = ReplayBuffer(args.buffer_size)

    train_collector = Collector(
        policy,
        train_env,
        buffer,
        exploration_noise=False,
    )

    test_collector = Collector(
        policy,
        test_env,
        exploration_noise=False,
    )

    if args.start_episodes > 0:
        print(f"[BR] Collecting {args.start_episodes} random startup episodes...")
        train_collector.collect(n_episode=args.start_episodes, random=True)

    # -------------------------------------------------------------------------
    # Logging.
    # -------------------------------------------------------------------------
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    log_path = os.path.join(
        args.logdir,
        args.task,
        "sac",
        str(args.seed),
        now,
    )

    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))

    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        policy_path = os.path.join(log_path, "policy.pth")
        torch.save(policy.state_dict(), policy_path)
        print(f"[BestPolicy] Saved best policy to: {policy_path}")

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(
            log_path,
            f"checkpoint_epoch{epoch}_envstep{env_step}_gradstep{gradient_step}.pth",
        )

        torch.save(
            {
                "model": policy.state_dict(),
                "optim_actor": actor_optim.state_dict(),
                "optim_critic1": critic1_optim.state_dict(),
                "optim_critic2": critic2_optim.state_dict(),
            },
            ckpt_path,
        )

        return ckpt_path

    # -------------------------------------------------------------------------
    # Schedule.
    # -------------------------------------------------------------------------
    max_steps = getattr(
        train_env.unwrapped if hasattr(train_env, "unwrapped") else train_env,
        "max_steps_per_episode",
        args.max_steps_per_episode,
    )

    steps_per_episode = int(max_steps)
    step_per_collect = steps_per_episode * int(args.episodes_per_epoch)
    step_per_epoch = step_per_collect

    print(f"[BR] steps_per_episode: {steps_per_episode}")
    print(f"[BR] episodes_per_epoch: {args.episodes_per_epoch}")
    print(f"[BR] step_per_collect: {step_per_collect}")
    print(f"[BR] step_per_epoch: {step_per_epoch}")

    # -------------------------------------------------------------------------
    # Train.
    # -------------------------------------------------------------------------
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        episode_per_test=args.test_num,
        max_epoch=args.epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        batch_size=args.batch_size,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        update_per_step=args.update_per_step,
    ).run()

    pprint.pprint(result)

    # -------------------------------------------------------------------------
    # Generate VTP/output after training using best policy.
    # -------------------------------------------------------------------------
    if args.generate_best_output:
        generate_output_from_best_policy(
            policy=policy,
            args=args,
            training_root=training_root,
            log_path=log_path,
        )


if __name__ == "__main__":
    training_sac()