"""Unit-test DepthCNNGRUModel on real Stairs-Student observations (shapes + hidden-state reset).

Usage:
    python scripts/tools/test_student_cnn.py --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Smoke tests the student CNNGRU model e2e.")
parser.add_argument("--task", type=str, default="Stairs-Student-Deeprobotics-M20-v0")
parser.add_argument("--num_envs", type=int, default=16)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

import rl_training.tasks  # noqa: F401  (registers the gym envs)
from rl_training.models import DepthCNNGRUModel


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain.max_init_terrain_level = 9  # spread envs over difficulty rows
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    base = env.unwrapped

    obs = env.get_observations()
    print("[INFO] observation groups:", {k: tuple(v.shape) for k, v in obs.items()})

    n, a = args_cli.num_envs, env.num_actions
    model = DepthCNNGRUModel(
        obs,
        {"student": ["policy", "depth"]},
        "student",
        a,
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
        cnn_cfg={"output_channels": [16, 32, 32], "kernel_size": [5, 4, 3], "stride": [2, 2, 1]},
    ).to(base.device)
    print(model)
    print(f"[INFO] cnn latent {model.cnn_latent_dim}, proprio {model.obs_dim}, gru input {model.rnn.rnn.input_size}")

    with torch.no_grad():
        out = model(obs)
        assert out.shape == (n, a), out.shape
        h = model.get_hidden_state()
        assert h.shape == (1, n, 256), h.shape
        assert h.abs().sum() > 0

        dones = torch.zeros(n, device=base.device)
        dones[0] = 1
        model.reset(dones)
        h = model.get_hidden_state()
        assert torch.all(h[:, 0] == 0), "env 0 hidden state not cleared"
        assert torch.all(h[:, 1:].abs().sum(-1) > 0), "other envs' hidden state was cleared"

        out2 = model(obs)
        assert out2.shape == (n, a), out2.shape
        assert torch.isfinite(out2).all()
        assert model.get_hidden_state()[:, 0].abs().sum() > 0

    print(f"[INFO] output {tuple(out.shape)}, hidden {tuple(h.shape)}")
    print("[PASS]")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
