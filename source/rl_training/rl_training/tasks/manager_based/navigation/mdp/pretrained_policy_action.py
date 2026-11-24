from __future__ import annotations
from dataclasses import MISSING
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg, ObservationGroupCfg
from isaaclab.managers.observation_manager import ObservationManager
from isaaclab.utils.configclass import configclass
import torch.nn as nn
import torch

__all__ = ["M20PreTrainedPolicyAction", "M20PreTrainedPolicyActionCfg"]


class M20PreTrainedPolicyAction(ActionTerm):
    cfg: "M20PreTrainedPolicyActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.policy = _load_actor(cfg.policy_path, cfg.hidden_dims, self.device)
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        lim = torch.tensor(cfg.command_limits, device=self.device)
        self._cmd_lo, self._cmd_hi = lim[:, 0], lim[:, 1]

        # Actions should always be the same, so we build the low level action terms.
        # [joint_pos cfg, joint_vel cfg]
        self._ll_terms = [c.class_type(c, env) for c in cfg.low_level_actions]
        self._ll_dims = [t.action_dim for t in self._ll_terms]
        # Holds the low-level policy's latest raw output
        self.low_level_actions = torch.zeros(self.num_envs, sum(self._ll_dims), device=self.device)

        # Builds the low level observations 
        obs_cfg = cfg.low_level_observations
        obs_cfg.velocity_commands.func = lambda dummy_env: self.processed_actions
        obs_cfg.velocity_commands.params = {}

        # Builds actions space
        obs_cfg.actions.func = lambda dummy_env: self.low_level_actions
        obs_cfg.actions.params = {}

        # Set up observations manager for ll policy, handles all observations for ll policy.
        self._ll_obs_manager = ObservationManager({"ll_policy": obs_cfg}, env)

        # Make sure the dimensions are correct for the ll policy
        obs_dim = self._ll_obs_manager.group_obs_dim["ll_policy"][0]
        assert obs_dim == self.policy[0].in_features, f"low-level obs {obs_dim} != policy input {self.policy[0].in_features}"
        assert sum(self._ll_dims) == self.policy[-1].out_features
        self._counter = 0

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, self._cmd_lo, self._cmd_hi)

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            obs = self._ll_obs_manager.compute_group("ll_policy")
            with torch.no_grad():                       # NOT inference_mode (taints sim buffers on reset)
                self.low_level_actions[:] = self.policy(obs)
            for term, a in zip(self._ll_terms, torch.split(self.low_level_actions, self._ll_dims, dim=1)):
                term.process_actions(a)
            self._counter = 0
        for term in self._ll_terms:
            term.apply_actions()
        self._counter += 1

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.low_level_actions[env_ids] = 0.0
        for term in self._ll_terms:
            term.reset(env_ids)


def _load_actor(path:str, hidden_dims, device) -> nn.Sequential:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = {k.removeprefix("mlp."): v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")}
    dims = [sd["0.weight"].shape[1], *hidden_dims]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    layers.append(nn.Linear(dims[-1], sd[f"{2 * len(hidden_dims)}.weight"].shape[0]))
    net = nn.Sequential(*layers)
    net.load_state_dict(sd)
    return net.to(device).eval().requires_grad_(False)


@configclass
class M20PreTrainedPolicyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = M20PreTrainedPolicyAction
    asset_name: str = MISSING
    policy_path: str = MISSING                          # rsl_rl checkpoint (model_XXXX.pt)
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    low_level_decimation: int = 4                       # physics steps per low-level step (50 Hz)
    low_level_actions: list[ActionTermCfg] = MISSING    # ORDER MATTERS: [joint_pos (12 legs), joint_vel (4 wheels)]
    low_level_observations: ObservationGroupCfg = MISSING
    # clamp high-level commands (vx, vy, wz) to the low level's training ranges;
    command_limits: tuple[tuple[float, float], ...] = ((-2.0, 2.0), (-1.0, 1.0), (-2.0, 2.0))