import copy
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.navigation.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import (
    DeeproboticsM20FlatEnvCfg,
)

LOW_LEVEL_ENV_CFG = DeeproboticsM20FlatEnvCfg()
LOW_LEVEL_POLICY_PATH = (
    "/workspace/rl_training/upstream_base/logs/rsl_rl/deeprobotics_m20_rough/2026-11-18_23-40-28/model_9999.pt"
)

def _ll_obs():
    obs = copy.deepcopy(LOW_LEVEL_ENV_CFG.observations.policy)   # deepcopy: the action term mutates it
    obs.enable_corruption = False
    return obs

@configclass
class ActionsCfg:
    velocity_command = mdp.M20PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
        low_level_actions=[LOW_LEVEL_ENV_CFG.actions.joint_pos, LOW_LEVEL_ENV_CFG.actions.joint_vel],
        low_level_observations=_ll_obs(),
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        last_cmd = ObsTerm(func=mdp.last_action)
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)   # placeholder until Step 3

@configclass
class CommandsCfg:
    pass                                             # Step 2: waypoints go here

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = LOW_LEVEL_ENV_CFG.terminations.illegal_contact
    bad_orientation_2 = LOW_LEVEL_ENV_CFG.terminations.bad_orientation_2

@configclass
class M20NavFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene = LOW_LEVEL_ENV_CFG.scene
    events = LOW_LEVEL_ENV_CFG.events
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim = LOW_LEVEL_ENV_CFG.sim                 # same dt (0.005), physics material
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 5   # high level at 10 Hz
        self.episode_length_s = 20.0
        self.events.randomize_push_robot = None          # no pushes for the smoke test
        self.events.randomize_apply_external_force_torque = None
