from __future__ import annotations

from rsl_rl.models import CNNModel
from rsl_rl.modules import RNN


class DepthCNNGRUModel(CNNModel):
    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        cnn_cfg=None,
        cnns=None,
        rnn_type="gru",
        rnn_num_layers=1,
        rnn_hidden_dim=256,
    ):  
        self.rnn_hidden_dim = rnn_hidden_dim
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            cnn_cfg,
            cnns,
        )

        # Create GRU from RSL-Rl's premade RNN module
        self.rnn = RNN(
            input_size=self.obs_dim + self.cnn_latent_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            type=rnn_type,
        )

    def _get_latent_dim(self):
        return self.rnn_hidden_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        x = super().get_latent(obs=obs)
        return self.rnn(x, masks, hidden_state).squeeze(0)

    def reset(self, dones=None, hidden_state=None):
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self):
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones=None):
        self.rnn.detach_hidden_state(dones)
