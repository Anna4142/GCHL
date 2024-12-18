from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCValue, Identity

class HGCIQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned IQL agent using GCDataset."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        # Get target Q-values
        next_v1_t, next_v2_t = self.network.select('target_value')(
            batch['next_observations'], batch['value_goals']
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        # Current value estimates
        v1_t, v2_t = self.network.select('target_value')(
            batch['observations'], batch['value_goals']
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        # Compute Q-values
        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        v1, v2 = self.network.select('value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )

        # Compute losses
        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v1.mean(),
            'v_max': v1.max(),
            'v_min': v1.min(),
        }

    def actor_loss(self, batch, grad_params):
        """Compute the IQL actor loss."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['value_goals'])
        v = (v1 + v2) / 2
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        nv = (nv1 + nv2) / 2
        adv = nv - v

        # Advantage weighted regression
        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'exp_adv': exp_a.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute total loss."""
        info = {}
        
        value_loss, value_info = self.value_loss(batch, grad_params)
        actor_loss, actor_info = self.actor_loss(batch, grad_params)

        for k, v in value_info.items():
            info[f'value/{k}'] = v
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update agent."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions."""
        dist = self.network.select('actor')(
            observations, goals, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Setup networks
        action_dim = ex_actions.shape[-1]
        ex_goals = ex_observations

        # Create encoders
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            value_encoder = GCEncoder(state_encoder=encoder_module())
            actor_encoder = GCEncoder(state_encoder=encoder_module())
        else:
            value_encoder = GCEncoder(state_encoder=Identity())
            actor_encoder = GCEncoder(state_encoder=Identity())

        # Create networks
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder
        )

        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=actor_encoder
        )

        network_info = {
            'value': (value_def, (ex_observations, ex_goals)),
            'target_value': (value_def, (ex_observations, ex_goals)),
            'actor': (actor_def, (ex_observations, ex_goals)),
        }

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize target network
        params = network.params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

    @staticmethod
    def get_config():
        return ml_collections.ConfigDict(
            dict(
                agent_name='gc_iql',
                lr=3e-4,
                batch_size=1024,
                actor_hidden_dims=(256, 256),
                value_hidden_dims=(256, 256),
                layer_norm=True,
                discount=0.99,
                tau=0.005,
                expectile=0.7,
                alpha=3.0,
                const_std=True,
                encoder=ml_collections.config_dict.placeholder(str),
                dataset_class='GCDataset',
                value_p_curgoal=0.2,
                value_p_trajgoal=0.5,
                value_p_randomgoal=0.3,
                value_geom_sample=True,
                actor_p_curgoal=0.0,
                actor_p_trajgoal=1.0,
                actor_p_randomgoal=0.0,
                actor_geom_sample=False,
                gc_negative=True,
                p_aug=0.0,
                frame_stack=ml_collections.config_dict.placeholder(int),
            )
        )