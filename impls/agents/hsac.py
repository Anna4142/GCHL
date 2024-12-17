
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, LogParam
class HSACAgent(flax.struct.PyTreeNode):
    """Hierarchical Soft Actor-Critic (H-SAC) agent with HCF."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the hierarchical critic loss."""
        # High-Level: Generate subgoals
        high_level_dist = self.network.select('high_actor')(batch['observations'])
        high_actions, _ = high_level_dist.sample_and_log_prob(seed=rng)

        # Target Q for subgoals
        next_high_qs = self.network.select('target_high_critic')(batch['next_observations'], high_actions)
        next_high_q = jnp.min(next_high_qs, axis=0)

        target_high_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_high_q

        # Low-Level: Use subgoals to compute low-level Q-values
        low_level_dist = self.network.select('low_actor')(batch['observations'], high_actions)
        low_actions, _ = low_level_dist.sample_and_log_prob(seed=rng)
        low_qs = self.network.select('critic')(batch['observations'], low_actions)
        low_q = jnp.min(low_qs, axis=0)

        # Critic loss combines both high-level and low-level losses
        critic_loss = (jnp.square(low_q - target_high_q)).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'low_q_mean': low_q.mean(),
            'high_q_mean': target_high_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute hierarchical actor loss."""
        # High-level actor: Generates subgoals
        high_dist = self.network.select('high_actor')(batch['observations'], params=grad_params)
        subgoals, log_prob_high = high_dist.sample_and_log_prob(seed=rng)

        # Low-level actor: Optimized for primitive actions
        low_dist = self.network.select('low_actor')(batch['observations'], subgoals, params=grad_params)
        actions, log_prob_low = low_dist.sample_and_log_prob(seed=rng)

        # Q-values for subgoals and actions
        low_qs = self.network.select('critic')(batch['observations'], actions)
        low_q = jnp.min(low_qs, axis=0)

        # Actor loss
        high_actor_loss = log_prob_high.mean() - low_q.mean()
        low_actor_loss = log_prob_low.mean() - low_q.mean()

        total_actor_loss = high_actor_loss + low_actor_loss

        return total_actor_loss, {
            'high_actor_loss': high_actor_loss,
            'low_actor_loss': low_actor_loss,
            'total_actor_loss': total_actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total hierarchical SAC loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # Critic loss
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Actor loss
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the hierarchical SAC agent."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a hierarchical SAC agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        subgoal_dim = action_dim  # Subgoal space is same as action space

        # High-Level Networks
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=subgoal_dim,
            state_dependent_std=True,
        )
        high_critic_def = GCValue(hidden_dims=config['value_hidden_dims'], ensemble=True)

        # Low-Level Networks
        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=True,
        )
        critic_def = GCValue(hidden_dims=config['value_hidden_dims'], ensemble=True)

        network_info = dict(
            high_actor=(high_actor_def, (ex_observations,)),
            target_high_critic=(copy.deepcopy(high_critic_def), (ex_observations, None)),
            low_actor=(low_actor_def, (ex_observations, None)),
            critic=(critic_def, (ex_observations, None, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='hsac',
            lr=1e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256),
            value_hidden_dims=(256, 256),
            discount=0.99,
            tau=0.005,
            tanh_squash=True,
            state_dependent_std=True,
        )
    )
    return config
