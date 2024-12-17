from typing import Any
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor


class GCHBCAgent(flax.struct.PyTreeNode):
    """Goal-Conditioned Hierarchical Behavioral Cloning (GCHBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def extract_hierarchical_subgoals(self, batch):
        """Decompose trajectories into hierarchical subgoals using reward change detection."""
        rewards = batch['rewards']  # Shape: (batch_size, time_steps)
        mean_reward = jnp.mean(rewards, axis=1, keepdims=True)

        # Detect significant reward changes: deviation from the mean reward
        reward_changes = jnp.abs(rewards - mean_reward)
        significant_changes = reward_changes > self.config['reward_change_threshold']

        # Identify subgoal states where significant reward changes occur
        subgoals = jnp.where(significant_changes[:, :, None], batch['next_observations'], batch['actor_goals'])

        # Weight subgoals based on the magnitude of reward changes
        weights = reward_changes / (jnp.max(reward_changes, axis=1, keepdims=True) + 1e-8)

        return subgoals, weights

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the hierarchical BC actor loss."""
        # Extract reward-conditioned hierarchical subgoals and weights
        subgoals, weights = self.extract_hierarchical_subgoals(batch)

        # Predict action distribution conditioned on subgoals
        dist = self.network.select('actor')(batch['observations'], subgoals, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        # Weight log probabilities by reward-conditioned subgoal importance
        actor_loss = -(weights * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'log_prob_mean': log_prob.mean(),
            'subgoal_weights_mean': weights.mean(),
        }
        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return actor_loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions from the actor conditioned on hierarchical subgoals."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new GCHBC agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define actor network
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=None,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=None,
            )

        network_info = dict(actor=(actor_def, (ex_observations, ex_goals)))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Configuration for the GCHBC agent."""
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gchbc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            const_std=True,  # Constant standard deviation for actor.
            discrete=False,  # Action space type.
            reward_change_threshold=0.5,  # Threshold for detecting significant reward changes.
            encoder=None,  # Encoder (optional).
            # Dataset hyperparameters.
            dataset_class='GCDataset',
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
        )
    )
    return config
