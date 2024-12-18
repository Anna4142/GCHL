import copy
from typing import Any, List, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, LogParam


class SACAgent(flax.struct.PyTreeNode):
    """Soft actor-critic (SAC) agent with hierarchical mechanism based on rewards."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng, subgoal_weights):
        """Compute the SAC critic loss with weighted subgoal integration."""
        next_dist = self.network.select('actor')(batch['next_observations'], params=grad_params)
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=rng)

        next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
        if self.config['min_q']:
            next_q = jnp.min(next_qs, axis=0)
        else:
            next_q = jnp.mean(next_qs, axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        target_q = target_q - self.config['discount'] * batch['masks'] * next_log_probs * self.network.select('alpha')()

        q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q)

        # Apply subgoal weights
        weighted_critic_loss = subgoal_weights * critic_loss
        weighted_critic_loss = weighted_critic_loss.mean()

        return weighted_critic_loss, {
            'critic_loss': weighted_critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng, subgoal_weights):
        """Compute the SAC actor loss with weighted subgoal integration."""
        # Actor loss.
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        qs = self.network.select('critic')(batch['observations'], actions)
        if self.config['min_q']:
            q = jnp.min(qs, axis=0)
        else:
            q = jnp.mean(qs, axis=0)

        actor_loss = (log_probs * self.network.select('alpha')() - q)
        # Apply subgoal weights
        weighted_actor_loss = subgoal_weights * actor_loss
        weighted_actor_loss = weighted_actor_loss.mean()

        # Entropy loss.
        alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

        total_loss = weighted_actor_loss + alpha_loss

        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': weighted_actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'std': action_std.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, subgoal_weights=None):
        """Compute the total loss with weighted subgoal integration."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # Pass subgoal_weights to loss functions
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, subgoal_weights)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, subgoal_weights)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        # Extract subgoal weights based on rewards
        subgoal_weights = self.extract_subgoal_weights(batch['rewards'], batch['masks'])

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, subgoal_weights=subgoal_weights)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Define critic and actor networks.
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            log_std_min=-5,
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )

        # Define the dual alpha variable.
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, None, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, None, ex_actions)),
            actor=(actor_def, (ex_observations, None)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

    def extract_subgoal_weights(self, rewards: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
            """
            Enhanced subgoal weight extraction with decaying influence and multi-step weighting.
            
            Args:
                rewards: Array of rewards for each timestep
                masks: Array indicating non-terminal (1) or terminal (0) transitions
                
            Returns:
                subgoal_weights: Array of weights for each transition
            """
            # Base weights
            subgoal_weights = jnp.ones_like(rewards)
            
            # Get indices of reward events
            reward_indices = jnp.where(rewards > 0)[0]
            
            # Parameters for weighting
            decay_factor = 0.9  # How quickly influence decays
            window_size = 5     # How many previous steps to weight
            
            def weight_window(idx, weights):
                # Apply decaying weights to window before reward
                window_start = jnp.maximum(0, idx - window_size)
                steps = jnp.arange(window_size)
                decay = decay_factor ** (window_size - steps - 1)
                
                for i, w in enumerate(decay):
                    curr_idx = idx - i - 1
                    weights = jax.lax.cond(
                        curr_idx >= window_start,
                        lambda: weights.at[curr_idx].multiply(
                            w * self.config.subgoal_weight_multiplier
                        ),
                        lambda: weights
                    )
                return weights

            # Apply weighting for each reward event
            subgoal_weights = jax.vmap(weight_window)(reward_indices, subgoal_weights)

            # Additional weighting based on value predictions
            if hasattr(self, 'network'):
                value_predictions = self.network.select('value')(
                    batch['observations'], batch['value_goals']
                )[0]
                # Increase weights for transitions leading to high-value states
                value_weight_factor = jnp.clip(value_predictions / value_predictions.mean(), 0.5, 2.0)
                subgoal_weights = subgoal_weights * value_weight_factor

            return subgoal_weights

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='sac',  # Agent name.
            lr=1e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(256, 256),  # Actor network hidden dimensions.
            value_hidden_dims=(256, 256),  # Value network hidden dimensions.
            layer_norm=False,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            min_q=True,  # Whether to use min Q (True) or mean Q (False).
            # Hierarchical Mechanism Parameters
            use_hierarchical=True,  # Enable hierarchical mechanism.
            subgoal_weighting=True,  # Enable subgoal weighting.
            tau_subgoal=0.005,  # Subgoal update rate if applicable.
            subgoal_weight_multiplier=2.0,  # Multiplier for subgoal weights
        )
    )
    return config