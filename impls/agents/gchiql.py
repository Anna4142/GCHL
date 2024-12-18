from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize


class GCHIQLAgent(flax.struct.PyTreeNode):
    """Hierarchical Implicit Q-Learning (HIQL) agent using GCDataset."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params, subgoal_weights):
        """Compute the IQL value loss with weighted subgoal integration."""
        # Compute target Q-values using target critic
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        # Current value estimates
        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        # Compute current value estimates
        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        # Expectile loss
        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        # Apply subgoal weights
        weighted_value_loss = subgoal_weights * value_loss
        weighted_value_loss = weighted_value_loss.mean()

        return weighted_value_loss, {
            'value_loss': weighted_value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params, subgoal_weights):
        """Compute the low-level actor loss with weighted integration."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob)

        # Apply subgoal weights
        weighted_actor_loss = subgoal_weights * actor_loss
        weighted_actor_loss = weighted_actor_loss.mean()

        actor_info = {
            'actor_loss': weighted_actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return weighted_actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params, subgoal_weights):
        """Compute the high-level actor loss with weighted integration."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1),
            params=grad_params,
        )
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob)

        # Apply subgoal weights
        weighted_actor_loss = subgoal_weights * actor_loss
        weighted_actor_loss = weighted_actor_loss.mean()

        return weighted_actor_loss, {
            'actor_loss': weighted_actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, subgoal_weights=None):
        """Compute the total loss with weighted subgoal integration."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params, subgoal_weights)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, subgoal_weights)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, subgoal_weights)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss
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
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
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
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations  # Assuming goals are part of observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        # Define the encoders that handle the inputs to the value and actor networks.
        if config['encoder'] is not None:
            # Pixel-based environments require visual encoders for state inputs, in addition to the pre-defined shared
            # encoder for subgoal representations.

            # Value: V(encoder^V(s), phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | encoder^h([s; g]))
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            # State-based environments only use the pre-defined shared encoder for subgoal representations.

            # Value: V(s, phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | s, g) (i.e., no encoder)
            high_actor_encoder_def = None

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
            )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize target value parameters
        params = network.params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
    def extract_subgoal_weights(self, rewards: jnp.ndarray, masks: jnp.ndarray, batch=None) -> jnp.ndarray:
            """
            Enhanced subgoal weight extraction with temporal decay and value-based weighting.
            
            Args:
                rewards: Array of rewards for each timestep
                masks: Array indicating non-terminal (1) or terminal (0) transitions
                batch: Optional batch data for value and goal-based weighting
            
            Returns:
                subgoal_weights: Array of weights for each transition
            """
            # Initialize base weights
            subgoal_weights = jnp.ones_like(rewards)

            # Get hyperparameters from config
            decay_factor = self.config.get('subgoal_decay', 0.9)
            window_size = self.config.get('subgoal_steps', 25)  
            base_multiplier = self.config.get('subgoal_weight_multiplier', 2.0)
            value_weight_scale = self.config.get('value_weight_scale', 0.5)
            
            # 1. Temporal Weighting
            def compute_temporal_weights(rewards, masks):
                # Find reward events
                reward_indices = jnp.where((rewards > 0) & (masks > 0))[0]
                
                def apply_decay_window(idx, base_weights):
                    # Calculate decay for window before reward
                    window_start = jnp.maximum(0, idx - window_size)
                    steps = jnp.arange(window_size)
                    decay = decay_factor ** (window_size - steps - 1)
                    
                    def update_weight(i, weights):
                        curr_idx = idx - i - 1
                        return jax.lax.cond(
                            curr_idx >= window_start,
                            lambda w: w.at[curr_idx].multiply(base_multiplier * decay[i]),
                            lambda w: w,
                            weights
                        )
                    
                    return jax.lax.fori_loop(0, window_size, update_weight, base_weights)
                
                # Apply temporal weighting for each reward
                temporal_weights = jax.lax.foldl(
                    apply_decay_window,
                    reward_indices,
                    subgoal_weights
                )
                return temporal_weights
            
            # 2. Value-Based Weighting
            def compute_value_weights(batch):
                # Get value predictions
                v1, v2 = self.network.select('value')(batch['observations'], batch['value_goals'])
                values = (v1 + v2) / 2
                
                # Normalize to [0.5, 1.5] range for stable weighting
                value_weights = 1.0 + value_weight_scale * (values - values.mean()) / (values.std() + 1e-8)
                return value_weights
            
            # 3. Goal-Based Importance
            def compute_goal_weights(batch):
                # Get state and goal representations
                state_reps = self.network.select('goal_rep')(batch['observations'])
                goal_reps = self.network.select('goal_rep')(batch['value_goals'])
                
                # Compute normalized distance to goals
                distances = jnp.linalg.norm(state_reps - goal_reps, axis=-1)
                norm_distances = distances / (distances.max() + 1e-8)
                
                # Convert to weights (closer to goal = higher weight)
                goal_weights = 1.0 + (1.0 - norm_distances)
                return goal_weights
            
            # Combine weightings
            weights = compute_temporal_weights(rewards, masks)
            
            if batch is not None:
                try:
                    # Add value-based weighting
                    value_weights = compute_value_weights(batch)
                    weights = weights * value_weights
                    
                    # Add goal-based weighting
                    goal_weights = compute_goal_weights(batch)
                    weights = weights * goal_weights
                except:
                    pass  # Fall back to temporal weights if value/goal weighting fails
            
            # Normalize weights
            weights = weights * masks  # Zero out terminal states
            weight_mean = weights.mean()
            weights = jnp.where(
                weight_mean > 0,
                weights / weight_mean,
                weights
            )
            
            return weights
def get_config():
        config = ml_collections.ConfigDict(
            dict(
                # Agent hyperparameters.
                agent_name='hiql',  # Agent name.
                lr=3e-4,  # Learning rate.
                batch_size=1024,  # Batch size.
                actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
                value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
                layer_norm=True,  # Whether to use layer normalization.
                discount=0.99,  # Discount factor.
                tau=0.005,  # Target network update rate.
                expectile=0.7,  # IQL expectile.
                low_alpha=3.0,  # Low-level AWR temperature.
                high_alpha=3.0,  # High-level AWR temperature.
                subgoal_steps=25,  # Subgoal steps.
                rep_dim=10,  # Goal representation dimension.
                low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
                const_std=True,  # Whether to use constant standard deviation for the actors.
                discrete=False,  # Whether the action space is discrete.
                encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
                # Dataset hyperparameters.
                dataset_class='GCDataset',  # Changed from 'HGCDataset' to 'GCDataset'.
                value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
                value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
                value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
                value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
                actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
                actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
                actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
                actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
                gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
                p_aug=0.0,  # Probability of applying image augmentation.
                frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
                
                use_hierarchical=True,              # Enable hierarchical mechanism
                subgoal_weighting=True,            # Enable subgoal weighting
                subgoal_weight_multiplier=2.0,     # Base multiplier for subgoal weights
                subgoal_decay=0.9,                 # Temporal decay factor
                subgoal_steps=25,                  # Window size for temporal weighting
                value_weight_scale=0.5,            # Scale factor for value-based weighting
                use_value_weighting=True,          # Enable value-based weighting
                use_goal_weighting=True,           # Enable goal-based weighting
                normalize_weights=True,            # Whether to normalize weights
                min_weight=0.5,                    # Minimum weight value
                max_weight=2.0,                    # Maximum weight value
            )
        )
        return config

  
    