import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue


class GCHIQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit Q-learning (GCIQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def weighted_mean(self, values, weights=None):
        """Compute weighted mean of values. If weights is None, do a simple mean."""
        if weights is None:
            return values.mean()
        w_sum = weights.sum() + 1e-8
        return (values * weights).sum() / w_sum

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss with subgoal weighting."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)

        weights = batch.get('subgoal_weights', None)
        losses = self.expectile_loss(q - v, q - v, self.config['expectile'])
        value_loss = self.weighted_mean(losses, weights)

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': self.weighted_mean(v, weights),
            'v_max': (v * (weights > 0) if weights is not None else v).max(),
            'v_min': (v * (weights > 0) if weights is not None else v).min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss with subgoal weighting."""
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
        weights = batch.get('subgoal_weights', None)

        losses = (q1 - q)**2 + (q2 - q)**2
        critic_loss = self.weighted_mean(losses, weights)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': self.weighted_mean(q, weights),
            'q_max': (q * (weights > 0) if weights is not None else q).max(),
            'q_min': (q * (weights > 0) if weights is not None else q).min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC) with subgoal weighting."""
        weights = batch.get('subgoal_weights', None)

        if self.config['actor_loss'] == 'awr':
            # AWR loss
            v = self.network.select('value')(batch['observations'], batch['actor_goals'])
            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            # Weighted AWR loss
            losses = -exp_a * log_prob
            actor_loss = self.weighted_mean(losses, weights)

            actor_info = {
                'actor_loss': actor_loss,
                'adv': self.weighted_mean(adv, weights),
                'bc_log_prob': self.weighted_mean(log_prob, weights),
            }
            if not self.config['discrete']:
                mse = (dist.mode() - batch['actions'])**2
                actor_info.update(
                    {
                        'mse': self.weighted_mean(mse, weights),
                        'std': jnp.mean(dist.scale_diag),  # Std doesn't depend on weights.
                    }
                )

            return actor_loss, actor_info

        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)

            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            q1, q2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
            q = jnp.minimum(q1, q2)

            q_mean = self.weighted_mean(q, weights)
            q_abs_mean = self.weighted_mean(jnp.abs(q), weights)

            q_loss = -q_mean / jax.lax.stop_gradient(q_abs_mean + 1e-6)
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -self.weighted_mean(self.config['alpha'] * log_prob, weights)

            actor_loss = q_loss + bc_loss

            actor_info = {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q_mean,
                'q_abs_mean': q_abs_mean,
                'bc_log_prob': self.weighted_mean(log_prob, weights),
                'mse': self.weighted_mean((dist.mode() - batch['actions'])**2, weights),
                'std': jnp.mean(dist.scale_diag),
            }
            return actor_loss, actor_info

        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss with subgoal weighting."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def extract_subgoal_weights(self, rewards: jnp.ndarray, masks: jnp.ndarray, batch=None) -> jnp.ndarray:
        """
        Enhanced subgoal weight extraction with temporal decay, value-based weighting, and goal-based weighting.
        """
        # Initialize base weights
        subgoal_weights = jnp.ones_like(rewards)

        # Get hyperparameters from config
        decay_factor = self.config.get('subgoal_decay', 0.9)
        window_size = self.config.get('subgoal_steps', 25)
        base_multiplier = self.config.get('subgoal_weight_multiplier', 2.0)
        value_weight_scale = self.config.get('value_weight_scale', 0.5)
        use_value_weighting = self.config.get('use_value_weighting', True)
        use_goal_weighting = self.config.get('use_goal_weighting', True)
        normalize_weights = self.config.get('normalize_weights', True)
        min_weight = self.config.get('min_weight', 0.5)
        max_weight = self.config.get('max_weight', 2.0)

        def compute_temporal_weights(rewards, masks):
            # Identify reward events
            reward_indices = jnp.where((rewards > 0) & (masks > 0))[0]

            def apply_decay_window(carry, idx):
                base_weights = carry
                window_start = jnp.maximum(0, idx - window_size)
                steps = jnp.arange(idx - window_start)
                decay = decay_factor ** (idx - window_start - steps - 1)
                indices = jnp.arange(window_start, idx)
                updates = base_multiplier * decay
                base_weights = base_weights.at[indices].multiply(updates)
                return base_weights, None

            temporal_weights, _ = jax.lax.scan(
                apply_decay_window,
                subgoal_weights,
                reward_indices,
            )
            return temporal_weights

        weights = compute_temporal_weights(rewards, masks)

        def compute_value_weights(batch):
            v1, v2 = self.network.select('value')(batch['observations'], batch['value_goals'])
            values = (v1 + v2) / 2
            return 1.0 + value_weight_scale * (values - values.mean()) / (values.std() + 1e-8)

        def compute_goal_weights(batch):
            state_reps = self.network.select('goal_rep')(batch['observations'])
            goal_reps = self.network.select('goal_rep')(batch['value_goals'])
            distances = jnp.linalg.norm(state_reps - goal_reps, axis=-1)
            norm_distances = distances / (distances.max() + 1e-8)
            return 1.0 + (1.0 - norm_distances)

        if batch is not None:
            try:
                if use_value_weighting:
                    value_weights = compute_value_weights(batch)
                    weights = weights * value_weights
                if use_goal_weighting:
                    goal_weights = compute_goal_weights(batch)
                    weights = weights * goal_weights
            except:
                pass

        weights = weights * masks
        weights = jnp.clip(weights, min_weight, max_weight)

        if normalize_weights:
            weight_mean = weights.mean()
            weights = jnp.where(weight_mean > 0, weights / weight_mean, weights)

        return weights

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)
        subgoal_weights = self.extract_subgoal_weights(batch['rewards'], batch['masks'], batch)
        batch = {**batch, 'subgoal_weights': subgoal_weights}

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = int(ex_actions.max()) + 1
        else:
            action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='gciql',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            actor_loss='ddpgbc',  # or 'awr'
            alpha=0.3,
            const_std=True,
            discrete=False,
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
            frame_stack=4,

            # Subgoal weighting parameters
            subgoal_steps=25,
            subgoal_decay=0.9,
            subgoal_weight_multiplier=2.0,
            value_weight_scale=0.5,
            use_value_weighting=True,
            use_goal_weighting=True,
            normalize_weights=True,
            min_weight=0.5,
            max_weight=2.0,
        )
    )
    return config