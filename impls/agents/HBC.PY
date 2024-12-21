from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, Identity, LengthNormalize


class HBCAgent(flax.struct.PyTreeNode):
    """Hierarchical Behavioral Cloning agent using HGCDataset."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def low_actor_loss(self, batch, grad_params):
        """Compute low-level actor loss."""
        # Get goal representation
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params
        )

        # Get action distribution
        dist = self.network.select('low_actor')(
            batch['observations'], goal_reps, goal_encoded=True, params=grad_params
        )
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -log_prob.mean()

        info = {
            'actor_loss': actor_loss,
            'log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            info.update({
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            })

        return actor_loss, info

    def high_actor_loss(self, batch, grad_params):
        """Compute high-level actor loss."""
        # Get goal distribution
        dist = self.network.select('high_actor')(
            batch['observations'], batch['high_actor_goals'], params=grad_params
        )

        # Get target goal representation
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1),
            params=grad_params
        )

        log_prob = dist.log_prob(target)
        actor_loss = -log_prob.mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute total loss."""
        info = {}
        
        low_loss, low_info = self.low_actor_loss(batch, grad_params)
        high_loss, high_info = self.high_actor_loss(batch, grad_params)

        for k, v in low_info.items():
            info[f'low_actor/{k}'] = v
        for k, v in high_info.items():
            info[f'high_actor/{k}'] = v

        loss = low_loss + high_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update agent."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions hierarchically."""
        high_seed, low_seed = jax.random.split(seed)

        # Sample subgoal
        high_dist = self.network.select('high_actor')(
            observations, goals, temperature=temperature
        )
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        # Sample action
        low_dist = self.network.select('low_actor')(
            observations, goal_reps, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Create goal representation network
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

        # Define encoders for value and actor networks
        if config['encoder'] is not None:
            # For pixel-based environments
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            # For state-based environments
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None

        # Define actor networks
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
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
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
                # Agent hyperparameters
                agent_name='hbc',
                lr=3e-4,
                batch_size=1024,
                actor_hidden_dims=(512, 512, 512),
                value_hidden_dims=(512, 512, 512),
                layer_norm=True,
                discrete=False,
                const_std=True,
                rep_dim=10,
                encoder=ml_collections.config_dict.placeholder(str),
                discount=0.99,
                subgoal_steps=25,

                

           
                dataset_class='HGCDataset',  # Dataset class name.
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
            )
        
   
        )
        return config