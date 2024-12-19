class GCHBCAgent(flax.struct.PyTreeNode):
    """Goal-Conditioned Hierarchical Behavioral Cloning (GCHBC) agent with enhanced subgoal extraction."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

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
            # For GCHBC, we'll use the actor's goal-conditioned predictions as a proxy for value
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'])
            log_probs = dist.log_prob(batch['actions'])
            values = jnp.exp(log_probs)  # Use action likelihood as value proxy
            return 1.0 + value_weight_scale * (values - values.mean()) / (values.std() + 1e-8)

        def compute_goal_weights(batch):
            # Simple euclidean distance in state space (could be enhanced with learned representations)
            distances = jnp.linalg.norm(
                batch['observations'] - batch['actor_goals'], 
                axis=-1
            )
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

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the hierarchical BC actor loss with enhanced subgoal weighting."""
        # Extract enhanced subgoal weights
        subgoal_weights = self.extract_subgoal_weights(
            batch['rewards'], 
            batch.get('masks', jnp.ones_like(batch['rewards'])),
            batch
        )

        # Predict action distribution conditioned on subgoals
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        # Weight log probabilities by enhanced subgoal importance
        actor_loss = -(subgoal_weights * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'log_prob_mean': log_prob.mean(),
            'subgoal_weights_mean': subgoal_weights.mean(),
            'subgoal_weights_max': subgoal_weights.max(),
            'subgoal_weights_min': subgoal_weights.min(),
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
        """Create a new GCHBC agent with enhanced subgoal extraction."""
        # [Rest of the create method remains the same as original GCHBC]

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Original GCHBC parameters
            agent_name='gcbc',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            discount=0.99,
            const_std=True,
            discrete=False,
            
            # Enhanced subgoal extraction parameters
            subgoal_steps=25,
            subgoal_decay=0.9,
            subgoal_weight_multiplier=2.0,
            value_weight_scale=0.5,
            use_value_weighting=True,
            use_goal_weighting=True,
            normalize_weights=True,
            min_weight=0.5,
            max_weight=2.0,
            
            # Original dataset parameters remain the same
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config