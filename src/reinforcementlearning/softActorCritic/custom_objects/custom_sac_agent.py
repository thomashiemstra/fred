import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents import SacAgent
from tf_agents.agents.sac.sac_agent import SacLossInfo
from tf_agents.policies import actor_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import numpy as np
from tf_agents.utils import nest_utils


class CustomSacAgent(SacAgent):
    def __init__(self, time_step_spec, action_spec, critic_network, actor_network, actor_optimizer, critic_optimizer,
                 alpha_optimizer, actor_loss_weight=1.0, critic_loss_weight=0.5, alpha_loss_weight=1.0,
                 actor_policy_ctor=actor_policy.ActorPolicy, critic_network_2=None, target_critic_network=None,
                 target_critic_network_2=None, target_update_tau=1.0, target_update_period=1,
                 td_errors_loss_fn=tf.math.squared_difference, gamma=1.0, reward_scale_factor=1.0,
                 initial_log_alpha=0.0, use_log_alpha_in_alpha_loss=True, target_entropy=None, gradient_clipping=None,
                 debug_summaries=False, summarize_grads_and_vars=False, train_step_counter=None, name=None, bc_errors_loss_fn=tf.math.squared_difference):
        # loss function for behavioral cloning
        self.bc_errors_loss_fn = bc_errors_loss_fn
        super().__init__(time_step_spec, action_spec, critic_network, actor_network, actor_optimizer, critic_optimizer,
                         alpha_optimizer, actor_loss_weight, critic_loss_weight, alpha_loss_weight, actor_policy_ctor,
                         critic_network_2, target_critic_network, target_critic_network_2, target_update_tau,
                         target_update_period, td_errors_loss_fn, gamma, reward_scale_factor, initial_log_alpha,
                         use_log_alpha_in_alpha_loss, target_entropy, gradient_clipping, debug_summaries,
                         summarize_grads_and_vars, train_step_counter, name)

    def get_actor_network(self):
        return self._actor_network

    def actor_loss_behavioral_cloning(self,
                                      time_steps,
                                      actions,
                                      bc_errors_loss_fn,
                                      training=False,
                                      weights=None):
        means, stds, network_state = self._actor_network.call_raw(time_steps.observation,
                                                                    time_steps.step_type,
                                                                    (),
                                                                    training=training)

        means_loss = bc_errors_loss_fn(means, actions)

        expected_stds = tf.fill(tf.shape(actions), 0.5)
        stds_loss = bc_errors_loss_fn(stds, expected_stds)

        total_loss = means_loss + stds_loss

        if nest_utils.is_batched_nested_tensors(
                time_steps, self.time_step_spec, num_outer_dims=2):
            # Sum over the time dimension.
            total_loss = tf.reduce_sum(input_tensor=total_loss, axis=1)
        reg_loss = self._actor_network.losses if self._actor_network else None
        agg_loss = common.aggregate_losses(
            per_example_loss=total_loss,
            sample_weight=weights,
            regularization_loss=reg_loss)
        total_loss = agg_loss.total_loss

        return total_loss

    def train_behavioral_cloning(self, experience, weights=None):
        """Returns a train op to update the agent's networks.

        This method trains with the provided batched experience.

        Args:
          experience: A time-stacked trajectory object.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.

        Returns:
          A train_op.

        Raises:
          ValueError: If optimizers are None and no default value was provided to
            the constructor.
        """
        squeeze_time_dim = not self._critic_network_1.state_spec
        time_steps, policy_steps, next_time_steps = (
            trajectory.experience_to_transitions(experience, squeeze_time_dim))
        actions = policy_steps.action

        trainable_critic_variables = (
                self._critic_network_1.trainable_variables +
                self._critic_network_2.trainable_variables)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_critic_variables, ('No trainable critic variables to '
                                                'optimize.')
            tape.watch(trainable_critic_variables)
            critic_loss = self._critic_loss_weight * self.critic_loss(
                time_steps,
                actions,
                next_time_steps,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)

        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
        self._apply_gradients(critic_grads, trainable_critic_variables,
                              self._critic_optimizer)

        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, ('No trainable actor variables to '
                                               'optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss = self._actor_loss_weight * self.actor_loss_behavioral_cloning(
                time_steps, actions, self.bc_errors_loss_fn, training=True)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
        self._apply_gradients(actor_grads, trainable_actor_variables,
                              self._actor_optimizer)

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='critic_loss', data=critic_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)

        self.train_step_counter.assign_add(1)
        self._update_target()

        total_loss = critic_loss + actor_loss

        return tf_agent.LossInfo(loss=total_loss, extra=None), actor_loss, critic_loss
