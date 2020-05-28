from tf_agents.agents import SacAgent
from tf_agents.policies import actor_policy
import tensorflow as tf


class CustomSacAgent(SacAgent):
    def __init__(self, time_step_spec, action_spec, critic_network, actor_network, actor_optimizer, critic_optimizer,
                 alpha_optimizer, actor_loss_weight=1.0, critic_loss_weight=0.5, alpha_loss_weight=1.0,
                 actor_policy_ctor=actor_policy.ActorPolicy, critic_network_2=None, target_critic_network=None,
                 target_critic_network_2=None, target_update_tau=1.0, target_update_period=1,
                 td_errors_loss_fn=tf.math.squared_difference, gamma=1.0, reward_scale_factor=1.0,
                 initial_log_alpha=0.0, target_entropy=None, gradient_clipping=None, debug_summaries=False,
                 summarize_grads_and_vars=False, train_step_counter=None, name=None):
        super().__init__(time_step_spec, action_spec, critic_network, actor_network, actor_optimizer, critic_optimizer,
                         alpha_optimizer, actor_loss_weight, critic_loss_weight, alpha_loss_weight, actor_policy_ctor,
                         critic_network_2, target_critic_network, target_critic_network_2, target_update_tau,
                         target_update_period, td_errors_loss_fn, gamma, reward_scale_factor, initial_log_alpha,
                         target_entropy, gradient_clipping, debug_summaries, summarize_grads_and_vars,
                         train_step_counter, name)

    def get_actor_network(self):
        return self._actor_network

    def get_critic_network_1(self):
        return self._critic_network_1

    def get_critic_network_2(self):
        return self._critic_network_2
