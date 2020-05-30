import tensorflow as tf
from tf_agents.agents import SacAgent
from tf_agents.policies import actor_policy


class CustomSacAgent(SacAgent):
    def __init__(self, time_step_spec, action_spec, critic_network, actor_network, actor_optimizer, critic_optimizer,
                 alpha_optimizer, actor_loss_weight=1.0, critic_loss_weight=0.5, alpha_loss_weight=1.0,
                 actor_policy_ctor=actor_policy.ActorPolicy, critic_network_2=None, target_critic_network=None,
                 target_critic_network_2=None, target_update_tau=1.0, target_update_period=1,
                 td_errors_loss_fn=tf.math.squared_difference, gamma=1.0, reward_scale_factor=1.0,
                 initial_log_alpha=0.0, target_entropy=None, gradient_clipping=None, debug_summaries=False,
                 summarize_grads_and_vars=False, train_step_counter=None, name=None):
        self.default_alpha_loss_weight = alpha_loss_weight

        # We don't want to train alpha here (turn down the temperature),
        # otherwise the agent won't do any random actions for exploration after training
        self.train_behavioral_cloning = True
        alpha_loss_weight = 0.0

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

    def actor_loss(self, time_steps, weights=None):
        if not self.train_behavioral_cloning:
            return super().actor_loss(time_steps, weights)
    #     TODO (good luck....)

    def train_behavioral_cloning(self):
        self.train_behavioral_cloning = True
        self._alpha_loss_weight = 0.0

    def train_normal_sac(self):
        self.train_behavioral_cloning = False
        self._alpha_loss_weight = self.default_alpha_loss_weight

