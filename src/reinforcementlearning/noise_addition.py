from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.sac import sac_agent
import tensorflow as tf
from tf_agents.utils.common import soft_variables_update

tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load("BipedalWalker-v2"))

tf.compat.v1.enable_v2_behavior()

time_step_spec = tf_env.time_step_spec()
observation_spec = time_step_spec.observation
action_spec = tf_env.action_spec()


def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


# actor_net.trainable_variables[1]
# actor_net.trainable_variables[1].assign_add(tf.compat.v2.random.normal((256,), stddev=0.1))


actor_net = actor_distribution_network.ActorDistributionNetwork(
    observation_spec,
    action_spec,
    fc_layer_params=(256, 256),
    continuous_projection_net=normal_projection_net)

print(actor_net.variables)

actor_copy = actor_net.copy()

actor_net.trainable_variables[1].assign_add(tf.compat.v2.random.normal((256,), stddev=0.1))

soft_variables_update(actor_copy.variables, actor_net.variables)

print(actor_net.variables)
