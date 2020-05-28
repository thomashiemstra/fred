from tf_agents.networks import actor_distribution_network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork, _categorical_projection_net, \
    _normal_projection_net
from tf_agents.utils import nest_utils
import tensorflow as tf


class ActorDistributionNetworkTrainable(ActorDistributionNetwork):

    def __init__(self, input_tensor_spec, output_tensor_spec, preprocessing_layers=None, preprocessing_combiner=None,
                 conv_layer_params=None, fc_layer_params=(200, 100), dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu, kernel_initializer=None, batch_squash=True, dtype=tf.float32,
                 discrete_projection_net=_categorical_projection_net, continuous_projection_net=_normal_projection_net,
                 name='ActorDistributionNetwork'):
        super().__init__(input_tensor_spec, output_tensor_spec, preprocessing_layers, preprocessing_combiner,
                         conv_layer_params, fc_layer_params, dropout_layer_params, activation_fn, kernel_initializer,
                         batch_squash, dtype, discrete_projection_net, continuous_projection_net, name)

    def call_raw(self,
             observations,
             step_type,
             network_state,
             training=False,
             mask=None):
        state, network_state = self._encoder(
            observations,
            step_type=step_type,
            network_state=network_state,
            training=training)
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)

        def call_projection_net(proj_net):
            means, stds = proj_net.call_raw(
                state, outer_rank, training=training, mask=mask)
            return means, stds

        means, stds = tf.nest.map_structure(
            call_projection_net, self._projection_networks)
        return means, stds, network_state
