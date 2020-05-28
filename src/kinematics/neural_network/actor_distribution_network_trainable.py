from tf_agents.networks import actor_distribution_network
from tf_agents.utils import nest_utils
import tensorflow as tf


class ActorDistributionNetworkTrainable(actor_distribution_network):

    def call(self,
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
            distribution, _ = proj_net(
                state, outer_rank, training=training, mask=mask)
            return distribution

        output_actions = tf.nest.map_structure(
            call_projection_net, self._projection_networks)
        return output_actions, network_state
