from tf_agents.networks.normal_projection_network import NormalProjectionNetwork
from tf_agents.networks import utils as network_utils
import tensorflow as tf


def tanh_squash_to_spec(inputs, spec):
    """Maps inputs with arbitrary range to range defined by spec using `tanh`."""
    means = (spec.maximum + spec.minimum) / 2.0
    magnitudes = (spec.maximum - spec.minimum) / 2.0

    return means + magnitudes * tf.tanh(inputs)


class NormalProjectionNetworkTrainable(NormalProjectionNetwork):

    def __init__(self, sample_spec, activation_fn=None, init_means_output_factor=0.1, std_bias_initializer_value=0.0,
                 mean_transform=tanh_squash_to_spec, std_transform=tf.nn.softplus,
                 state_dependent_std=False,
                 scale_distribution=False, name='NormalProjectionNetwork'):
        super().__init__(sample_spec, activation_fn, init_means_output_factor, std_bias_initializer_value,
                         mean_transform, std_transform, state_dependent_std, scale_distribution, name)

    def call_raw(self, inputs, outer_rank, training=False, mask=None):
        if inputs.dtype != self._sample_spec.dtype:
            raise ValueError(
                'Inputs to NormalProjectionNetwork must match the sample_spec.dtype.')

        if mask is not None:
            raise NotImplementedError(
                'NormalProjectionNetwork does not yet implement action masking; got '
                'mask={}'.format(mask))

        # outer_rank is needed because the projection is not done on the raw
        # observations so getting the outer rank is hard as there is no spec to
        # compare to.
        batch_squash = network_utils.BatchSquash(outer_rank)
        inputs = batch_squash.flatten(inputs)

        means = self._means_projection_layer(inputs, training=training)
        means = tf.reshape(means, [-1] + self._sample_spec.shape.as_list())

        # If scaling the distribution later, use a normalized mean.
        if not self._scale_distribution and self._mean_transform is not None:
            means = self._mean_transform(means, self._sample_spec)
        means = tf.cast(means, self._sample_spec.dtype)

        if self._state_dependent_std:
            stds = self._stddev_projection_layer(inputs, training=training)
        else:
            stds = self._bias(tf.zeros_like(means), training=training)
            stds = tf.reshape(stds, [-1] + self._sample_spec.shape.as_list())

        if self._std_transform is not None:
            stds = self._std_transform(stds)
        stds = tf.cast(stds, self._sample_spec.dtype)

        means = batch_squash.unflatten(means)
        stds = batch_squash.unflatten(stds)

        return means, stds
