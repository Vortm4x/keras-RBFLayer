from keras import ops
from keras.api.layers import Layer
from keras.src import initializers
from keras.src import regularizers
from keras.src import constraints

class RBFLayer(Layer):
    def __init__(
        self,
        units: int,
        gamma: float,
        mu_initializer='uniform',
        mu_regularizer=None,
        mu_constraint=None,
        activity_regularizer=None,
        **kwargs
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.gamma = ops.cast(gamma, 'float32')
        self.mu_initializer = initializers.get(mu_initializer)
        self.mu_regularizer = regularizers.get(mu_regularizer)
        self.mu_constraint = constraints.get(mu_constraint)


    def build(self, input_shape):
        input_dim=int(input_shape[-1])

        self._mu = self.add_weight(
            name='mu',
            shape=(self.units, input_dim),
            initializer=self.mu_initializer,
            regularizer=self.mu_regularizer,
            constraint=self.mu_constraint,
        )
        super(RBFLayer, self).build(input_shape)
        self.built = True

    @property
    def mu(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `mu`."
            )
        return self._mu

    def call(self, inputs):
        diff = ops.expand_dims(inputs, 1) - self._mu
        l2 = ops.sum(ops.square(diff), axis=1)
        res = ops.exp(-self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "mu_initializer": initializers.serialize(
                self.mu_initializer
            ),
            "mu_regularizer": regularizers.serialize(
                self.mu_regularizer
            ),
            "mu_constraint": constraints.serialize(self.mu_constraint),
        }
        return {**base_config, **config}
