import sys
import keras
import tensorflow as tf

sys.path.append('/afs/cern.ch/user/y/yiiyama/src/caloGraphNN')
import caloGraphNN_keras as cgnn

class GarNetClassificationModel(keras.Model):
    def __init__(self, n_classes, momentum=0.99):
        super(GarNetClassificationModel, self).__init__()

        aggregators = [4] * 11
        filters = [32] * 11
        propagate = [20] * 11
        
        self.blocks = []

        block_params = zip(aggregators, filters, propagate)

        self.input_gex = self._add_layer(cgnn.GlobalExchange, name='input_gex')
        self.input_batchnorm = self._add_layer(keras.layers.BatchNormalization, momentum=momentum, name='input_batchnorm')
        self.input_dense = self._add_layer(keras.layers.Dense, 32, activation='tanh', name='input_dense')

        for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
            garnet = self._add_layer(cgnn.GarNet, n_aggregators, n_filters, n_propagate, name='garnet_%d' % i)
            batchnorm = self._add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm_%d' % i)

            self.blocks.append((garnet, batchnorm))

        self.output_dense_0 = self._add_layer(keras.layers.Dense, 48, activation='relu', name='output_0')
        self.output_dense_1 = self._add_layer(keras.layers.Dense, 3, activation='relu', name='output_1')

    def call(self, inputs):
        feats = []

        x = inputs[0]

        x = self.input_gex(x)
        x = self.input_batchnorm(x)
        x = self.input_dense(x)

        for block in self.blocks:
            for layer in block:
                x = layer(x)

            feats.append(x)

        x = tf.concat(feats, axis=-1)

        x = self.output_dense_0(x)
        x = self.output_dense_1(x)

        return x

    def _add_layer(self, cls, *args, **kwargs):
        layer = cls(*args, **kwargs)
        self._layers.append(layer)
        return layer


def make_garnet(inputs, n_classes, n_regressions, other_options=[], dropout_rate=0.01, momentum=0.8):
    x = inputs[1]

    vertex_mask = keras.layers.Lambda(lambda x: tf.cast(tf.not_equal(x[..., 9:10], 0.), tf.float32))(x)

    x = cgnn.GlobalExchange(name='input_gex', vertex_mask=vertex_mask)(x)
    x = keras.layers.BatchNormalization(momentum=momentum, name='input_batchnorm')(x)
    x = keras.layers.Dense(16, activation='tanh', name='input_dense')(x)

    aggregators = [4] * 11
    filters = [32] * 11
    propagate = [20] * 11
    block_params = zip(aggregators, filters, propagate)

    feats = []

    for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
        x = cgnn.GarNet(n_aggregators, n_filters, n_propagate, name='garnet_%d' % i, vertex_mask=vertex_mask)(x)
        x = keras.layers.BatchNormalization(momentum=momentum, name='batchnorm_%d' % i)(x)
        x = keras.layers.Multiply()([vertex_mask, x])
        feats.append(x)

    x = keras.layers.Concatenate(axis=-1)(feats)

    x = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
    print('reduced', x)
    x = keras.layers.Dense(48, activation='relu', name='output_0')(x)
    x = keras.layers.Dense(n_classes, activation='softmax', name='output_1')(x)

    return keras.Model(inputs=inputs, outputs=[x])


if __name__ == '__main__':
    from DeepJetCore.training.training_base import training_base
    
    train = training_base(testrun=False, resumeSilently=False, renewtokens=True)

    if not train.modelSet():
        # for regression use the regression model
        train.setModel(make_garnet)
    
        # for regression use a different loss, e.g. mean_squared_error
        train.compileModel(learningrate=0.00005,
                           loss='categorical_crossentropy')
    
    model, history = train.trainModel(nepochs=100,
                                      batchsize=128,
                                      checkperiod=200,  # saves a checkpoint model every N epochs
                                      verbose=1)
