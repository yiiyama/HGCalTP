import sys
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda
sys.path.append('/afs/cern.ch/work/a/abgupta/deepjet2/calo2/caloGraphNN')
import caloGraphNN_keras as cgnn
import numpy as np
def make_garnet(inputs, n_classes, n_regressions, other_options=[], dropout_rate=0.05, momentum=0.9):
    #print(np.shape(inputs))
    x = inputs[0]
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    #vertex_mask = keras.layers.Lambda(lambda x: tf.cast(tf.not_equal(x[..., 9:10], 0.), tf.float32))(x)
    #z_mask = cgnn.CreateZeroMask()(x)
    #x = cgnn.GlobalExchange(name='input_gex')(x)
    #x = keras.layers.BatchNormalization(momentum=momentum, name='input_batchnorm')(x)
    #x = keras.layers.Dense(16, activation='tanh', name='input_dense')(x)
    #x = keras.layers.BatchNormalization(momentum=momentum)(x)
    #

#Normal
    x = cgnn.GarNet5(4,4,4,name='gar_1')(x)
    #x = keras.layers.BatcihNormalization(momentum=momentum)(x)
    #x = keras.layers.Multiply()([z_mask, x])
    # x1 = x
    #x = cgnn.GarNet(6,6,6,name='gar_2')(x)
    # x = keras.layers.BatchNormalization(momentum=momentum)(x)
    # x = keras.layers.Multiply()([z_mask, x])
    # x2 = x
    #x = cgnn.GarNet(4,4,4,name='gar_3')(x)
    # x = keras.layers.BatchNormalization(momentum=momentum)(x)
    # x = keras.layers.Multiply()([z_mask, x])
    # x3 = x
    #x = cgnn.GarNet(4,4,4,name='gar_4')(x)
    # x = keras.layers.BatchNormalization(momentum=momentum)(x)
    # x = keras.layers.Multiply()([z_mask, x])
    # x4 = x

#Kinda inception
    # t1 = cgnn.GarNet(8,8,8,name='gar_1')(x)
    # t1 = keras.layers.BatchNormalization(momentum=momentum)(t1)
    # # t1 = keras.layers.Multiply()([z_mask, t1])
    # # #x = x
    # t2 = cgnn.GarNet2(8,8,8,name='gar_2')(x)
    # t2 = keras.layers.BatchNormalization(momentum=momentum)(t2)
    # t2 = keras.layers.Multiply()([z_mask, t2])
    # x2 = x
    # x = cgnn.GarNet(4,4,4,name='gar_3')(x)
    # x = keras.layers.BatchNormalization(momentum=momentum)(x)
    # x = keras.layers.Multiply()([z_mask, x])
    # x3 = x
    # x = cgnn.GarNet(4,4,4,name='gar_4')(x)
    # x = keras.layers.BatchNormalization(momentum=momentum)(x)
    # x = keras.layers.Multiply()([z_mask, x])
    # x4 = x

    #x = keras.layers.Concatenate(axis=-1)([t1,t2])

    # for i, (n_aggregators, n_filters, n_propagate) in enumerate(block_params):
    #     x = cgnn.GarNet(n_aggregators, n_filters, n_propagate, name='garnet_%d' % i)(x)
    #     x = keras.layers.BatchNormalization(momentum=momentum, name='batchnorm_%d' % i)(x)
    #     x = keras.layers.Multiply()([z_mask, x])
    #     feats.append(x)

    #x = keras.layers.Concatenate(axis=-1)([x1,x2,x3,x4])


    #print(x.shape)
    x = keras.layers.AveragePooling1D(pool_size=250, data_format='channels_last')(x)
    x = keras.layers.Flatten()(x)
    #x = Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
    #x = keras.layers.Dense(tf.reduce_mean(x,axis=1))(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)

    print('reduced', x)
    x = keras.layers.Dense(16, activation='relu')(x)
    x = keras.layers.BatchNormalization(momentum=0.8)(x)
    x = keras.layers.Dense(4, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=[x])


if __name__ == '__main__':
    from DeepJetCore.training.training_base import training_base

    train = training_base(testrun=False, resumeSilently=False, renewtokens=True)

    if not train.modelSet():
        # for regression use the regression model
        train.setModel(make_garnet)

        # for regression use a different loss, e.g. mean_squared_error
        train.compileModel(learningrate=0.0005,
                           loss='categorical_crossentropy')
    print(train.keras_model.summary())
    #from keras.utils import plot_mode
    class_weight = {0: 1., 1: 500., 2: 3.75, 3: 1.}
    model, history = train.trainModel(nepochs=30,
                                      batchsize=512,
                                      checkperiod=50,  # saves a checkpoint model every N epochs
                                      verbose=1)
                                      #class_weight=class_weight)
    #plot_model(model, to_file='graph_1.png')
    from keras.models import model_from_json
    model_json = model.to_json()
    with open("/afs/cern.ch/work/a/abgupta/public/gar_f.json","w") as json_file:
        json_file.write(model_json)
