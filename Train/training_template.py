
from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense #etc

def my_model(Inputs,nclasses,nregressions,otheroption):
    
    input_a = Inputs[0] #this is the self.x list from the TrainData data structure
    x = Dense(2)(input_a)
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=True)


if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model,otheroption=1)
    
    train.compileModel(learningrate=0.01,
                   loss='mean_squared_error') 
                   

model,history = train.trainModel(nepochs=10, 
                                 batchsize=100,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)

