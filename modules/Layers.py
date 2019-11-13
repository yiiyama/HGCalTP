
# Define custom layers here and add them to the global_layers_list dict (important!)
#global_layers_list = {'GlobalExchange':layer.__GlobalExchange__}
import tensorflow as tf
import keras
import sys

#change this path to your calGraphNN
sys.path.append('/afs/cern.ch/work/a/abgupta/deepjet2/calo2/caloGraphNN')

#from caloGraphNN import euclidean_squared, gauss, gauss_of_lin
from caloGraphNN_keras import GlobalExchange, GarNet, CreateZeroMask, GarNet2, GarNet3, GarNet4, GarNet5

global_layers_list = {'tf':tf,'GlobalExchange':GlobalExchange, 'GarNet':GarNet,'CreateZeroMask':CreateZeroMask,'GarNet2':GarNet2, 'GarNet3':GarNet3, 'GarNet4':GarNet4, 'GarNet5':GarNet5}
