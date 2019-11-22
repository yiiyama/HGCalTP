import sys

#path to caloGraphNN
sys.path.append('/afs/cern.ch/work/a/abgupta/deepjet2/caloGraphNN')
from caloGraphNN_keras import GlobalExchange, GarNet
# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {'tf':tf,'GlobalExchange':GlobalExchange,'GarNet':GarNet}
