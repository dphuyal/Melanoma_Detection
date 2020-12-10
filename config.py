# train set paths
TRAIN_CSV_PATH = '../new_dataset/'  # roman's dataset
TRAIN_CSV_PATH_20 = '../dataset_256/'
#data_path = '../dataset_256/'

# test set paths
TEST_CSV_PATH_20 = '../dataset_256/'

# submission file path provided by Kaggle
SUB_FILE_PATH_20 = '../dataset_256/sample_submission.csv'

# select efficientnet models or resnet-50 
model_type = 'efficientnet'
# dictionary to map b2 with efficientnet-b2, b3 with efficientnet-b3...
effnet_dict  = {'b2':'efficientnet-b2'}
# select either b2, b3, b4...
effnet_type = 'b2'

# STATICS
batch_size = 64
n_split = 5
epochs = 24
learning_rate = 0.0005
es_patience = 3 # early stopping patience - for how many epochs with no improvements to wait
image_size = 256

output_size = 1 
num_cols = 3 

model_path_ = 'saved_models_resnet50'
logs_path_ = 'logs'