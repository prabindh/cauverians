##
# Some aspects derived from  https://github.com/SkalskiP/ILearnDeepLearning.py
# Added own data generator, changed matrix types, added mse, normalize, minutiae
# Kaggle
# normalize each input separately not just x
# add more layers
# convert text input to a parametric form - simple map to 0,1,2, ???
# input Data reading from .csv format, cleanup, make to {x1,x2,...} datapoints
# programmatic creation of net architecture
# And finally does not work properly in unit tests
# (c) prabindh 2020

import autograd.numpy as np
import math, random, time
import sys, signal
from cauverians import cauverians as cauverians

class create_config():
    def __init__(self):
        ### CONFIGURATIONS ###############################################
        self.MIN_CONTIGUOUS_HIGH_ACC_EPOCHS = 100
        self.COST_CUTOFF = 0.001
        self.ACCURACY_CUTOFF = 0.017
        # # number of samples in the test data set
        self.N_SAMPLES = 100
        self.NN_EPOCHS = 10000
        # ratio between training and test sets
        self.NN_TEST_SIZE = 0.1
        self.NN_LEARNING_RATE = 0.0005
        self.NN_PARAMS_NPY_FILE= None   # None or .pkl file
        self.NN_LOG_TARGET = False
        self.NN_MULTI_ENCODE_TEXT_VARS = False
        self.NN_APPLY_DATA_SCIENCE = True
        self.NN_SHUFFLE_ROWS_EPOCH = False
        self.NN_BATCH_ROWS_EPOCH = False
        self.NN_INPUT_TO_HIDDEN_MULTIPLIER = 1
        self.NN_NORMALIZE = False
        self.NN_ZERO_MEAN_NORMALIZE = False # True will make zero mean set(with +,- values) so will not work rmsle
        self.NN_RUN_MODE = "line" #"kaggle_home" # separated_datapoints or kaggle_home
        self.NN_SHAPE = "wide" # long, wide
        self.NN_DROPOUT = False
        self.NN_REGULARISATION = True

        # Debug configs
        self.NN_DEBUG_PRINT_EPOCH_COUNT = 100
        self.NN_DEBUG_EXIT_EPOCH_ONE = False
        self.NN_DEBUG_SHAPES = False
        self.NN_DEBUG_GRADIENTS = False
        # Derived values
        if self.NN_RUN_MODE == "kaggle_home":
            self.NN_TYPE = "regressor"
        elif self.NN_RUN_MODE == "separated_datapoints":
            self.NN_TYPE = "classifier"
        elif self.NN_RUN_MODE == "line":
            self.NN_TYPE = "regressor"

        if self.NN_TYPE == "classifier":
            self.NN_ARCHITECTURE_LOSS_TYPE = "cross_entropy"
        elif self.NN_TYPE == "regressor":
            if self.NN_RUN_MODE == "kaggle_home":
                # "root_mean_sq_log_error" to be used, if log.Y is not taken reading from CSV. Else "root_mean_sq_error"
                self.NN_ARCHITECTURE_LOSS_TYPE = "root_mean_sq_log_error"
                # self.NN_ARCHITECTURE_LOSS_TYPE = "root_mean_sq_error"
            else:
                self.NN_ARCHITECTURE_LOSS_TYPE = "mean_sq_error"
        if self.NN_DEBUG_SHAPES:
            self.NN_DEBUG_EXIT_EPOCH_ONE = True

###############################################################
# Use cauvery to perform regression
###############################################################
config = create_config()
cauvery = cauverians(config)
if(config.NN_RUN_MODE == "kaggle_home"):
    X_mapping = []  # Same mapping to be used in train/test !!
    X_train,X_train_normalize_state, X_mapping, Y_train, Y_train_normalize_state = \
        cauvery.read_housing_csv_2("kaggle-housing-price/train.csv", X_mapping, "SalePrice")
    X_test, X_test_normalize_state, X_mapping, Y_test, _ = \
        cauvery.read_housing_csv_2("kaggle-housing-price/test.csv", X_mapping)
else:
    X_train, X_train_normalize_state, _, Y_train, Y_train_normalize_state = cauvery.generate_line()
    X_test, X_test_normalize_state, _, Y_test, Y_test_normalize_state = cauvery.generate_line()

# Training
input_size = X_train.shape[1]
arch = cauvery.make_network_arch(input_size)
cauvery.print_model()
params_values = cauvery.train_model(X_train, Y_train)

# Prediction
Y_test_hat, _ = cauvery.evaluate_model(X_test)

if config.NN_DEBUG_EXIT_EPOCH_ONE is True:
    print ("Exiting due to config.NN_DEBUG_EXIT_EPOCH_ONE...")
    sys.exit(-1)

# Accuracy achieved on the test set
if (config.NN_RUN_MODE == "line"):
    acc_test = cauvery.get_accuracy_value(Y_test_hat, Y_test)
    if config.NN_NORMALIZE:
        Y_test_hat_denormalized = cauvery.denormalize0(Y_test_hat,
                    Y_train_normalize_state)
        Y_test_denormalized = cauvery.denormalize0(Y_test,
                    Y_train_normalize_state)
        print (params_values, "y_test_hat=",Y_test_hat, "y_test=",Y_test_denormalized)
    else:
        print (params_values, "y_test_hat=",Y_test_hat, "y_test=",Y_test)
    print("Numpy test accuracy: {:.2f}".format(acc_test))
else:
    Y_test_hat_denormalized = cauvery.denormalize0(Y_test_hat,
                    Y_train_normalize_state)
    # Take anti-log of saleprice to get actuals, if log of price used for training
    if config.NN_LOG_TARGET is True:
        Y_test_hat_denormalized = np.exp(Y_test_hat_denormalized)
    #print (Y_test_hat_denormalized)
    timestr = str(time.time())
    cauvery.save_kaggle(Y_test_hat_denormalized, "submission-"+ timestr +".csv")
    cauvery.save_params("params-"+ timestr +".pkl")
print ("Exiting ...")
