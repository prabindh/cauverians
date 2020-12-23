# cauverians
Python utility for gradient pushing

Uses only numpy.

Written from scratch using only numpy (no Keras, TF, PyT, …)

Autograd based differentiator

numpy override by autograd

Not SGD or Batch, only GD

Configurable knobs

## Knobs

NN_EPOCHS = 200000
NN_LEARNING_RATE = 0.205
NN_PARAMS_NPY_FILE= None
NN_NORMALIZE = True
NN_LOG_TARGET = False
NN_MULTI_ENCODE_TEXT_VARS = False
NN_APPLY_DATA_SCIENCE = True
NN_SHUFFLE_ROWS_EPOCH = True

NN_BATCH_ROWS_EPOCH = False
NN_INPUT_TO_HIDDEN_MULTIPLIER = 3
NN_ZERO_MEAN_NORMALIZE = False # True will make zero mean set(with +,- values) so will not work rmsle
NN_RUN_MODE = "kaggle_home"
NN_SHAPE = "wide" # long, wide
NN_DROPOUT = False

## Debug configs
NN_DEBUG_PRINT_EPOCH_COUNT = 100
NN_DEBUG_EXIT_EPOCH_ONE = False
NN_DEBUG_SHAPES = False

Remove outliers, Sorted (Manual in csv)
