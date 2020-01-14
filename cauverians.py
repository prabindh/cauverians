"""
Utilities for performing regression using a neural network
Using only numpy
"""
import autograd.numpy as np
from autograd import grad
import math, random, time
from sklearn.model_selection import train_test_split
import sys, signal
import pickle

# https://towardsdatascience.com/linear-regression-from-scratch-with-numpy-implementation-finally-8e617d8e274c
class LinearRegression():
    def __init__(self, X, y, alpha=0.03, n_iter=1500):
        self.alpha = alpha
        self.n_iter = n_iter
        self.n_samples = len(y)
        self.n_features = np.size(X, 1)
        self.X = np.hstack((np.ones(
            (self.n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.params = np.zeros((self.n_features + 1, 1))
        self.coef_ = None
        self.intercept_ = None

    def fit(self):
        for i in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * \
            self.X.T @ (self.X @ self.params - self.y)

        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]
        return self

    def score(self, X=None, y=None):
        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack((np.ones(
                (n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]

        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())
        return score

    def predict(self, X):
        n_samples = np.size(X, 0)
        y = np.hstack((np.ones((n_samples, 1)), (X-np.mean(X, 0)) \
                            / np.std(X, 0))) @ self.params
        return y

    def get_params(self):
        return self.params

class cauverians():
    def __init__(self, in_config):
        np.seterr(all='raise')
        self.g_curr_epochs = 0
        self.g_exit_signalled = 0
        signal.signal(signal.SIGINT, self.signal_handler)
        self.config = in_config
        self.neighborhood_vals = {}
        self.g_contiguous_high_acc_epochs = 0
        self.g_highest_acc_state = {"acc": 0.0, "epoch": 1}
        self.nn_architecture = None
        self.params_values = None
        print ("Initialised cauverians")
        return
    def signal_handler(self, sig, frame):
            print('Setting exit flag...')
            self.g_exit_signalled = self.g_exit_signalled + 1
            if self.g_exit_signalled > 5:
                sys.exit(0)
    def make_network_arch(self, input_dim):
        if (input_dim % 2):
            print ("WARN: Number of features not multiple, ", input_dim)
        nn = []
        # set network configuration
        if self.config.NN_RUN_MODE == "line":
            nn.append({"layername": "in", "input_dim": input_dim, \
                "output_dim": 1, "activation": "identity"})
        else:
            final_activation = "relu"
            if self.config.NN_TYPE == "classifier":
                final_activation = "sigmoid"

            start_dim = input_dim*self.config.NN_INPUT_TO_HIDDEN_MULTIPLIER
            nn.append({"layername": "input", "input_dim": input_dim, \
                    "output_dim": int(start_dim), "activation": "relu"})
            if (self.config.NN_SHAPE == "wide"):
                for id in range(1000):
                    if (start_dim <= 60): break
                    div_factor = 2
                    nn.append({"layername": "hidden"+str(id+1), "input_dim": int(start_dim), \
                            "output_dim": int(start_dim/div_factor), "activation": "relu"})
                    start_dim = start_dim / div_factor
                nn.append({"layername": "output", "input_dim": int(start_dim), "output_dim": 1, \
                        "activation": final_activation})
            elif (self.config.NN_SHAPE == "long"): # long (from left to right)
                for id in range(4):
                    nn.append({"layername": "hidden-wide"+str(id+1), "input_dim": int(start_dim), \
                            "output_dim": int(start_dim), "activation": "relu"})
                nn.append({"layername": "output", "input_dim": int(start_dim), "output_dim": 1, \
                        "activation": final_activation})
            else:
                raise Exception("Unknown network_shape ", self.config.NN_SHAPE)

        self.nn_architecture = nn
        self.params_values = self.init_layers(2)
        return nn

    def generate_line(self):
        X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
        X = np.reshape(X, (X.shape[0],1))
        Y = 2.0 * X
        Y = np.reshape(Y, (Y.shape[0],1))
        # Normalize
        Y_normalize_state = X_normalize_state = None
        if (self.config.NN_NORMALIZE):
            Y, Y_normalize_state = self.normalize0(Y, axis=0)
            X, X_normalize_state = self.normalize0(X, axis=0)
        if (self.config.NN_DEBUG_SHAPES):
            print (X.shape, Y.shape, X[:, 0], Y[:,0], X[0][0].dtype)
        return X,X_normalize_state, None, Y, Y_normalize_state

        if self.config.NN_DEBUG_SHAPES:
            print (X.shape, Y.shape)
        return X, None, None, Y, None

    def multi_encode_text_variables(self, column_name, data, vals):
        return None, None

    def augment_training_data(self, data, target_name=None):
        recent_remodel = np.zeros((data.shape[0], 1))
        very_new_house = np.zeros((data.shape[0], 1))

        for col in range(data.shape[1]):
            if data[0][col] == "YearRemodAdd":
                remodelled = data[:, col]
                break
        for col in range(data.shape[1]):
            if data[0][col] == "YrSold":
                YrSold = data[:, col]
                break
        for col in range(data.shape[1]):
            if data[0][col] == "YearBuilt":
                YrBuilt = data[:, col]
                break

        for row in range(1,data.shape[0]):
            if (remodelled[row][0] == YrSold[row][0]):
                recent_remodel[row][0] = 1.0
        for row in range(1,data.shape[0]):
            if (YrBuilt[row][0] == YrSold[row][0]):
                very_new_house[row][0] = 1.0

        # Append new cols
        insert_pos = 1
        appended = np.hstack((data[:,:insert_pos], recent_remodel, data[:,insert_pos:]))
        return_data = np.hstack((appended[:,:insert_pos], very_new_house, appended[:,insert_pos:]))

        # remove rows only for training set (target_name not None)
        assert ("Id" == return_data[0][0])
        if target_name is not None:
            assert ("SalePrice" == return_data[0][return_data.shape[1]-1])
        if self.config.NN_DEBUG_SHAPES:
            print (return_data.shape)
        return return_data

    def filter_training_data(self, data, target_name = None):
        return None

    def read_housing_csv_2(self, file_name, x_mapping_state, target_name=None):
        return None, None, None, None, None


    def print_model(self):
        nn_architecture = self.nn_architecture
        params = self.params_values
        print ("Layer (name)\tInput_dim\tOutput_dim\tW shape")
        for idx, layer in enumerate(nn_architecture):
            print("{}\t{}\t{}\t{}".format(layer["layername"],
                        layer["input_dim"],
                        layer["output_dim"], params['W' + str(idx+1)].shape))


    def denormalize0(self, data, normalize_state):
        mean = normalize_state["mean"]
        var = normalize_state["var"]
        minimum = normalize_state["min"]
        maximum = normalize_state["max"]
        stdn = normalize_state["stdn"]
        if (self.config.NN_ZERO_MEAN_NORMALIZE == True):
            denorm = (stdn * data) + mean
        else:
            denorm = data * (maximum - minimum) + minimum
        return denorm
    # normalize based on each feature separately
    def normalize0(self, data, axis=0):

        assert (np.isfinite(data).all() == True)

        mean = np.mean(data, axis=axis)
        var = np.var(data, axis=axis)
        stdn = np.std(data, axis=axis)
        minimum_arr = np.amin(data, axis=axis, keepdims=True)
        maximum_arr = np.amax(data, axis=axis, keepdims=True)
        normalize_state = {"mean": mean, "var":var, "min": minimum_arr, "max": maximum_arr, "stdn": stdn}

        if (self.config.NN_ZERO_MEAN_NORMALIZE == True):
            normalized = (data - mean) / (stdn + 0.00001)
        else:
            normalized = (data - minimum_arr) / (maximum_arr - minimum_arr + 0.0001)

        return normalized.reshape(data.shape), normalize_state

    def init_layers(self, seed = 99):
        # random seed initiation
        np.random.seed(seed)
        # number of layers in our neural network
        number_of_layers = len(self.nn_architecture)
        # parameters storage initiation
        self.params_values = {}

        # W = np.random.randn(ndim, ndim)
        # u, s, v = np.linalg.svd(W)

        # iteration over network layers
        for idx, layer in enumerate(self.nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1

            # extracting the number of units in layers
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            # initiating the values of the W matrix
            # and vector b for subsequent layers
            # Wrand1 = np.random.randn(layer_input_size, layer_output_size)
            Wrand1 = np.ones((layer_input_size, layer_output_size))

            # Unused section
            """
            Wrand2 = np.random.random((layer_input_size, layer_output_size))
            u, s, v = np.linalg.svd(Wrand2, full_matrices=False)
            if self.config.NN_DEBUG_SHAPES:
                print (u.shape, v.shape, Wrand1.shape, Wrand2.shape)
            if layer_input_size <= layer_output_size:
                self.params_values['W' + str(layer_idx)] = v
            else:
                self.params_values['W' + str(layer_idx)] = u
            """
            self.params_values['W' + str(layer_idx)] = Wrand1
            self.params_values['b' + str(layer_idx)] = np.zeros((1, layer_output_size))
            mu, sigma = 0, 0.1
            self.params_values['b' + str(layer_idx)] = np.random.normal(mu, sigma, (1, layer_output_size))

        return self.params_values

    #Activation functions
    def sigmoid(self, Z):
        sigm = 1/(1+np.exp(-Z))
        return sigm

    def relu(self, Z):
        return np.maximum(0,Z)

    def sigmoid_backward(self, dA, Z):
        sig = sigmoid(Z)
        return dA @ sig @ (1 - sig)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z > 0] = 1
        dZ[Z <= 0] = 0
        return dZ
    def leaky_relu(self, Z):
        lrelu = np.where(Z > 0, Z, Z * 0.00001)
        return lrelu
    def leaky_relu_backward(self, dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0.00001 # Same coeff as forward!!
        return dZ
    def activation_identity(self, Z):
        return Z
    def activation_identity_backward(self, dA, Z):
        return dA

    def make_dropout(self, input, bound):
        rand_val = int(np.random.rand() * input.shape[1]* bound)
        # A_curr[rand_val:rand_val][:] = 0.0 # For relu, place before activation
        input[:, [rand_val]] = 0.0  # Zero all rows of this col
        return input


    def evaluate_single_layer(self, A_prev, W_curr, b_curr, activation="relu"):
        # calculation of the input value for the activation function
        #  Z = A*W + b
        #    Then Activation is applied A = activate(Z), to get Input to next layer


        #Z_curr = np.dot(A_prev, W_curr) + b_curr
        Z_curr = np.dot(A_prev, W_curr)

        # selection of activation function
        if activation is "relu":
            activation_func = self.relu
        elif activation is "sigmoid":
            activation_func = self.sigmoid
        elif activation is "leaky_relu":
            activation_func = self.leaky_relu
        elif activation is "identity":
            activation_func = self.activation_identity
        else:
            raise Exception('Non-supported activation function')

        return activation_func(Z_curr), Z_curr

    def evaluate_model(self, X):
        memory = {}
        A_curr = X
        dropout = self.config.NN_DROPOUT

        if dropout == True:
            A_curr = make_dropout(A_curr, 0.1)

        # iteration over network layers
        for idx, layer in enumerate(self.nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1
            # transfer the activation from the previous iteration
            A_prev = A_curr

            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]
            # extraction of W for the current layer
            W_curr = self.params_values["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = self.params_values["b" + str(layer_idx)]
            # calculation of activation for the current layer
            A_curr, Z_curr = self.evaluate_single_layer(A_prev, W_curr, b_curr, activ_function_curr)

            if (self.config.NN_DEBUG_SHAPES):
                print ("Forward: layer / A_curr/Z_curr shapes:", layer_idx, A_curr.shape, Z_curr.shape)

            # saving calculated values in the memory
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, memory

    # rmsle
    def evaluate_rmsle(self, Y_hat, Y):
        val = np.log1p(Y) - np.log1p(Y_hat)
        cost = - np.sqrt(np.mean(val**2))
        if self.config.NN_REGULARISATION:
            cost = cost #todo
        return cost
    def evaluate_rmse(self, Y_hat, Y):
        rmse = - np.sqrt(((Y_hat - Y) ** 2).mean())
        return rmse
    def evaluate_mse(self, loss):
        mse = - np.mean(loss ** 2) / 2
        return mse

    def evaluate_grad_mse_y_hat(self, loss):
        gradient = loss
        return gradient


    def evaluate_cost_value(self, X, Y_hat, Y, method, derivative=None):
        minutiae = 0.0001
        # number of examples
        m = Y_hat.shape[0]
        derivative_cost = cost = None
        if (method is "cross_entropy"):
            if (derivative is None):
                # calculation of the cross entropy
                cost = (-1 / m) * (np.dot(Y.T, np.log(Y_hat + minutiae)) +
                                np.dot(1 - Y.T, np.log(1 - Y_hat + minutiae)))
            else:
                # Calculation of first derivative
                derivative_cost = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        elif (method is "mean_sq_error"):
            loss = (Y_hat - Y)
            if (derivative is None):
                cost = self.evaluate_mse(loss)
            else:
                # Calculation of non-normalised first derivative
                #grad_mse = grad(self.evaluate_mse)
                #derivative_cost = (-1) * grad_mse(loss)
                derivative_cost = self.evaluate_grad_mse_y_hat(loss)
        elif (method is "root_mean_sq_log_error"):
            if (derivative is None):
                # Calculation of rmsle
                cost = self.evaluate_rmsle(Y_hat, Y)
            else:
                grad_rmsle = grad(self.evaluate_rmsle)
                derivative_cost = (-1) * grad_rmsle(Y_hat, Y)
        elif (method is "root_mean_sq_error"):
            if (derivative is None):
                # Calculation of rmse
                cost = self.evaluate_rmse(Y_hat, Y)
            else:
                grad_rmse = grad(evaluate_rmse)
                derivative_cost = -1 * grad_rmse(Y_hat, Y)
        else:
            raise Exception ("Error: Unknown cost method {}".format(method))
        return np.squeeze(cost), derivative_cost

    def check_training_stop(self, state):
        accuracy = 5.0
        rms = state["rms"]
        cost = state["cost"]
        epochs = state["epochs"]
        if (abs(cost) < self.config.COST_CUTOFF):
            self.g_contiguous_high_acc_epochs = self.g_contiguous_high_acc_epochs + 1
        else:
            self.g_contiguous_high_acc_epochs = 0
        # max acc check, disregard trivial values
        if (abs(accuracy) > 0.01 and abs(accuracy) < self.g_highest_acc_state["acc"]):
            self.g_highest_acc_state["acc"] = abs(accuracy)
            self.g_highest_acc_state["epoch"] = epochs
        stop = False
        if (self.g_contiguous_high_acc_epochs > self.config.MIN_CONTIGUOUS_HIGH_ACC_EPOCHS):
            stop = True
        # Prevent overfitting
        if(abs(accuracy) < self.config.ACCURACY_CUTOFF):
            stop = True
        return stop
    # an auxiliary function that converts probability into class
    # Poor man's sigmoid
    def convert_prob_into_class(self, probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    """
    Kaggle house price Submissions are evaluated on Root-Mean-Squared-Error (RMSE)
    between the logarithm of the predicted value and the logarithm of the observed sales price
    """
    def get_accuracy_value(self, Y_hat, Y):
        if self.config.NN_TYPE == "classifier":
            Y_hat_ = convert_prob_into_class(Y_hat)
            acc = (Y_hat_ == Y).all(axis=0).mean()
        else:
            if self.config.NN_RUN_MODE == "line":
                loss = Y_hat - Y
                acc = self.evaluate_mse(loss)
            else:
                acc = self.evaluate_rmsle(Y_hat, Y)
            # If already kaggle Y is in log, so just use rmse
            # acc = evaluate_rmse(Y_hat, Y)
        return acc

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        # number of examples
        m = A_prev.shape[0]

        # selection of activation function
        if activation is "relu":
            backward_activation_func = self.relu_backward
        elif activation is "leaky_relu":
            backward_activation_func = self.leaky_relu_backward
        elif activation is "sigmoid":
            backward_activation_func = self.sigmoid_backward
        elif activation is "identity":
            backward_activation_func = self.activation_identity_backward
        else:
            raise Exception('Non-supported activation function')

        # calculation of the activation function derivative
        dZ_curr = backward_activation_func(dA_curr, Z_curr)

        # derivative of the matrix W
        dW_curr = np.dot(dZ_curr.T, A_prev) / m
        # derivative of the vector b
        db_curr = np.sum(dZ_curr, keepdims=True) / m
        if (self.config.NN_DEBUG_SHAPES):
            print ("Back: W_curr/dW_curr, b_curr/db_curr shape:", W_curr.shape, \
                    dW_curr.shape, b_curr.shape, db_curr.shape)
        # derivative of the matrix A_prev
        dA_prev = np.dot(dZ_curr, W_curr.T)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, X, Y_hat, Y, memory):
        grads_values = {}

        # number of examples
        m = Y.shape[0]
        # initiation of gradient descent algorithm
        _, dA_prev = self.evaluate_cost_value(X, Y_hat, Y, self.config.NN_ARCHITECTURE_LOSS_TYPE, "first_derivative")

        if (self.config.NN_DEBUG_GRADIENTS):
            print ("dA_prev = ", dA_prev, "Y_hat0,Y=", Y_hat[0], Y[0])

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            # we number network layers from 1
            layer_idx_curr = layer_idx_prev + 1
            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]

            W_curr = self.params_values["W" + str(layer_idx_curr)]
            b_curr = self.params_values["b" + str(layer_idx_curr)]

            if (self.config.NN_DEBUG_SHAPES):
                print ("Back: shape dA/W/b/Z/A_prev = ",dA_curr.shape, W_curr.shape, \
                            b_curr.shape, Z_curr.shape, A_prev.shape)

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr
        return grads_values

    def update_model(self, grads_values, learning_rate):

        # iteration over network layers
        for layer_idx, layer in enumerate(self.nn_architecture, 1):
            wRate = learning_rate * grads_values["dW" + str(layer_idx)]
            bRate = learning_rate * grads_values["db" + str(layer_idx)]
            if (self.config.NN_DEBUG_SHAPES):
                print ("layer-name=",layer["layername"], " shape wRate/param_W/bRate/param_b = ",
                    wRate.shape, self.params_values["W" + str(layer_idx)].shape,
                    bRate.shape, self.params_values["b" + str(layer_idx)].shape)
            self.params_values["W" + str(layer_idx)] -= wRate.T
            self.params_values["b" + str(layer_idx)] -= bRate

        return self.params_values

    def train_model(self, X, Y):
        if self.config.NN_PARAMS_NPY_FILE is not None:
            # initiation of neural net parameters from prev
            pkl_file = open(self.config.NN_PARAMS_NPY_FILE, 'rb')
            self.params_values = pickle.load(pkl_file)
            pkl_file.close()

        train_state = {}
        merged = np.append(X,Y, axis=1)

        if (self.config.NN_DEBUG_SHAPES):
            print ("X/Y input shape = ", X.shape, Y.shape)
        # performing calculations for subsequent iterations
        for i in range(self.config.NN_EPOCHS):
            self.g_curr_epochs = i
            # step forward

            if self.config.NN_SHUFFLE_ROWS_EPOCH:
                np.random.shuffle(merged)
                X_batch = merged[0:,0:merged.shape[1]-1]
                Y_batch = merged[0:,-1].reshape(merged.shape[0],1)
            else:
                X_batch = np.array(X, copy=True)
                Y_batch = np.array(Y, copy=True)

            if self.config.NN_BATCH_ROWS_EPOCH:
                delete_n = int(X_batch.shape[0]/4)
                X_batch = X_batch[:-delete_n, :]
                Y_batch = Y_batch[:-delete_n, :]
            Y_hat, cache = self.evaluate_model(X_batch)
            if (self.config.NN_DEBUG_SHAPES):
                print ("Y_hat shape = ", Y_hat.shape)
            cost, _ = self.evaluate_cost_value(X_batch, Y_hat, Y_batch, self.config.NN_ARCHITECTURE_LOSS_TYPE) # Kaggle measured logloss
            rms_error, _ = self.evaluate_cost_value(X_batch, Y_hat, Y_batch, "root_mean_sq_error")

            # step backward - calculating gradient
            grads_values = self.full_backward_propagation(X_batch, Y_hat, Y_batch, cache)
            # updating model state
            self.params_values = self.update_model(grads_values, self.config.NN_LEARNING_RATE)

            if(i % self.config.NN_DEBUG_PRINT_EPOCH_COUNT == 0):
                print("Epoch: {:06} - cost: {:.5f} - rms_error: {:.5f}".format(self.g_curr_epochs, cost, -rms_error))
                if self.config.NN_DEBUG_GRADIENTS:
                    print(grads_values)

            train_state["cost"] = cost
            train_state["rms"] = rms_error
            train_state["epochs"] = self.g_curr_epochs
            stop = self.check_training_stop(train_state)
            if (True == stop or self.config.NN_DEBUG_EXIT_EPOCH_ONE == True):
                print ("Breaking out of training, state = ", train_state)
                break
            if self.g_exit_signalled > 0:
                break
        self.print_model()
        return self.params_values

    def save_params(self, filename):
        pklfile = open(filename, 'wb')
        pickle.dump(self.params_values, pklfile)
        pklfile.close()
        print ("Model output and Parameters saved")

    def save_kaggle(self, predictions, filename):
        Id = np.empty([2919-1461+1,1], dtype=int)
        for x in range(2919-1461+1):
            Id[x][0] = x + 1461
        predictions = np.append(Id, predictions,1)
        np.savetxt(filename,predictions, fmt="%d,%d", delimiter=",")