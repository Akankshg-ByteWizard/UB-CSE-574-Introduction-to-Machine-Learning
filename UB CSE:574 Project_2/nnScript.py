import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    l1 = 1.0 / (1.0 + np.exp(-1.0 * z))
    return  l1

def feature_select(data):
    features = []
    D = data.shape[1] #Total d features
    for x in range(D):
        if sum(data[:, x]) > 0:
            features = features + [x]
    return features

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    # Pick a reasonable size for validation data

  
    trainpp = np.zeros(shape=(50000, 784))
    flagpp = np.zeros(shape=(10000, 784))
    testpp = np.zeros(shape=(10000, 784))
    trainlabelpp = np.zeros(shape=(50000,))
    flaglabelpp = np.zeros(shape=(10000,))
    ttestlabelpp = np.zeros(shape=(10000,))
   
    trainlength = 0
    flaglen = 0
    testlength = 0
    trainlabellength = 0
    flaglabellength = 0
 
    for key in mat:
        
        if "train" in key:
            label = key[-1]  
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  
            tag_len = tup_len - 1000  

            
            trainpp[trainlength:trainlength + tag_len] = tup[tup_perm[1000:], :]
            trainlength += tag_len

            trainlabelpp[trainlabellength:trainlabellength + tag_len] = label
            trainlabellength += tag_len

            
            flagpp[flaglen:flaglen + 1000] = tup[tup_perm[0:1000], :]
            flaglen += 1000

            flaglabelpp[flaglabellength:flaglabellength + 1000] = label
            flaglabellength += 1000

            
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            ttestlabelpp[testlength:testlength + tup_len] = label
            testpp[testlength:testlength + tup_len] = tup[tup_perm]
            testlength += tup_len
           
    train_size = range(trainpp.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = trainpp[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = trainlabelpp[train_perm]

    validation_size = range(flagpp.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = flagpp[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = flaglabelpp[vali_perm]

    test_size = range(testpp.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = testpp[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = ttestlabelpp[test_perm]

    # Feature selection
    # Your code here.

    global features
    features = []
    features = feature_select(train_data)                 
    train_data = train_data[:, features]
    validation_data = validation_data[:, features]
    test_data = test_data[:, features]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    N = training_label.shape[0]

    y = np.zeros((N, n_class))
    y[np.arange(N, dtype = "int"), training_label.astype(int)] = 1

    training_data = np.column_stack((np.array(training_data), np.array(np.ones(N))))

    z = np.dot(training_data,np.transpose(w1))
    Z = sigmoid(z)
    K = Z.shape[0]
    Z = np.column_stack((Z, np.ones(K)))
    o = np.dot(Z, np.transpose(w2))
    O = sigmoid(o)

    error_obtained = O - y

    grad_w1 = np.dot(((1 - Z) * Z * (np.dot(error_obtained, w2))).T, training_data)
    del_var = 0

    grad_w1 = np.delete(grad_w1, n_hidden, del_var)
    grad_w2 = np.dot(error_obtained.T, Z)

    obj_val = (np.sum(-1 * (y * np.log(O) + (1 - y) * np.log(1 - O)))) / training_data.shape[0] + ((lambdaval / (2 * training_data.shape[0])) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))
    
    grad_w1 = (grad_w1 + (lambdaval * w1)) / training_data.shape[0]
    grad_w2 = (grad_w2 + (lambdaval * w2)) / training_data.shape[0]

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    N, D = data.shape
    B = np.ones(N)*1
    data = np.column_stack((data, B))
    z = sigmoid(np.dot(data, np.transpose(w1)))
    o = sigmoid(np.dot(np.column_stack((z, B.T)), np.transpose(w2)))

    labels = o.argmax(axis = 1)
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 5

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

obj_dump = [features, n_hidden, w1, w2, lambdaval]
pickle.dump(obj_dump, open("params.pickle", "wb"))