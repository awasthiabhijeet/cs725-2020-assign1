import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        self.min=None
        self.max=None
    def __call__(self,features, is_train=False):
        m,n = features.shape
        if is_train:
            self.min = np.min(features,axis=0,keepdims=True)
            self.max = np.max(features,axis=0,keepdims=True)

        assert self.min is not None and self.max is not None
        features = (features - self.min)/(self.max - self.min + 1e-20)
        ones = np.ones([m,1]) #for bias
        features = np.concatenate([ones,features],1)
        return features


def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''

    def df_to_features(df):
        array = df.to_numpy()
        array = array[:,1:-1].astype(float)
        return array

    df = pd.read_csv(csv_path)
    features = df_to_features(df)
    if scaler is not None:
        features = scaler(features,is_train)
    
    return features

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    df = pd.read_csv(csv_path)
    df=df[' shares']
    return df.to_numpy().astype(float).reshape(len(df),1)
     

def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 4b
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape m x 1
    '''

    m,n = feature_matrix.shape
    #print(np.linalg.det(np.matmul(feature_matrix.T,feature_matrix)))
    #print(np.linalg.det(np.matmul(feature_matrix.T,feature_matrix) + C * np.eye(n)))

    solution = np.matmul(feature_matrix.T,feature_matrix) + C * np.eye(n)
    solution = np.matmul(np.linalg.inv(solution), np.matmul(feature_matrix.T,targets))
    return solution 

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''

    return np.matmul(feature_matrix,weights)

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    predictions = get_predictions(feature_matrix,weights)
    mse_loss = np.mean(np.power(predictions-targets,2))
    return mse_loss

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    return np.sum(np.power(weights,2))

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''

    loss = mse_loss(feature_matrix,weights,targets) + C * l2_regularizer(weights)
    return loss

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    m,n = feature_matrix.shape
    temp_1 = np.matmul(feature_matrix.T, (np.matmul(feature_matrix,weights)-targets))
    temp_2 = C * weights
    grad = (2 * (temp_1 + temp_2))/m
    return grad

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    pos = np.random.choice(np.arange(len(feature_matrix)), size=batch_size, replace=False)
    feature = np.array([feature_matrix[p] for p in pos])
    target = np.array([targets[p] for p in pos])
    return (feature, target)
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    return np.random.uniform(0,0.01,(n,1))
    #return np.random.randn(n,1)

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    

    weights = weights - lr*gradients
    return weights

def early_stopping(patience, step, patience_threshold, min_steps):
    # modify argument list as per need
    # return True or False
    if step < min_steps:
        return False
    if patience >= patience_threshold:
        return True
    else:
        return False
    

def do_gradient_descent(train_feature_matrix, 
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1e-15,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    m,n = train_feature_matrix.shape
    weights = initialize_weights(n)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    best_dev_loss = dev_loss
    best_weights = weights
    patience = 0

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):
        #print('weights: ',weights)
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        gradients = compute_gradients(features, weights, targets, C)
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))

            if dev_loss < best_dev_loss:
                patience = 0
                best_dev_loss = dev_loss
                best_weights = weights
            else:
                patience +=1
                if early_stopping(patience,step,patience_threshold=1000,min_steps=(2*m)/batch_size):
                    print('Stopping Early at step: {}'.format(step))
                    break
    return best_weights

def do_evaluation(feature_matrix, targets, weights):
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    scaler = Scaler()
    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',False,scaler), get_targets('data/dev.csv')
    test_features, test_targets = get_features('data/test.csv',False,scaler), get_targets('data/test.csv')

    a_solution = analytical_solution(train_features, train_targets, C=1e-8)
    print('evaluating analytical_solution...')
    test_loss=do_evaluation(test_features, test_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, test_loss: {} '.format(train_loss, test_loss))

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.1,
                        C=0.0001,
                        batch_size=32,
                        max_steps=2000000,
                        eval_steps=5)

    print('evaluating iterative_solution...')
    test_loss=do_evaluation(test_features, test_targets, gradient_descent_soln)
    print('gradient_descent_soln loss: {}'.format(test_loss))
    


