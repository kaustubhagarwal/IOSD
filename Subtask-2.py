import numpy as np
from util.paramInitializer import initialize_parameters  # import function to initialize weights and biases


class LinearLayer:

    def __init__(self, input_shape, n_out, ini_type="plain"):
        self.m = input_shape[1] 
        self.params = initialize_parameters(input_shape[0], n_out, ini_type) 
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))  

    def forward(self, A_prev):
        self.A_prev = A_prev =
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b'] 

    def backward(self, upstream_grad):
        self.dW = np.dot(upstream_grad, self.A_prev.T)
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

    def update_params(self, learning_rate=0.1):
        self.params['W'] = self.params['W'] - learning_rate * self.dW
        self.params['b'] = self.params['b'] - learning_rate * self.db 

class SigmoidLayer:
    def __init__(self, shape):
        self.A = np.zeros(shape)

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z)) 

    def backward(self, upstream_grad):
        self.dZ = upstream_grad * self.A*(1-self.A)

def compute_bce_cost(Y, P_hat):
    m = Y.shape[1]
    cost = (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    cost = np.squeeze(cost)
    dP_hat = (1/m) * (-(Y/P_hat) + ((1-Y)/(1-P_hat)))
    return cost, dP_hat

def compute_stable_bce_cost(Y, Z):
    m = Y.shape[1]
    cost = (1/m) * np.sum(np.maximum(Z, 0) - Z*Y + np.log(1+ np.exp(- np.abs(Z))))
    dZ_last = (1/m) * ((1/(1+np.exp(- Z))) - Y)  # from Z computes the Sigmoid so P_hat - Y, where P_hat = sigma(Z)
    return cost, dZ_last

def compute_keras_like_bce_cost(Y, P_hat, from_logits=False):
    if from_logits:
        return compute_stable_bce_cost(Y, Z=P_hat)

    else:
        EPSILON = 1e-07
        P_MAX = 1- EPSILON 
        P_hat = np.clip(P_hat, a_min=EPSILON, a_max=P_MAX)
        Z = np.log(P_hat/(1-P_hat))
        return compute_stable_bce_cost(Y, Z)

# number of samples in the train data set
N_TRAIN_SAMPLES = 50000
# number of samples in the test data set
N_TEST_SAMPLES = 2500
# number of samples in the validation data set
N_VALID_SAMPLES = 250
# number of classes
N_CLASSES = 2
# image size
IMAGE_SIZE = 28

((trainX, trainY), (testX, testY)) = #Load data

X_train = trainX[:N_TRAIN_SAMPLES, :, :]
y_train = trainY[:N_TRAIN_SAMPLES]

X_test = trainX[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLES, :, :]
y_test = trainY[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLES]

X_valid = testX[:N_VALID_SAMPLES, :, :]
y_valid = testY[:N_VALID_SAMPLES]


X_train = X_train / 255
X_train = np.expand_dims(X_train, axis=3)
y_train = convert_categorical2one_hot(y_train)
X_test = X_test / 255
X_test = np.expand_dims(X_test, axis=3)
y_test = convert_categorical2one_hot(y_test)
X_valid = X_valid / 255
X_valid = np.expand_dims(X_valid, axis=3)
y_valid = convert_categorical2one_hot(y_valid)

learning_rate = 1
number_of_epochs = 5000

np.random.seed(48)
 
Z1 = LinearLayer(input_shape=X_train.shape, n_out=1, ini_type='plain')
A1 = SigmoidLayer(Z1.Z.shape)

costs = []

for epoch in range(number_of_epochs):
    Z1.forward(X_train)
    A1.forward(Z1.Z)
    cost, dZ1 = compute_stable_bce_cost(Y_train, Z1.Z)
    if (epoch % 100) == 0 or epoch == number_of_epochs - 1:
        print("Cost at epoch#{}: {}".format(epoch, cost))
        costs.append(cost)  
    Z1.backward(dZ1)
    Z1.update_params(learning_rate=learning_rate)

np.save('weights.pkl',costs)

model = np.load('weights.pkl')

y_hat = model.predict(X_valid)
acc = softmax_accuracy(y_hat, y_valid)
print("Accuracy: ", acc)