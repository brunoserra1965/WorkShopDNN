import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

class DeepNeuralNetwork:
    def _initialize_parameters(self, layer_neurons):
        np.random.seed(1)

        L = len(layer_neurons)
        params = {}
        
        for i in range(1, L):
            params[f"W{i}"] = np.random.randn(layer_neurons[i], layer_neurons[i-1]) / np.sqrt(layer_neurons[i-1])
            params[f"b{i}"] = np.zeros((layer_neurons[i], 1), dtype=int) 

        self.params = params

    def _sigmoid(self, X):
        return 1 / ( 1 + np.exp(-X) )
        
    def _relu(self, X):
        return np.maximum(0,X)
    
    def _calculate_cost(self, Y, Yhat):
        m = Y.shape[1]
        cost = (-1./m) * np.sum((Y @ np.log(Yhat).T) + ((1 - Y) @ np.log(1 - Yhat).T))
        cost = np.squeeze(cost)
        return cost
        
    def _sigmoid_derivative(self, X, dA):
        s = self._sigmoid(X)
        return dA * s * (1 - s)  

    def _relu_derivative(self, X, dA):
        dZ = np.array(dA, copy=True) 
        dZ[X <= 0] = 0
        return dZ
    
    def _activation_derivative(self, Z, dA, activation):
        if activation == "sigmoid":
            return self._sigmoid_derivative(Z, dA)
        elif activation == "relu":
            return self._relu_derivative(Z, dA)

    def _activation(self, Z, activation):
        if activation == "sigmoid":
            return self._sigmoid(Z)
        elif activation == "relu":
            return self._relu(Z)

    def _cost_derivative(self, Y, Yhat):
        return - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))

    def _forward_propagate(self, A, W, b, activation = "sigmoid"):
        Z_next = W @ A + b
        A_next = self._activation(Z_next, activation)
                
        return Z_next, A_next

    def _forward_propagate_all(self, X, layer_neurons, activation = "sigmoid"):
        L = len(layer_neurons)
        self.intermediate_res = {}
        
        A = X
        self.intermediate_res["A0"] = X
        
        for i in range(1, L):
            if i == (L - 1):
                Z, A = self._forward_propagate(A, self.params[f"W{i}"], self.params[f"b{i}"])
            else:
                Z, A = self._forward_propagate(A, self.params[f"W{i}"], self.params[f"b{i}"], activation)
            
            self.intermediate_res[f"Z{i}"] = Z
            self.intermediate_res[f"A{i}"] = A
            
        return A

    def _backward_propagate(self, dA, Z, W, A_prev, activation = "sigmoid"):
        m = A_prev.shape[1]
        
        dZ = self._activation_derivative(Z, dA, activation) 
        dW = ((1/m) * dZ) @ A_prev.T
        db = (1/m) * np.sum(dZ , axis=1, keepdims=True)
        dA_prev = W.T @ dZ
        
        return dA_prev, dW, db

    def _backward_propagate_all(self, Y, Yhat, layer_neurons, activation):
        L = len(layer_neurons)
        
        dA = self._cost_derivative(Y, Yhat)

        for i in range(L - 1, 0, -1):
            if i == (L - 1):
                dA, dW, db = self._backward_propagate(dA, self.intermediate_res[f"Z{i}"], self.params[f"W{i}"], self.intermediate_res[f"A{i - 1}"], activation=activation)
            else:
                dA, dW, db = self._backward_propagate(dA, self.intermediate_res[f"Z{i}"], self.params[f"W{i}"], self.intermediate_res[f"A{i - 1}"], activation=activation)
                    
            self.params[f"dA{i}"] = dA
            self.params[f"dW{i}"] = dW
            self.params[f"db{i}"] = db

    def _update_parameters(self, learning_rate, layer_neurons):
        L = len(layer_neurons)
        
        for i in range(1, L):
            self.params[f"W{i}"] = self.params[f"W{i}"] - (learning_rate * self.params[f"dW{i}"])
            self.params[f"b{i}"] = self.params[f"b{i}"] - (learning_rate * self.params[f"db{i}"])

    def fit(self, X, Y, learning_rate = 0.1, num_iterations = 1000, layer_hidden_neurons = [100, 50], activation = "sigmoid", print_cost = True):
        
        layer_neurons = [X.shape[0]] + layer_hidden_neurons + [Y.shape[0]]
        self.layer_neurons = layer_neurons
        self._initialize_parameters(layer_neurons)
        
        for i in range(num_iterations + 1):
            A = self._forward_propagate_all(X, layer_neurons, activation=activation)
            
            if i % (num_iterations / 10) == 0 and print_cost:
                print(f"Costo en la iteración {i}: {self._calculate_cost(Y, A)}")
                
            self._backward_propagate_all(Y, A, layer_neurons, activation=activation)
            
            self._update_parameters(learning_rate, layer_neurons)

    def predict(self, X, y):
        m = X.shape[1]
        p = np.zeros((1,m))
        
        A = self._forward_propagate_all(X, self.layer_neurons)

        for i in range(0, A.shape[1]):
            if A[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p

    def print_mislabeled_images(self, classes, X, y, p):
        """
        Plots images where predictions and truth were different.
        X -- dataset
        y -- true labels
        p -- predictions
        """
        a = p + y
        mislabeled_indices = np.asarray(np.where(a == 1))
        plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
        num_images = len(mislabeled_indices[0])
        for i in range(num_images):
            index = mislabeled_indices[1][i]
            
            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes