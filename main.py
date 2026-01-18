import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import deepcopy

def relu(matrix):
    return np.maximum(matrix, 0)

def softmax(matrix):
    matrix = matrix - np.max(matrix)
    e_powers = np.exp(matrix)
    sums = np.sum(e_powers,axis=1, keepdims=True)
    return e_powers/sums

def one_hot_encoding(y, num_classes=10):
    matrix = np.zeros((y.size,num_classes))
    matrix[np.arange(y.size),y.astype(int)] = 1.0
    return matrix
        
    

#constants
N_INPUTS= 784
N_HIDDEN = 128
N_OUTPUTS = 10

np.random.seed(1)

# data import
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

y_train = train_data["label"] # labels

x_train = train_data.drop("label", axis=1) # features
x_train /=255 # data normalization to [0,1]

y_test = test_data["label"]
x_test = test_data.drop("label", axis=1)
x_test /=255


x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=1)

x_train=x_train.values
x_test = x_test.values
x_val = x_val.values
# converting vectors to matrices with correct outputs

y_train = one_hot_encoding(y_train)
y_val = one_hot_encoding(y_val)
y_test = one_hot_encoding(y_test)


biases_hidden = np.zeros((1,N_HIDDEN))
biases_output = np.zeros((1,N_OUTPUTS))

weights_input = np.random.randn(N_INPUTS, N_HIDDEN)
weights_hidden = np.random.randn(N_HIDDEN,N_OUTPUTS)

#hyperparameters

learning_rate = 0.1
epochs = 40
batch_size = 64

num_samples = x_train.shape[0]

while learning_rate<=3:
    for epoch in range(epochs):
        for i in range(0,num_samples,batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # forward propagation

            input_hidden = np.dot(x_batch, weights_input) + biases_hidden #64,128
            output_hidden = relu(input_hidden)

            input_final = np.dot(output_hidden, weights_hidden) + biases_output
            output_final = softmax(input_final)

            # loss function

            output_error = output_final - y_batch # y_batch is correct outputs, matrix(64,10)

            grad_weights_hidden = np.dot(output_hidden.T, output_error)/batch_size #matrix (128,10)
            grad_biases_output = np.sum(output_error,axis=0,keepdims=True)/batch_size # matrix 1,10

            hidden_error = np.dot(output_error, weights_hidden.T)
            final_hidden_error = hidden_error * (input_hidden > 0)

            grad_weights_input = np.dot(x_batch.T, final_hidden_error)/batch_size
            grad_biases_hidden = np.sum(final_hidden_error, axis=0,keepdims=True)/batch_size

            biases_hidden -= grad_biases_hidden*learning_rate
            biases_output -= grad_biases_output*learning_rate
            weights_input -= grad_weights_input*learning_rate
            weights_hidden -= grad_weights_hidden*learning_rate

        input_hidden = np.dot(x_val, weights_input) + biases_hidden #64,128
        output_hidden = relu(input_hidden)  
        input_final = np.dot(output_hidden, weights_hidden) + biases_output
        output_val = softmax(input_final)
        predictions = np.argmax(output_val,axis=1)
        true_answers = np.argmax(y_val,axis=1)
        correct_count = np.sum((predictions == true_answers))
        incorrect_count = predictions.size - correct_count
        accuracy = correct_count/predictions.size

        #print(f"Epoka {epoch}, poprawnych: {correct_count}, niepoprawnych: {incorrect_count}, poprawność: {accuracy:.4f}")


    input_hidden = np.dot(x_test, weights_input) + biases_hidden #64,128
    output_hidden = relu(input_hidden)  
    input_final = np.dot(output_hidden, weights_hidden) + biases_output
    output_test = softmax(input_final)
    predictions = np.argmax(output_test,axis=1)
    true_answers = np.argmax(y_test,axis=1)
    correct_count = np.sum((predictions == true_answers))
    incorrect_count = predictions.size - correct_count
    accuracy = correct_count/predictions.size
    print(f"Wynik końcowy, lr: {learning_rate}, poprawnych: {correct_count}, niepoprawnych: {incorrect_count}, poprawność: {accuracy:.4f}")
    learning_rate+=0.2