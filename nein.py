import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Set working directory and load data
os.chdir('/Users/bugraaydemir/Desktop/ANN/iris_dataset')

# Load and check dataset using pandas
iris_dataset = pd.read_csv('iris.csv')
print(iris_dataset.head())
print(iris_dataset[50:56])
print(iris_dataset.tail())

# Visualize raw dataset
scatter_matrix(iris_dataset, alpha=0.5, figsize=(20, 20))
plt.show()
iris_dataset.plot(subplots=True, figsize=(10, 10), sharex=False, sharey=False)
plt.show()

# Data cleaning process
print('[INFO] create numeric classes for species (0,1,2) ...')
iris_dataset.loc[iris_dataset['species'] == 'setosa', 'species'] = 0
iris_dataset.loc[iris_dataset['species'] == 'versicolor', 'species'] = 1
iris_dataset.loc[iris_dataset['species'] == 'virginica', 'species'] = 2

iris_label = np.array(iris_dataset['species'])
iris_data = np.array(iris_dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

# Splitting dataset into training, validation, and testing
random.seed(123)

def separate_data():
    A = iris_dataset[0:40]
    tA = iris_dataset[40:50]
    B = iris_dataset[50:90]
    tB = iris_dataset[90:100]
    C = iris_dataset[100:140]
    tC = iris_dataset[140:150]
    train = np.concatenate((A, B, C))
    test = np.concatenate((tA, tB, tC))
    return train, test

print('[INFO] separate data to train data and test data')
iris_dataset = np.column_stack((iris_data, iris_label.T)) # Join X and Y
iris_dataset = list(iris_dataset)
random.shuffle(iris_dataset)
Filetrain, Filetest = separate_data()

# Further split the training data into training and validation sets
def train_validation_split(data, validation_split=0.2):
    random.shuffle(data)
    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    validation_data = data[split_idx:]
    return train_data, validation_data

Filetrain, Filevalidation = train_validation_split(Filetrain)

train_X = np.array([i[:4] for i in Filetrain]).astype('float')
train_y = np.array([i[4] for i in Filetrain]).astype('float')
validation_X = np.array([i[:4] for i in Filevalidation]).astype('float')
validation_y = np.array([i[4] for i in Filevalidation]).astype('float')
test_X = np.array([i[:4] for i in Filetest]).astype('float')
test_y = np.array([i[4] for i in Filetest]).astype('float')

print('train data shape: ', train_X.shape)
print('validation data shape: ', validation_X.shape)
print('test data shape: ', test_X.shape)
print('train label shape: ', train_y.shape)
print('validation label shape: ', validation_y.shape)
print('test label shape: ', test_y.shape)

# Define MLP Class Object
class MultiLayerPerceptron: 
    def __init__(self, params=None):
        # Default MLP Layer if not specified
        if params is None:
            self.inputLayer = 4                        # Input Layer
            self.hiddenLayer = 5                       # Hidden Layer
            self.outputLayer = 3                       # Output Layer
            self.learningRate = 0.005                  # Learning rate
            self.max_epochs = 600                      # Epochs
            self.BiasHiddenValue = -1                  # Bias HiddenLayer
            self.BiasOutputValue = -1                  # Bias OutputLayer
            self.activation = self.activation['sigmoid'] # Activation function
            self.deriv = self.deriv['sigmoid']
        else:
            self.inputLayer = params['InputLayer']
            self.hiddenLayer = params['HiddenLayer']
            self.OutputLayer = params['OutputLayer']
            self.learningRate = params['LearningRate']
            self.max_epochs = params['Epochs']
            self.BiasHiddenValue = params['BiasHiddenValue']
            self.BiasOutputValue = params['BiasOutputValue']
            self.activation = self.activation[params['ActivationFunction']]
            self.deriv = self.deriv[params['ActivationFunction']]
        
        # Initialize Weight and Bias value
        self.WEIGHT_hidden = self.starting_weights(self.hiddenLayer, self.inputLayer)
        self.WEIGHT_output = self.starting_weights(self.OutputLayer, self.hiddenLayer)
        self.BIAS_hidden = np.array([self.BiasHiddenValue for i in range(self.hiddenLayer)])
        self.BIAS_output = np.array([self.BiasOutputValue for i in range(self.OutputLayer)])
        self.classes_number = 3 
    
    def starting_weights(self, x, y):
        return [[2 * random.random() - 1 for i in range(x)] for j in range(y)]

    # Define activation and derivation function based on Mathematical formula
    activation = {
        'sigmoid': (lambda x: 1/(1 + np.exp(-x * 1.0))),
        'tanh': (lambda x: np.tanh(x)),
        'Relu': (lambda x: x * (x > 0)),
    }

    deriv = {
        'sigmoid': (lambda x: x * (1 - x)),
        'tanh': (lambda x: 1 - x ** 2),
        'Relu': (lambda x: 1 * (x > 0))
    }
    
    # Define Backpropagation process algorithm
    def Backpropagation_Algorithm(self, x):
        DELTA_output = []
        
        # Stage 1 - Error: OutputLayer
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = (-1) * (ERROR_output) * self.deriv(self.OUTPUT_L2)
        
        # Stage 2 - Update weights OutputLayer and HiddenLayer
        for i in range(self.hiddenLayer):
            for j in range(self.OutputLayer):
                self.WEIGHT_output[i][j] -= self.learningRate * (DELTA_output[j] * self.OUTPUT_L1[i])
                self.BIAS_output[j] -= self.learningRate * DELTA_output[j]
      
        # Stage 3 - Error: HiddenLayer
        delta_hidden = np.matmul(self.WEIGHT_output, DELTA_output) * self.deriv(self.OUTPUT_L1)
 
        # Stage 4 - Update weights HiddenLayer and InputLayer(x)
        for i in range(self.OutputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= self.learningRate * (delta_hidden[j] * x[i])
                self.BIAS_hidden[j] -= self.learningRate * delta_hidden[j]
    
    # Function for plotting error value for each epoch
    def show_err_graphic(self, v_error, v_val_error, v_epoch):
        plt.figure(figsize=(9, 4))
        plt.plot(v_epoch, v_error, label="Training Error")
        plt.plot(v_epoch, v_val_error, label="Validation Error", linestyle='--')
        plt.xlabel("Number of Epochs")
        plt.ylabel("Squared error (MSE)")
        plt.title("Error Minimization")
        plt.legend()
        plt.show()
    
    # Define predict function for prediction test data
    def predict(self, X, y):
        my_predictions = []
        
        # Just doing Forward Propagation
        forward = np.matmul(X, self.WEIGHT_hidden) + self.BIAS_hidden
        forward = np.matmul(forward, self.WEIGHT_output) + self.BIAS_output
                                 
        for i in forward:
            my_predictions.append(max(enumerate(i), key=lambda x: x[1])[0])
            
        # Print predicted value    
        print(" Number of Sample  | Class |  Output  | Hoped Output")   
        for i in range(len(my_predictions)):
            if my_predictions[i] == 0: 
                print(f"id:{i}    | Iris-Setosa  |  Output: {my_predictions[i]} | Hoped Output: {y[i]}  ")
            elif my_predictions[i] == 1: 
                print(f"id:{i}    | Iris-Versicolour    |  Output: {my_predictions[i]} | Hoped Output: {y[i]} ")
            elif my_predictions[i] == 2: 
                print(f"id:{i}    | Iris-Iris-Virginica   |  Output: {my_predictions[i]} | Hoped Output: {y[i]} ")
                
        return my_predictions

    # Define fit function for training process with train data
    def fit(self, X, y, validation_X, validation_y):  
        count_epoch = 1
        total_error = 0
        validation_error = 0
        n = len(X)
        epoch_array = []
        error_array = []
        validation_error_array = []
        W0 = []
        W1 = []
        while count_epoch <= self.max_epochs:
            for idx, inputs in enumerate(X): 
                self.output = np.zeros(self.classes_number)
                
                # Stage 1 - (Forward Propagation)
                self.OUTPUT_L1 = self.activation(np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden.T)
                self.OUTPUT_L2 = self.activation(np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output.T)
                
                # Stage 2 - One-Hot-Encoding
                if y[idx] == 0: 
                    self.output = np.array([1, 0, 0]) # Class1 {1,0,0}
                elif y[idx] == 1:
                    self.output = np.array([0, 1, 0]) # Class2 {0,1,0}
                elif y[idx] == 2:
                    self.output = np.array([0, 0, 1]) # Class3 {0,0,1}
                
                square_error = 0
                for i in range(self.OutputLayer):
                    erro = (self.output[i] - self.OUTPUT_L2[i]) ** 2
                    square_error = square_error + (0.05 * erro)
                    total_error = total_error + square_error
         
                # Backpropagation: Update Weights
                self.Backpropagation_Algorithm(inputs)
            
            # Compute validation error
            validation_error = 0
            for idx, inputs in enumerate(validation_X):
                self.output = np.zeros(self.classes_number)
                
                # Forward propagation for validation data
                self.OUTPUT_L1 = self.activation(np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden.T)
                self.OUTPUT_L2 = self.activation(np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output.T)
                
                if validation_y[idx] == 0: 
                    self.output = np.array([1, 0, 0]) # Class1 {1,0,0}
                elif validation_y[idx] == 1:
                    self.output = np.array([0, 1, 0]) # Class2 {0,1,0}
                elif validation_y[idx] == 2:
                    self.output = np.array([0, 0, 1]) # Class3 {0,0,1}
                
                square_error = 0
                for i in range(self.OutputLayer):
                    erro = (self.output[i] - self.OUTPUT_L2[i]) ** 2
                    square_error = square_error + (0.05 * erro)
                    validation_error = validation_error + square_error
            
            total_error = total_error / n
            validation_error = validation_error / len(validation_X)
            
            # Print error value for each epoch
            if count_epoch % 50 == 0 or count_epoch == 1:
                print("Epoch ", count_epoch, "- Training Error: ", total_error, "- Validation Error: ", validation_error)
                error_array.append(total_error)
                validation_error_array.append(validation_error)
                epoch_array.append(count_epoch)
                
            W0.append(self.WEIGHT_hidden)
            W1.append(self.WEIGHT_output)
             
            count_epoch += 1
            
        self.show_err_graphic(error_array, validation_error_array, epoch_array)
        
        # Print weight Hidden layer acquired during training
        print('')
        print('weight value of Hidden layer acquired during training: ')
        print(W0[0])
        
        # Plot weight Output layer acquired during training
        print('')
        print('weight value of Output layer acquired during training: ')
        print(W1[0])

# Train and evaluate MLP Model using different activation functions
activations = ['sigmoid', 'tanh', 'Relu']
for activation in activations:
    print(f"\nEvaluating {activation} activation function")
    dictionary = {'InputLayer': 4, 'HiddenLayer': 5, 'OutputLayer': 3, 'Epochs': 700, 'LearningRate': 0.005, 'BiasHiddenValue': -1, 'BiasOutputValue': -1, 'ActivationFunction': activation}
    Perceptron = MultiLayerPerceptron(dictionary)
    Perceptron.fit(train_X, train_y, validation_X, validation_y)

    # Predict Test Data using MLP Model
    pred = Perceptron.predict(test_X, test_y)
    pred = np.array(pred)
    true = test_y.astype('int')

    def compute_confusion_matrix(true, pred):
        K = len(np.unique(true)) # Number of classes 
        result = np.zeros((K, K))
        for i in range(len(true)):
            result[true[i]][pred[i]] += 1
        return result

    conf_matrix = compute_confusion_matrix(true, pred)
    print('Confusion matrix result: ')
    print(conf_matrix)

    if activation == 'sigmoid':
        # Classification Report
        classes = ['setosa', 'versicolor', 'virginica']
        def accuracy_average(confusion_matrix):
            diagonal_sum = confusion_matrix.trace()
            sum_of_all_elements = confusion_matrix.sum()
            return diagonal_sum / sum_of_all_elements 

        def precision(label, confusion_matrix):
            col = confusion_matrix[:, label]
            return confusion_matrix[label, label] / col.sum()
        
        def recall(label, confusion_matrix):
            row = confusion_matrix[label, :]
            return confusion_matrix[label, label] / row.sum()

        def f1_score(label, confusion_matrix):
            num = precision(label, confusion_matrix) * recall(label, confusion_matrix)
            denum = precision(label, confusion_matrix) + recall(label, confusion_matrix)
            return 2 * (num / denum)

        def precision_macro_average(confusion_matrix):
            rows, columns = confusion_matrix.shape
            sum_of_precisions = 0
            for label in range(rows):
                sum_of_precisions += precision(label, confusion_matrix)
            return sum_of_precisions / rows

        def recall_macro_average(confusion_matrix):
            rows, columns = confusion_matrix.shape
            sum_of_recalls = 0
            for label in range(columns):
                sum_of_recalls += recall(label, confusion_matrix)
            return sum_of_recalls / columns

        def f1_score_average(confusion_matrix):
            num = precision_macro_average(confusion_matrix) * recall_macro_average(confusion_matrix)
            denum = precision_macro_average(confusion_matrix) + recall_macro_average(confusion_matrix)
            return 2 * (num / denum)

        print("label      precision  recall  f1_score")
        for index in range(len(classes)):
            print(f"{classes[index]} {precision(index, conf_matrix):9.3f} {recall(index, conf_matrix):6.3f}  {f1_score(index, conf_matrix):6.3f}")

        print()
        print('Average accuracy:  ', accuracy_average(conf_matrix))
        print('Average precision: ', precision_macro_average(conf_matrix))
        print('Average recall:    ', recall_macro_average(conf_matrix))
        print('Average F1 score:  ', f1_score_average(conf_matrix))

        # Plotting hits and faults
        hits = accuracy_average(conf_matrix) * 100
        faults = 100 - hits

        graph_hits = []
        graph_hits.append(hits)
        graph_hits.append(faults)

        print("Porcents: %.2f%% hits and %.2f%% faults" % (hits, faults))
        print("Total samples of test", len(test_y))

        # Calculate the number of samples for each class
        n_set = sum(test_y == 0)
        n_vers = sum(test_y == 1)
        n_virg = sum(test_y == 2)

        print("*Iris-Setosa:", n_set, "samples")
        print("*Iris-Versicolour:", n_vers, "samples")
        print("*Iris-Virginica:", n_virg, "samples")

        labels = 'Hits', 'Faults'
        explode = (0, 0.14)

        fig1, ax1 = plt.subplots()
        ax1.pie(graph_hits, explode=explode, colors=['green', 'red'], labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')
        plt.title("Hits vs Faults")
        plt.show()

        # Accuracy for each class
        score_set = sum((pred == 0) & (test_y == 0))
        score_vers = sum((pred == 1) & (test_y == 1))
        score_virg = sum((pred == 2) & (test_y == 2))

        acc_set = (score_set / n_set) * 100
        acc_vers = (score_vers / n_vers) * 100
        acc_virg = (score_virg / n_virg) * 100

        print("- Accuracy Iris-Setosa:", "%.2f" % acc_set, "%")
        print("- Accuracy Iris-Versicolour:", "%.2f" % acc_vers, "%")
        print("- Accuracy Iris-Virginica:", "%.2f" % acc_virg, "%")

        names = ["Setosa", "Versicolour", "Virginica"]
        x1 = [2.0, 4.0, 6.0]
        fig, ax = plt.subplots()
        plt.bar(x1[0], acc_set, color='orange', label='Iris-Setosa')
        plt.bar(x1[1], acc_vers, color='green', label='Iris-Versicolour')
        plt.bar(x1[2], acc_virg, color='purple', label='Iris-Virginica')
        plt.ylabel('Scores %')
        plt.xticks(x1, names)
        plt.title('Scores by iris flowers - Multilayer Perceptron')
        plt.legend()
        plt.show()

# Activation functions and performance visualization
def show_test():
    ep1 = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    h_5 = [0, 60, 70, 70, 83.3, 93.3, 96.7, 86.7, 86.7, 76.7, 73.3, 66.7, 66.7]
    h_4 = [0, 40, 70, 63.3, 66.7, 70, 70, 70, 70, 66.7, 66.7, 43.3, 33.3]
    h_3 = [0, 46.7, 76.7, 80, 76.7, 76.7, 76.6, 73.3, 73.3, 73.3, 73.3, 76.7, 76.7]
    plt.figure(figsize=(10, 4))
    l1, = plt.plot(ep1, h_3, "--", color='b', label="node-3", marker=11)
    l2, = plt.plot(ep1, h_4, "--", color='g', label="node-4", marker=8)
    l3, = plt.plot(ep1, h_5, "--", color='r', label="node-5", marker=5)
    plt.legend(handles=[l1, l2, l3], loc=1)
    plt.xlabel("number of Epochs")
    plt.ylabel("% Hits")
    plt.title("Number of Hidden Layers - Performance")
    
    ep2 = [0, 100, 200, 300, 400, 500, 600, 700]
    tanh = [0.18, 0.027, 0.025, 0.022, 0.0068, 0.0060, 0.0057, 0.00561]
    sigm = [0.185, 0.0897, 0.060, 0.0396, 0.0343, 0.0314, 0.0296, 0.0281]
    Relu = [0.185, 0.05141, 0.05130, 0.05127, 0.05124, 0.05123, 0.05122, 0.05121]
    plt.figure(figsize=(10, 4))
    l1, = plt.plot(ep2, tanh, "--", color='b', label="Hyperbolic Tangent", marker=11)
    l2, = plt.plot(ep2, sigm, "--", color='g', label="Sigmoid", marker=8)
    l3, = plt.plot(ep2, Relu, "--", color='r', label="ReLU", marker=5)
    plt.legend(handles=[l1, l2, l3], loc=1)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Activation Functions - Performance")
    
    fig, ax = plt.subplots()
    names = ["Hyperbolic Tangent", "Sigmoid", "ReLU"]
    x1 = [2.0, 4.0, 6.0]
    plt.bar(x1[0], 53.4, 0.4, color='b')
    plt.bar(x1[1], 96.7, 0.4, color='g')
    plt.bar(x1[2], 33.2, 0.4, color='r')
    plt.xticks(x1, names)
    plt.ylabel('% Hits')
    plt.title('Hits - Activation Functions')
    plt.show()

# Call the show_test function to display the performance graphs
show_test()
