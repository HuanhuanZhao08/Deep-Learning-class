import numpy as np
import sys
import matplotlib.pyplot as plt

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    # initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self, activation, input_num, lr, weights=None):
        # print('constructor')
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        if weights is None:
            W = np.random.rand(self.input_num)
            self.weights = np.append(W, 1.0)
        else:
            self.weights = weights

    # This method returns the activation of the net
    def activate(self, net):
        # print('activate')
        if self.activation == 1:
            return 1.0 / (1.0 + np.exp(-net))
        elif self.activation == 0:
            return net

    # Calculate the output of the neuron should save the input and output for back-propagation.
    def calculate(self, input):
        # print('calculate')
        self.input = input
        output = self.activate(np.dot(self.input, self.weights[:-1]) + self.weights[-1])
        self.output = output
        return self.output

    # This method returns the derivative of the activation function with respect to the net
    def activationderivative(self):
        # print('activationderivative')
        if self.activation == 0:
            return self.output
        elif self.activation == 1:
            return self.output * (1 - self.output)

    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        # print('calcpartialderivative')
        W = self.weights[:-1]
        self.delta = np.sum(wtimesdelta) * self.activationderivative()
        self.derivW = self.delta * self.input
        self.derivB = self.delta
        return self.delta * W

    # Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        # print('updateweight')
        self.weights[:-1] = self.weights[:-1] - self.lr * self.derivW
        self.weights[-1] = self.weights[-1] - self.lr * self.derivB
        return self.weights


# A fully connected layer
class FullyConnected:
    # initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
        # print('constructor')
        self.numOfNeurons = numOfNeurons
        self.input_num = input_num
        self.activation = activation
        self.lr = lr
        self.neurons = []

        self.neurons = []
        self.weights = []
        if weights is None:
            for i in range(self.numOfNeurons):
                neuron = Neuron(self.activation, self.input_num, self.lr, None)
                self.neurons.append(neuron)
                self.weights.append(neuron.weights)
        else:
            self.weights = weights
            for i in range(self.numOfNeurons):
                neuron = Neuron(self.activation, self.input_num, self.lr, weights[i])
                self.neurons.append(neuron)

    # calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)
    def calculate(self, input):
        # print('calculate')
        self.input = input
        self.output = np.zeros(self.numOfNeurons)
        for i in range(self.numOfNeurons):
            self.output[i] = self.neurons[i].calculate(self.input)
        return self.output

    # given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.
    def calcwdeltas(self, wtimesdelta):
        # print('calcwdeltas')
        w_delta = []
        for i in range(self.numOfNeurons):
            w_delta.append(self.neurons[i].calcpartialderivative(wtimesdelta[:, i]))
            self.weights[i] = self.neurons[i].updateweight()
        w_delta = np.array(w_delta)
        return w_delta


# An entire neural network
class NeuralNetwork:
    # initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
        # print('constructor')
        self.numOfLayers = numOfLayers
        self.numOfNeurons = numOfNeurons
        self.inputSize = inputSize
        self.activation = activation
        self.loss = loss
        self.lr = lr

        self.layers = []
        self.weights = []
        input_num = self.inputSize
        if weights is None:
            for i in range(self.numOfLayers):
                fc = FullyConnected(self.numOfNeurons[i], self.activation[i], input_num, self.lr, None)
                input_num = self.numOfNeurons[i]
                self.layers.append(fc)

                self.weights.append(fc.weights)
        else:
            self.weights = weights
            for i in range(self.numOfLayers):
                fc = FullyConnected(self.numOfNeurons[i], self.activation[i], input_num, self.lr, weights[i])
                input_num = self.numOfNeurons[i]
                self.layers.append(fc)

    # Given an input, calculate the output (using the layers calculate() method)
    def calculate(self, input):
        # print('constructor')
        tmp_input = input
        for i in range(self.numOfLayers):
            output = self.layers[i].calculate(tmp_input)
            tmp_input = output
        self.output = output
        return self.output

    # Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self, yp, y):
        # print('calculate')
        N = len(y)
        if self.loss == 0:
            error = (1 / N) * np.sum(np.square(yp - y))
        if self.loss == 1:
            error = -(1 / N) * np.sum(np.multiply(y, np.log(yp)) + np.multiply((1 - y), np.log(1 - yp)))
        return error

    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)
    def lossderiv(self, yp, y):
        # print('lossderiv')
        if self.loss == 0:
            deriv = (yp - y)
        if self.loss == 1:
            deriv = (-y) / yp + (1 - y) / (1 - yp)
        return deriv

    # Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values
    def train(self, x, y, epochs=1):

        losses = []
        for j in range(epochs):
            yp = self.calculate(x)
            deriv = self.lossderiv(yp, y)
            deriv = np.reshape(deriv, (-1, len(deriv)))
            # print("loss", self.calculateloss(yp,y))
            wtimesdelta = deriv

            i = self.numOfLayers - 1
            while i >= 0:
                wtimesdelta = self.layers[i].calcwdeltas(wtimesdelta)
                i -= 1
            # print(self.calculateloss(yp, y))
            # self.Loss = self.calculateloss(yp, y)
            losses.append(self.calculateloss(yp, y))
        self.losses = losses


if __name__ == "__main__":
    # print("sys.argv", sys.argv)
    #for lr in [0.05, 0.01, 0.005, 0.001]:
    lr = float(sys.argv[2])

    if len(sys.argv) < 2:
        print('a good place to test different parts of your code')

    elif sys.argv[1] == 'example':
        print('run example from class (single step)')
        w = np.array([[[.15, .2, .35], [.25, .3, .35]], [[.4, .45, .6], [.5, .55, .6]]])
        x = np.array([0.05, 0.1])
        y = np.array([0.01, 0.99])
        numOfLayers = w.shape[0]
        numOfNeurons = np.array([2, 2])
        inputSize = len(x)
        activation = [1, 1]
        loss = 0
        nn = NeuralNetwork(numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=w)
        nn.train(x, y, 1)
        print("weights after first step: ")
        print(nn.weights)
        # print("input ", "target ", "prediction probability", "prediction_class")
        # print(y, nn.calculate(x))

    elif sys.argv[1] == 'and':
        print('learn and')
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])

        numOfLayers = 1
        numOfNeurons = np.array([1])
        inputSize = 2
        activation = [1]
        loss = 1
        nn = NeuralNetwork(numOfLayers, numOfNeurons, inputSize, activation, loss, lr)
        indexes = np.random.randint(0, 4, 100)
        for i in indexes:
            nn.train(X[i], y[i], 2000)
        print("input ", "target ", "prediction probability", "prediction_class")
        for i in range(X.shape[0]):
            print(X[i], y[i], nn.calculate(X[i]), [0 if nn.calculate(X[i]) < 0.5 else 1])

    elif sys.argv[1] == 'xor':
        print('learn xor')

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        # model 1: with a single neuron
        print("train the data with a single perceptron")
        numOfLayers = 1
        numOfNeurons = np.array([1])
        inputSize = 2
        activation = [1]
        loss = 1
        nn = NeuralNetwork(numOfLayers, numOfNeurons, inputSize, activation, loss, lr)
        indexes = np.random.randint(0, 4, 1000)
        for i in indexes:
            nn.train(X[i], y[i], 2000)
        print("input ", "target ", "prediction_probability", "prediction_class")
        for i in range(X.shape[0]):
            print(X[i], y[i], nn.calculate(X[i]), [0 if nn.calculate(X[i]) < 0.5 else 1])
        # model 2: with a model contains hidden layer
        print("train the data with a model contain one hidden layer")
        numOfLayers = 2
        numOfNeurons = np.array([4, 1])
        inputSize = 2
        activation = [1, 1]
        loss = 1
        nn = NeuralNetwork(numOfLayers, numOfNeurons, inputSize, activation, loss, lr)
        indexes = np.random.randint(0, 4, 500)
        for i in indexes:
            nn.train(X[i], y[i], 10000)
        print("input ", "target ", "prediction probability", "prediction_class")
        for i in range(X.shape[0]):
            print(X[i], y[i], nn.calculate(X[i]), [0 if nn.calculate(X[i]) < 0.5 else 1])

        # plt.plot(nn.losses, alpha=0.5, label=lr)
        # plt.legend(loc='upper right')
        # plt.title("Loss dropping with different learning rates")
        # plt.xlabel("training epochs")
        # plt.ylabel("loss dropping")
        # plt.savefig('and.png')
