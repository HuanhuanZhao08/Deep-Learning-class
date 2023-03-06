import numpy as np
import sys
from numpy import unravel_index
from numpy import newaxis


# A class which represents a single neuron
class Neuron:
    # initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self, activation, input_num, lr, weights=None):
        #         print('constructor')
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
        #         print('activate')
        if self.activation == 1:
            return 1.0 / (1.0 + np.exp(-net))
        elif self.activation == 0:
            return net

    # Calculate the output of the neuron should save the input and output for back-propagation.
    def calculate(self, input):
        #         print('calculate')
        self.input = input
        output = self.activate(np.dot(self.input, self.weights[:-1]) + self.weights[-1])
        self.output = output
        return self.output

    # This method returns the derivative of the activation function with respect to the net
    def activationderivative(self):
        #         print('activationderivative')
        if self.activation == 0:
            return self.output
        elif self.activation == 1:
            return self.output * (1 - self.output)

    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        #         print('calcpartialderivative')
        W = self.weights[:-1]
        self.delta = np.sum(wtimesdelta) * self.activationderivative()
        self.derivW = self.delta * self.input
        self.derivB = self.delta
        return self.delta * W

    # Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        #         print('updateweight')
        self.weights[:-1] = self.weights[:-1] - self.lr * self.derivW
        self.weights[-1] = self.weights[-1] - self.lr * self.derivB
        return self.weights


# A fully connected layer
class FullyConnected:
    # initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None):
        # print('constructor')
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.lr = lr
        self.neurons = []
        self.weights = weights
        if self.weights is None:
            for i in range(self.numOfNeurons):
                neuron = Neuron(self.activation, input_num, self.lr, None)
                self.neurons.append(neuron)
                self.weights.append(neuron.weights)
        else:
            for i in range(self.numOfNeurons):
                neuron = Neuron(self.activation, input_num, self.lr, weights[i])
                self.neurons.append(neuron)

    # calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)
    def calculate(self, input):
        #         print('fc calculate')
        self.input = input
        self.output = np.zeros(self.numOfNeurons)
        for i in range(self.numOfNeurons):
            self.output[i] = self.neurons[i].calculate(self.input)
        return self.output

    # given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.
    def calcwdeltas(self, wtimesdelta):
        w_delta = []
        for i in range(self.numOfNeurons):
            w_delta.append(self.neurons[i].calcpartialderivative(wtimesdelta[:, i]))
            self.weights[i] = self.neurons[i].updateweight()
        w_delta = np.array(w_delta)
        return w_delta


# An entire neural network
class NeuralNetwork:
    # initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, inputSize, loss, lr):
        # print('constructor')
        #         self.numOfLayers = []
        #         self.numOfNeurons = []
        self.inputSize = inputSize
        self.loss = loss
        self.lr = lr
        self.layer_num = 0
        self.layers = []
        self.output = None

    def addLayer(self, category, parameters):
        # parameters is a dictionary contain all the parameters needed current layer
        if self.layer_num == 0:
            input_dim = self.inputSize
        else:
            input_dim = self.layers[-1].output.shape
        if category == 'conv':
            self.layers.append(ConvolutionalLayer(input_dim, parameters['filter_size'],
                                                  parameters['num_filter'], parameters['activation'],
                                                  self.lr, parameters['weights']))
        elif category == 'maxpool':
            # kernelSize, inputDimension
            self.layers.append(MaxPoolingLayer(parameters['filter_size'], input_dim))
        elif category == 'flatten':
            self.layers.append(FlattenLayer(input_dim))
        elif category == 'dense':
            self.layers.append(FullyConnected(parameters['numOfNeurons'], parameters['activation'], input_dim,
                                              self.lr, parameters['weights']))
        self.layer_num += 1

    # Given an input, calculate the output (using the layers calculate() method)
    def calculate(self, input):
        tmp_input = input
        for i in range(len(self.layers)):
            output = self.layers[i].calculate(tmp_input)
            tmp_input = output
        self.output = output
        return self.output

    # Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self, yp, y):
        # print('calculate')
        N = len(y)
        if self.loss == 0:
            error = np.sum(np.square(yp - y))
        if self.loss == 1:
            error = -(1 / N) * np.sum(np.multiply(y, np.log(yp)) + np.multiply((1 - y), np.log(1 - yp)))
        return error

    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)
    def lossderiv(self, yp, y):
        # print('lossderiv')
        if self.loss == 0:
            deriv = 2 * (yp - y)
        if self.loss == 1:
            deriv = (-y) / yp + (1 - y) / (1 - yp)
        return deriv

    def train(self, x, y, epochs=1):

        losses = []
        for j in range(epochs):
            yp = self.calculate(x)
            deriv = self.lossderiv(yp, y)
            deriv = np.reshape(deriv, (-1, len(deriv)))
            wtimesdelta = deriv

            i = len(self.layers) - 1
            while i >= 0:
                wtimesdelta = self.layers[i].calcwdeltas(wtimesdelta)
                i -= 1
            losses.append(self.calculateloss(yp, y))
        self.losses = losses

        return self.losses


class ConvolutionalLayer:
    def __init__(self, input_shape, filter_size, num_filter, activation, lr, weights=None):
        self.input_shape = input_shape  # 3d example(7,7,2) (rows,columns, input_channels)
        self.filter_size = filter_size  # 2d example (3,3)
        self.num_filter = num_filter  # number of fiters
        self.lr = lr

        self.input_channel = self.input_shape[-1]
        rows = self.input_shape[0] - self.filter_size + 1
        columns = self.input_shape[1] - self.filter_size + 1
        self.output_shape = (rows, columns, self.num_filter)
        self.output = np.zeros(self.output_shape)
        self.neurons = np.full([self.output_shape[0], self.output_shape[1], self.num_filter], None)
        self.weights = weights

        if self.weights is None:
            ''' w = [(3,3,1,2),[b1.b2]] '''
            self.weights = [np.random.rand(self.filter_size, self.filter_size, self.input_channel, self.num_filter),
                            np.array([np.random.rand(1), np.random.rand(1)])]

        for m in range(self.num_filter):
            weight = self.weights[0][:, :, :, m]
            weight = weight.flatten()
            weight = np.append(weight, weights[1][m])
            for j in range(rows):
                for k in range(columns):
                    self.neurons[j, k, m] = Neuron(activation, filter_size * filter_size * self.input_channel,
                                                   lr, weight)

    def calculate(self, input):

        for m in range(self.num_filter):

            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    x_start = i
                    y_start = j
                    x_end = i + self.filter_size
                    y_end = j + self.filter_size
                    cur_input = input[x_start:x_end, y_start:y_end, :]
                    cur_input = cur_input.flatten()
                    #                     print(self.neurons[i,j,m].weights)
                    self.output[i, j, m] = self.neurons[i, j, m].calculate(cur_input)
        return self.output  # if there are two filters, the output shape = (rows,columns,num_filters)

    def calcwdeltas(self, wtimesdelta):

        w_delta = np.zeros(self.input_shape)
        for n in range(self.num_filter):
            derivB = 0
            #             for m in range(self.input_channel):
            derivW = np.zeros((self.filter_size, self.filter_size, self.input_channel))
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    cur_w_delta = self.neurons[i, j, n].calcpartialderivative(wtimesdelta[i, j, n])
                    cur_w_delta = cur_w_delta.reshape((self.filter_size, self.filter_size, self.input_channel))
                    x_start = i
                    y_start = j
                    x_end = i + self.filter_size
                    y_end = j + self.filter_size
                    w_delta[x_start:x_end, y_start:y_end, :] = w_delta[x_start:x_end, y_start:y_end, :] + cur_w_delta

                    #                     print(self.neurons[i,j,n].derivW)
                    #                     print(self.neurons[i,j,n].derivW.reshape(derivW.shape)[:,:,0])
                    #                     print('******',self.neurons[i,j,n].derivW)
                    derivW = derivW + self.neurons[i, j, n].derivW.reshape(derivW.shape)

                    derivB = derivB + self.neurons[i, j, n].derivB

            self.weights[0][:, :, :, n] = self.weights[0][:, :, :, n] - self.lr * derivW
            self.weights[1][n] = self.weights[1][n] - self.lr * derivB

        return w_delta


class MaxPoolingLayer:
    def __init__(self, filter_size, input_shape):
        self.filter_size = filter_size
        self.input_shape = input_shape  # 3d(7,7,2)
        self.stride = filter_size

        rows = int(self.input_shape[0] / self.filter_size)
        columns = int(self.input_shape[1] / self.filter_size)

        self.output_shape = (rows, columns, input_shape[2])
        self.output = np.zeros(self.output_shape)
        self.maxIndex = np.full(self.output_shape, None)

    def calculate(self, input):
        step = self.filter_size
        for m in range(self.output_shape[2]):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    x_start = i * step
                    y_start = j * step
                    x_end = (i + 1) * step
                    y_end = (j + 1) * step
                    cur_input = input[x_start:x_end, y_start:y_end, m]
                    x_index, y_index = unravel_index(cur_input.argmax(), cur_input.shape)
                    self.output[i][j][m] = cur_input[x_index, y_index]
                    self.maxIndex[i][j][m] = (x_index, y_index)
        return self.output

    def calcwdeltas(self, wtimesdelta):
        #         print('maxpool: wtimesdelta from flatten layer', wtimesdelta[:,:,0].shape, wtimesdelta[:,:,0])
        step = self.filter_size
        w_delta = np.zeros(self.input_shape)
        for m in range(self.output_shape[2]):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    index_x, index_y = self.maxIndex[i][j][m]
                    w_delta[i * step + index_x][j * step + index_y][m] = wtimesdelta[index_x][index_y][m]
        return w_delta


class FlattenLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output = np.zeros(input_shape)
        self.output = self.output.flatten()

    def calculate(self, input):
        self.output = input.flatten()
        return self.output

    def calcwdeltas(self, wtimesdelta):
        #         print('flatten: wtimesdelta from dense layer',wtimesdelta)
        self.wtimesdelta = wtimesdelta.reshape(self.input_shape)
        #         print('flatten: wtimesdelta for maxpool layer',self.wtimesdelta )
        return self.wtimesdelta


# Generate data and weights for "example2"
def generateExample2(input_size, output_weight):
    # Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(10)

    # First hidden layer, two kernels
    l1k1 = np.random.rand(3, 3)
    l1k2 = np.random.rand(3, 3)
    l1b1 = np.random.rand(1)
    l1b2 = np.random.rand(1)

    # second hidden layer, one kernel, two channels
    l2k1 = np.random.rand(3, 3, 2)
    l2b = np.random.rand(1)

    # output layer, fully connected
    l3 = np.random.rand(1, output_weight)
    l3b = np.random.rand(1)

    # input and output
    input = np.random.rand(input_size, input_size)
    output = np.random.rand(1)

    return l1k1, l1k2, l1b1, l1b2, l2k1, l2b, l3, l3b, input, output



if __name__ == "__main__":
    # Arguments parsing
    learning_rate = 100
    # learning_rate = float(sys.argv[1])
    example = sys.argv[1]

    if example == 'example2':
        # Set weights to desired values
        l1k1, l1k2, l1b1, l1b2, l2k1, l2b, l3, l3b, input, output = generateExample2(7, 9)

        # setting weights and bias of first layer.
        l1k1 = l1k1.reshape(3, 3, 1, 1)
        l1k2 = l1k2.reshape(3, 3, 1, 1)

        w1 = np.concatenate((l1k1, l1k2), axis=3)

        w1_b = [w1, np.array([l1b1[0], l1b2[0]])]

        # get weights and bias of second layer.

        w2 = l2k1.reshape(3, 3, 2, 1)
        w2_b = [w2, l2b]
        w3_b = np.expand_dims(np.concatenate([l3.flatten(), l3b]), axis=0)
        img = np.expand_dims(input, axis=2)

        model = NeuralNetwork(inputSize=img.shape, loss=0, lr=learning_rate)
        parameters = {'num_filter': 2, 'filter_size': 3, 'activation': 1, 'weights': w1_b}
        model.addLayer(category='conv', parameters=parameters)
        parameters = {'num_filter': 1, 'filter_size': 3, 'activation': 1, 'weights': w2_b}
        model.addLayer(category='conv', parameters=parameters)
        model.addLayer(category='flatten', parameters=None)
        parameters = {'numOfNeurons': 1, 'activation': 1, 'weights': w3_b}
        model.addLayer(category='dense', parameters=parameters)

        # losses = model.train( img, output, epochs=1)
        model.train(img, output, epochs=1)
        print("Output before: \n")
        print(model.output, '\n')
        print("layer1 kernel1 weights and bias\n")
        weight1 = model.layers[0].weights
        print(weight1[0][:, :, 0, 0], weight1[1][0], '\n')
        print("layer1 kernel 2 weights and bias\n")
        print(weight1[0][:, :, 0, 1], weight1[1][1], '\n')
        print("layer2 kernel 1 channel 1 weights and bias\n")
        weight2 = model.layers[1].weights
        print(weight2[0][:, :, 0, 0], '\n')
        print("layer2 kernel 1 channel 2 weights and bias\n")
        print(weight2[0][:, :, 1, 0], '\n')
        print("layer2 kernel 1 bias\n")
        print(weight2[1], '\n')
        print("fully connected layer weights and bias\n")
        weight3 = model.layers[3].weights
        print(weight3[0][:-1], weight3[0][-1], '\n')
        model.train(img, output, epochs=1)
        print("Output after: \n")
        print(model.output, '\n')

   
