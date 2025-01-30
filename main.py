
import numpy as np
import matplotlib.pyplot as plt
import sys
from parameters import generateExample2


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr=0.01, weights=None):
        self.activation = activation
        self.no_input = input_num
        self.lr = lr    

        self.input = None
        if weights is None:
            self.weights = np.random.rand(self.no_input)
            self.bias = np.random.rand()
        else:
            self.weights = weights[:-1]
            self.bias = weights[-1]
 
    #This method returns the activation of the net
    def activate(self,net):
        if self.activation == 0:  # linear
            output = net
        else:  # logistic-sigmoid
            output = 1 / (1 + np.exp(-1 * net))   
        
        return output
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        self.input = input
        net = self.input * self.weights 
        net = np.sum(np.sum(net) + self.bias)
        self.output = self.activate(net)
        return self.output

    # This method returns the derivative of the activation function with respect to the net
    def activationderivative(self):
        # errorder is the derivative of loss/error with respect to the output
        if self.activation == 1:  # logistic
            act_der = self.output * (1 - self.output)
        else:  # linear
            act_der = 1
        return act_der  # a scalar
    
    # This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):  # wtimesdelta is a matrix
        # Be careful about the last hidden layer  wtimesdelta should be error_der times actder only
        delta = wtimesdelta * self.activationderivative()  

        self.cal_der_w = delta * np.array(self.input)  # One node only have one delta value.
        self.cal_der_b = delta * 1

        wtimesdelta = np.array(delta) * self.weights

        
        # update weights

        return wtimesdelta, self.cal_der_w , self.cal_der_b

    # Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self,cal_der_w, cal_der_b):
        self.weights = self.weights - np.array(self.lr) * cal_der_w
        self.bias = self.bias - np.array(self.lr) * cal_der_b
        


#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the inputsize, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.no_input = input_num
        self.lr = lr   
        self.all_neurons = []
        
        if weights is None:
            weights = [[[np.random.rand(1,input_num[0])], [np.random.rand(1)]]]*numOfNeurons
        
        for i in range(numOfNeurons):
            self.all_neurons.append(Neuron(self.activation, self.no_input, self.lr, weights[i]))

    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.output_vector = []  # list
        for i in range(self.numOfNeurons):
            self.output_vector.append(self.all_neurons[i].calculate(input))
        return np.array(self.output_vector)  # array


    
    # given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value),
    # sum up its ownw*delta, and then update the wieghts (using the updateweight() method).
    # I should return the sum of w*delta. - A vector
    def calculatewdeltas(self, wtimesdelta):
        lst = []
        for i in range(self.numOfNeurons):
            non_sum_wtimesdelta, cal_der_w , cal_der_b = self.all_neurons[i].calcpartialderivative(np.array(wtimesdelta))
            lst.append(non_sum_wtimesdelta)
            self.all_neurons[i].updateweight(cal_der_w, cal_der_b)
        
        lst = np.array(lst)
        wtimesdelta = np.sum(lst, axis=0)
        return wtimesdelta

    def show_weight(self):
        for i in range(self.numOfNeurons):
            print("Weights")
            print(self.all_neurons[i].weights)
            print("Bias")
            print(self.all_neurons[i].bias)


#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,inputSize, loss, lr):
        #self.numOfLayers = numOfLayers
        self.loss = loss
        self.layers = []
        self.lr = lr
        self.num_Layers = 0
        self.inputSize = inputSize
        self.num_Neurons = []

    def addLayer(self,type,layer_param=None):

        if self.num_Layers == 0:
            input_dim = self.inputSize
        else:
            input_dim = self.layers[-1].output.shape
        
        if type == 'conv':
            self.layers.append(ConvolutionalLayer(layer_param['num_kernel'], layer_param['kernel_size'],layer_param['activation'],input_dim, self.lr, layer_param['weights']))
        elif type == 'maxpool':
            #kernelSize, inputDimension
            self.layers.append(MaxPoolingLayer(layer_param['kernel_size'], input_dim))
        elif type == 'flatten':
            self.layers.append(FlattenLayer(input_dim))
        elif type == 'dense':
            self.layers.append(FullyConnected(layer_param['numOfNeurons'], layer_param['activation'],input_dim, self.lr, layer_param['weights']))
        self.num_Layers += 1
        

    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input, y):
        self.output = np.zeros(y.shape)

        for j in range(input.shape[0]): 
            tmp_output = input[j]
            for i in range(len(self.layers)):
                tmp_output = self.layers[i].calculate(tmp_output)
            s = tmp_output.shape[0]
            self.output[j] = tmp_output.reshape(s)

        return np.array(self.output)

    # Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self, y_hat, y):  # all in type of ndarray
        error = 0
        
        if self.loss == 0:  # sum of square errors
            error = np.sum(0.5*(y_hat-y)**2)
        else:  # binary cross entropy
            for i in range(y_hat.shape[0]):
                error += np.sum(-(y[i] * np.log(y_hat[i]) + (1-y[i])*np.log(1 -y_hat[i])))
            
        return error / y_hat.shape[0]


    # Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)
    def lossderiv(self, yp, y):

        if self.loss == 0:
            error_der = 0
            for i in range(yp.shape[0]):
                error_der += (yp[i] - y[i])
                

        else:   
            error_der = 0
            for i in range(yp.shape[0]):
                error_der += (-(y[i]/yp[i])+((1-y[i])/(1-yp[i])))  # derivative of loss and yp times derivative of yp and O

        return (1 / yp.shape[0]) * error_der

# Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values
    def train(self, x, y, num_epochs=1000):
        error = []
        for epoch_index in range(num_epochs):
            y_hat = self.calculate(x,y)  
            
            if error == []:
                past_loss = 1e+10
            else:
                past_loss = current_loss

            current_loss = self.calculateloss(y_hat, y)
            
            if np.abs(current_loss - past_loss) <= 1e-10:
                break
            
            error.append(current_loss)
            error_deriv = self.lossderiv(y_hat, y)
            wtimesdelta = error_deriv 

            for layer_index in range(len(self.layers)-1, -1, -1):  # Layers for L[0] to L[numOfLayer-1]
                wtimesdelta = self.layers[layer_index].calculatewdeltas(wtimesdelta) 
    
        return error, y_hat


#Convolution Class
class ConvolutionalLayer:
    def __init__(self,num_kernel, kernel_size, activation, input_dim, lr, weights=None):
        self.num_kernel = num_kernel
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_dim = input_dim
        self.lr = lr
        self.weights = weights 
        self.output_size = int((input_dim[1] - kernel_size)/1+1)
        self.all_neurons = np.full([self.num_kernel, self.output_size, self.output_size], None)
        self.output = np.full([self.num_kernel, self.all_neurons.shape[1],self.all_neurons.shape[1]], None)
        
        #define the weights for conv layer.
        if self.weights is None:
            self.weights = [[[np.random.rand(self.kernel_size,self.kernel_size)],[np.random.rand(1)]]]* self.num_kernel

        self.row_neurons = []
        for kernel in range(self.num_kernel):
                for i in range (self.output_size):
                    for j in range (self.output_size):
                        self.all_neurons[kernel][i][j] = Neuron(self.activation, self.kernel_size ** 2+1, self.lr, weights= self.weights[kernel])
        
    #calculate the output of each neuron in the conv layer.     
    def calculate(self, input):
        self.channel = input.shape[0]
        for kernel in range (self.num_kernel):
                for i in range (self.output_size):
                    for j in range (self.output_size):
                        temp = 0
                        neuron_input = []
                        for channel in range (self.channel):
                            for a in range (self.kernel_size):
                                for b in range (self.kernel_size):
                                    neuron_input.append(input[channel][i+a][j+b])
                        self.output[kernel][i][j] = self.all_neurons[kernel][i][j].calculate(np.array(neuron_input).reshape(self.channel, self.kernel_size,self.kernel_size))
                        
                                      
                    
        return self.output
        
    #calulate the weight updates and also generates the deltaW values for prevoius layer. 
    def calculatewdeltas(self, input):
        self.wtimesdelta = input
        update_wtimesdelta = []
        update_weight = []
        for kernel in range(self.num_kernel):
            der_w_lst = []
            der_b_lst = []
            for i in range (self.output_size):
                for j in range (self.output_size):
                    wtimesdeltassss , der_w , der_b = self.all_neurons[kernel][i][j].calcpartialderivative(self.wtimesdelta[kernel][i][j])
                    der_w_lst.append(der_w)
                    der_b_lst.append(der_b)
                    update_wtimesdelta.append(wtimesdeltassss)
                    
            update_weight.append([sum(x) for x in zip(*der_w_lst)])
            update_bias = sum(der_b_lst)
            
            #update the weights of each neuron
            for i in range (self.output_size):
                for k in range (self.output_size):
                    self.all_neurons[kernel][i][k].updateweight(update_weight, update_bias)
        
        
        update_wtimesdelta = np.array(update_wtimesdelta)   
        self.wtimesdelta = np.zeros(self.input_dim)
        
        #cast the wtimesDelta to shape of previous layer. 
        for chan in range (self.channel):
            index = 0
            for i in range (self.output_size):
                for j in range (self.output_size):
                    for a in range (self.kernel_size):
                        for b in range (self.kernel_size):
                            self.wtimesdelta[chan][i+a][j+b] += update_wtimesdelta[index][0][chan][a][b] 
                    index +=1
        return self.wtimesdelta
    
    #Print the weights
    def show_weight(self):
        for kernel in range(self.num_kernel):
            print("Kernel: {} weights".format(kernel+1))
            print(self.all_neurons[kernel][0][0].weights[0])
            print("Kernel: {} bias".format(kernel+1))
            print(self.all_neurons[kernel][0][0].bias)


#MAxpool layer
class MaxPoolingLayer:
    #maxpool initialization
    def __init__(self,kernel_size, input_dim):
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.stride = kernel_size
        h = int((self.input_dim [1] - self.kernel_size)/self.stride)+1 
        w = int((self.input_dim [2] - self.kernel_size)/self.stride)+1 
        self.output = np.zeros((input_dim[0], h, w)) 

    #calculate the output of maxpool layer
    def calculate(self, input):
        self.input = input
        for chan in range(self.input_dim[0]):
            j = out_y = 0
            # slide the max pooling window vertically across the image
            while j + self.kernel_size <= self.input_dim [1]:
                i = out_x = 0
            # slide the max pooling window horizontally across the image
                while i + self.kernel_size <= self.input_dim [2]:
                # choose the maximum value within the window at each step and store it to the output matrix
                    self.output[chan, out_y, out_x] = np.max(self.input[chan, j:j+self.kernel_size, i:i+self.kernel_size])
                    i += self.stride
                    out_x += 1
                j += self.stride
                out_y += 1
        return self.output
    
    #cast the wtimesDelta form next layer to the prevoius layer using the mask.
    def calculatewdeltas(self, input):
    
        self.wtimesdelta = np.zeros(self.input_dim)
    
        for chan in range(self.input_dim[0]):
            j = out_y = 0
            while j + self.kernel_size <= self.input_dim [1]:
                i = out_x = 0
                while i + self.kernel_size <= self.input_dim [2]:
                # obtain index of largest value in input for current window
                    idx = np.nanargmax(self.input[chan, j:j+self.kernel_size, i:i+self.kernel_size])
                    (a, b) = np.unravel_index(idx, self.input[chan, j:j+self.kernel_size, i:i+self.kernel_size].shape)
                    self.wtimesdelta[chan, j+a, i+b] = input[chan, out_y, out_x]
                
                    i += self.stride
                    out_x += 1
                j += self.stride
                out_y += 1
        
        return self.wtimesdelta


#Flatten Layer class
class FlattenLayer:
    def __init__(self,input_dim):
        self.input_dim = input_dim
        self.output = np.zeros(input_dim)
        self.output = self.output.flatten()

    #Flatten layer calculate
    def calculate(self, input):
        self.output = input.flatten()

        return self.output

    def calculatewdeltas(self, input):
        self.wtimesdelta = input.reshape(self.input_dim)
        return self.wtimesdelta



#Main Function
if __name__=="__main__":

    #Arguments parsing
    learning_rate = float(sys.argv[1])
    example = sys.argv[2]
    
    
    if example == 'example3':
        l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, y = generateExample2(input_dim=8, final_layer=18)
        img=np.expand_dims(input,axis=(0,1))
    
        #Weights
        w1 = [[[l1k1],[l1b1]], 
             [[l1k2],[l1b2]]]

        w3 = [[[l3], [l3b]]]

        test = NeuralNetwork(inputSize = img[0].shape,loss=0, lr=learning_rate)
        layer_param={'num_kernel':2, 'kernel_size': 3, 'activation': 1, 'weights': w1}
        test.addLayer(type='conv', layer_param=layer_param)
        layer_param={'kernel_size': 2}
        test.addLayer(type='maxpool', layer_param=layer_param)
        test.addLayer(type='flatten')
        layer_param={'numOfNeurons' : 1, 'activation' : 1, 'weights': w3}
        test.addLayer(type='dense',layer_param=layer_param)

        losses, y_pred = test.train( img, y, num_epochs=20)
        print("Output:",y_pred)
    
        print("Weights and Bias of 1st Conv Layer")
        test.layers[0].show_weight()
        print("Weights and Bias of Output Layer")
        test.layers[3].show_weight()

        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss per Epochs (Example-3)")
        plt.show()


    elif example == 'example2':
        l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, y = generateExample2(input_dim=7, final_layer=9)
        img=np.expand_dims(input,axis=(0,1))
    
    
        w1 = [[[l1k1],[l1b1]], 
             [[l1k2],[l1b2]]]
    
   
        w2 = [[[l2c1,l2c2], [l1b1]]]

        w3 = [[[l3], [l3b]]]
    
        test = NeuralNetwork(inputSize = img[0].shape,loss=0, lr=learning_rate)
        layer_param={'num_kernel':2, 'kernel_size': 3, 'activation': 1, 'weights': w1}
        test.addLayer(type='conv', layer_param=layer_param)
        layer_param={'num_kernel':1, 'kernel_size': 3, 'activation': 1, 'weights': w2}
        test.addLayer(type='conv', layer_param=layer_param)
        test.addLayer(type='flatten')
        layer_param={'numOfNeurons' : 1, 'activation' : 1, 'weights': w3}
        test.addLayer(type='dense',layer_param=layer_param)

        losses, y_pred = test.train( img, y, num_epochs=20)
        print("Output:",y_pred)
    
        print("Weights and Bias of 1st Conv Layer")
        test.layers[0].show_weight()
        print("Weights and Bias of 2nd Conv Layer")
        test.layers[1].show_weight()
        print("Weights and Bias of Output Layer")
        test.layers[3].show_weight()

        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss per Epochs (Example-2)")
        plt.show()

    elif example == 'example1':
        l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, y = generateExample2(input_dim=5, final_layer=9)
        img=np.expand_dims(input,axis=(0,1))
    
        w1 = [[[l1k1],[l1b1]], 
             [[l1k2],[l1b2]]]

        w3 = [[[l3], [l3b]]]
    
        test = NeuralNetwork(inputSize = img[0].shape,loss=0, lr=learning_rate)
        layer_param={'num_kernel':1, 'kernel_size': 3, 'activation': 1, 'weights': w1}
        test.addLayer(type='conv', layer_param=layer_param)
        test.addLayer(type='flatten')
        layer_param={'numOfNeurons' : 1, 'activation' : 1, 'weights': w3}
        test.addLayer(type='dense',layer_param=layer_param)

        losses, y_pred = test.train( img, y, num_epochs=20)
        print("Output:",y_pred)
    
        print("Weights and Bias of 1st Conv Layer")
        test.layers[0].show_weight()
        print("Weights and Bias of Output Layer")
        test.layers[2].show_weight()

        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss per Epochs (Example-1)")
        plt.show()