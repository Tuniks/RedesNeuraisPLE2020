import numpy as np
import random
import math


def threshold(x):
	return 1 if x > 0 else 0

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def tanh(x):
	return (math.exp(2*x) - 1) / (math.exp(2*x) + 1)

def linear(x):
	return x

def rectifier(x):
	return max(0, x)

funcDictionary = {
	"thr": threshold,
	"sig": sigmoid,
	"tgh": tanh,
	"lin": linear,
	"ret": rectifier
}


class NeuralNetwork():
	def __init__(self, topology, functions):
		self.topology = topology
		self.functions = functions
		self.weights = []
		return

	def SetRandomWeights(self):
		for index, layer in enumerate(self.topology[:-1]):
			weight = [ [ random.random() for i in range(topology[index+1]) ] for j in range(layer + 1)]

			self.weights.append(weight)

		# self.weights = [
		# 	[[-0.3, -0.1, 0], [0.2, 0.1, 0.4]],
		# 	[[0.3, 0, -0.1], [0.4, 0, 0.4], [0.1, 0.3, 0]],
		# 	[[0.3, -0.1, 0.5, -0.1], [0.4, 0.3, -0.2, 0.4]]
		# ]

		print(self.weights)


	def Propagate(self, x):
		for index, layer in enumerate(self.topology[:-1]):
			arrayX = np.array([1] + x)

			arrayWeight = np.array(self.weights[index])

			arrayResult = np.matmul(arrayX, arrayWeight)

			result = arrayResult.tolist()

			for j, r in enumerate(result):
				result[j] = funcDictionary[self.functions[index][j]](r)
			x = result

		print("result: ", x)




topology = input("Enter with your network topology. It should be written as a series of numbers divided by spaces indicating the numbers of nodes in each layer (first number = input layer, last number = output layer)\n").split()
topology = [int(i) for i in topology]

print()
print("Now choose each node's activation function. Input each function code separated by space. Input layer does not have activation functions.")
print("Avaiable functions: \nthr: threshold \nsig: sigmoid \ntgh: hyperbolic tangent \nlin: linear \nret: rectifier\n")

functions = []
for index, layer in enumerate(topology[1:]):
	f = input("Enter activation functions of layer " + str(index+2) + "\n").split()
	functions.append(f)
print()

x = input("Enter your input to the network. Input values separated by space\n").split()
x = [float(i) for i in x]
print()

neural = NeuralNetwork(topology, functions)
neural.SetRandomWeights()
neural.Propagate(x)
