import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


np.random.seed(0)
input_size = 2
output_size = 1
learning_rate = 0.1


weights = np.random.rand(input_size, output_size)
biases = np.random.rand(output_size)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [0], [0], [1]])

l=[]

for epoch in range(1500):

    layer_input = np.dot(X, weights) + biases
    layer_output = sigmoid(layer_input)
    
    
    loss = np.mean((layer_output - Y) ** 2)
    l.append(loss)

    gradient = 2 * (layer_output - Y) * layer_output * (1 - layer_output)
    d_weights = np.dot(X.T, gradient)
    d_biases = np.sum(gradient, axis=0)
    

    weights -= learning_rate * d_weights
    biases -= learning_rate * d_biases

predictions = sigmoid(np.dot(X, weights) + biases)

print("预测结果：")
print(predictions)
plt.plot(list(range(0,1500)),l)
plt.show()
