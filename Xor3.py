import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(0)
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1


np.random.seed(0)
input_layer_weights = np.random.rand(input_size, hidden_size)
hidden_layer_weights = np.random.rand(hidden_size, output_size)
input_layer_biases = np.random.rand(1, hidden_size)
hidden_layer_biases = np.random.rand(1, output_size)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
wh1=np.zeros_like(input_layer_weights)
bh1=np.zeros_like(input_layer_biases)
wh2=np.zeros_like(hidden_layer_weights)
bh2=np.zeros_like(hidden_layer_biases)
l=[]

for epoch in range(7500):

    input_layer_activation =sigmoid(np.dot(X, input_layer_weights) + input_layer_biases)
    output_layer_activation = sigmoid(np.dot(input_layer_activation, hidden_layer_weights) + hidden_layer_biases)
    

    loss = np.mean((output_layer_activation - Y) ** 2)
    l.append(loss)
 
    d_output_layer = 2 * (output_layer_activation - Y) * output_layer_activation * (1 - output_layer_activation)
    d_input_layer = d_output_layer.dot(hidden_layer_weights.T) * input_layer_activation * (1 - input_layer_activation)
    d_weights = np.dot(input_layer_activation.T, d_output_layer)
    d_biases = np.sum(d_output_layer, axis=0,keepdims=True)
    d_weights_2 = np.dot(X.T, d_input_layer)
    d_biases_2 = np.sum(d_input_layer, axis=0,keepdims=True)
    
    wh1+=d_weights_2*d_weights_2
    bh1+=d_biases_2*d_biases_2
    wh2+=d_weights*d_weights
    bh2+=d_biases*d_biases

    hidden_layer_weights -= learning_rate * d_weights/(np.sqrt(wh2))
    hidden_layer_biases -= learning_rate * d_biases/(np.sqrt(bh2))
    input_layer_weights -= learning_rate * d_weights_2/(np.sqrt(wh1))
    input_layer_biases -= learning_rate * d_biases_2/ (np.sqrt(bh1))



input_layer_activation = sigmoid(np.dot(X, input_layer_weights) + input_layer_biases)
output_layer_activation = sigmoid(np.dot(input_layer_activation, hidden_layer_weights) + hidden_layer_biases)
print("预测结果：")
print(output_layer_activation)
plt.plot(list(range(0,7500)),l)
plt.show()
