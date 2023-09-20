import numpy as np
import matplotlib.pyplot as plt
# 定义神经网络的激活函数（这里使用 sigmoid 函数）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# 定义神经网络的权重和偏置
# 这里我们初始化权重和偏置为随机值，你也可以选择其他初始化方法
np.random.seed(0)
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1

# 初始化权重和偏置
np.random.seed(0)
input_layer_weights = np.random.rand(input_size, hidden_size)
hidden_layer_weights = np.random.rand(hidden_size, output_size)
input_layer_biases = np.random.rand(1, hidden_size)
hidden_layer_biases = np.random.rand(1, output_size)

# 训练数据集，包括输入和对应的期望输出
# 逻辑与操作的真值表
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

l=[]
# 训练神经网络
for epoch in range(7500):
    # 前向传播
    input_layer_activation =sigmoid(np.dot(X, input_layer_weights) + input_layer_biases)
    output_layer_activation = sigmoid(np.dot(input_layer_activation, hidden_layer_weights) + hidden_layer_biases)
    
    # 计算损失函数（均方误差）
    loss = np.mean((output_layer_activation - Y) ** 2)
    l.append(loss)
    # 计算梯度
    d_output_layer = 2 * (output_layer_activation - Y) * output_layer_activation * (1 - output_layer_activation)
    d_input_layer = d_output_layer.dot(hidden_layer_weights.T) * input_layer_activation * (1 - input_layer_activation)
    d_weights = np.dot(input_layer_activation.T, d_output_layer)
    d_biases = np.sum(d_output_layer, axis=0,keepdims=True)
    d_weights_2 = np.dot(X.T, d_input_layer)
    d_biases_2 = np.sum(d_input_layer, axis=0,keepdims=True)
    
    # 使用梯度下降来更新权重和偏置
    hidden_layer_weights -= learning_rate * d_weights
    hidden_layer_biases -= learning_rate * d_biases
    input_layer_weights -= learning_rate * d_weights_2
    input_layer_biases -= learning_rate * d_biases_2


# 使用训练好的神经网络进行预测
input_layer_activation = sigmoid(np.dot(X, input_layer_weights) + input_layer_biases)
output_layer_activation = sigmoid(np.dot(input_layer_activation, hidden_layer_weights) + hidden_layer_biases)
print("预测结果：")
print(output_layer_activation)
plt.plot(list(range(0,7500)),l)
plt.show()
