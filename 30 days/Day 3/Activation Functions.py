import matplotlib.pyplot as plt
import numpy as np

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU Function
def relu(x):
    return np.maximum(0, x)

# Generate data
x_data = np.linspace(-10, 10, 100)

plt.figure(figsize=(12, 5))

# Plot Sigmoid
plt.subplot(1, 2, 1)
plt.plot(x_data, sigmoid(x_data), color='blue')
plt.title("Sigmoid (0 to 1)")
plt.grid()

# Plot ReLU
plt.subplot(1, 2, 2)
plt.plot(x_data, relu(x_data), color='green')
plt.title("ReLU (0 to Infinity)")
plt.grid()

plt.show()







