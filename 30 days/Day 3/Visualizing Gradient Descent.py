import matplotlib.pyplot as plt
import numpy as np

# The function: y = x^2
def function(x):
    return x**2

# The Derivative: dy/dx = 2x
# This tells us the slope at any point x
def derivative(x):
    return 2*x

# Gradient Descent Variables
x = 10  # Starting point (far away from 0)
learning_rate = 0.1 # Size of the steps we take
steps = [] # To store history for plotting

# # The Learning Loop
# for i in range(15):
#     steps.append(x)
#     gradient = derivative(x) # Calculate slope
#     x = x - (learning_rate * gradient) # Move opposite to the slope

for i in range(20):
    steps.append(x)
    gradient = derivative(x)
    x = x - (learning_rate * gradient)

# Visualize the path taken
x_vals = np.linspace(-10, 10, 100)
plt.plot(x_vals, function(x_vals), label="Loss Function (Bowl)")
plt.scatter(steps, [function(s) for s in steps], color='red', label="AI Steps")
plt.title("Visualizing Gradient Descent")
plt.xlabel("Parameter Value")
plt.ylabel("Error (Loss)")
plt.legend()
plt.show()

print("Final value of x:", x) # Should be very close to 0