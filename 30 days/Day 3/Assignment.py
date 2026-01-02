# Assignment for Day 3
# 1. Create two NumPy arrays (vectors) of size 5 with random numbers.
# 2. Calculate the Dot Product.
# 3. Calculate the Sum of both arrays.
# 4. Pass the result of the Dot Product through the Sigmoid function you wrote above.
# 5. plot the sum
# 6. Print: "The probability is: [result]"



import numpy as np
import matplotlib.pyplot as plt


arr1 = np.random.rand(5)
arr2 = np.random.rand(5)

dot = np.dot(arr1, arr2)
print(dot)
s = arr1 + arr2
print(s)


def sig(x):
    return 1 / (1 + np.exp(-x))
sm = sig(s)


plt.figure(figsize=(12, 5))
plt.plot(sm, sig(sm), color= "red")
plt.scatter(sm, sig(sm), color= "blue")
plt.title("Sigmoid")
plt.grid()
plt.show()

probability = sig(dot).round(2)
print(f"Probability: {probability}")










