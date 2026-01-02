import numpy as np

# Imagine a simple AI trying to predict if you will buy a house.
# Inputs [Price (normalized), Size (normalized), Location Score]
inputs = np.array([0.8, 0.9, 0.2])

# Weights (How important is each factor?)
# Price is negative (high price = bad), Size is positive, Location is positive
weights = np.array([-0.5, 0.8, 0.1])

# The Dot Product
# Calculation: (0.8 * -0.5) + (0.9 * 0.8) + (0.2 * 0.1)
#              (-0.4)       + (0.72)      + (0.02)      = 0.34
prediction = np.dot(inputs, weights)

print(f"Prediction Score: {prediction}")

# If score > 0, the AI predicts "Yes, buy".
if prediction > 0:
    print("AI Decision: Buy the house")
else:
    print("AI Decision: Don't buy")