import numpy as np

# 1. Creating an Array
# A simple 1D array (Vector)
arr = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr)

# A 2D array (Matrix) - Think of this like an Excel sheet or an Image
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Matrix:\n", matrix)

# 2. Key Properties
print("\nShape of matrix:", matrix.shape)  # Output: (2, 3) -> 2 rows, 3 columns
print("Data Type:", matrix.dtype)

# 3. Operations
# If you multiply a list in Python: [1,2] * 2 = [1,2,1,2]
# In NumPy, it does Math:
print("\nMatrix * 2:\n", matrix * 2)

# 4. Useful functions for AI initialization
zeros = np.zeros((3,3)) # Creates a 3x3 matrix of 0s
print("\nZeros Matrix:\n", zeros)