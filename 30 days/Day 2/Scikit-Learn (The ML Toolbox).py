import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# Let's use the DataFrame 'df' from Part 2

# 1. Simple Bar Chart

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 22],
    'Salary': [50000, 60000, 70000, 80000, 45000],
    'Department': ['HR', 'IT', 'IT', 'Finance', 'HR']
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split

# Features (Inputs) - Usually denoted as X
X = df[['Age']]

# Target (Output) - Usually denoted as y
y = df['Salary']

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total Data: {len(df)}")
print(f"Training Data: {len(X_train)}")
print(f"Testing Data: {len(X_test)}")