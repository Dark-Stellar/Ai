import matplotlib.pyplot as plt
import pandas as pd
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

names = df['Name']
salaries = df['Salary']

plt.figure(figsize=(8, 5)) # Set graph size
plt.bar(names, salaries, color='skyblue')

plt.title('Employee Salary Comparison')
plt.xlabel('Names')
plt.ylabel('Salary ($)')
plt.show()

# 2. Scatter Plot (Used often in Regression)
plt.scatter(df['Age'], df['Salary'], color='red')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid(True)
plt.show()