import pandas as pd

# 1. Creating a synthetic dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 22],
    'Salary': [50000, 60000, 70000, 80000, 45000],
    'Department': ['HR', 'IT', 'IT', 'Finance', 'HR']
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# 2. Viewing Data
print("First 3 rows:")
print(df.head(3))

# 3. Basic Analysis
print("\nStatistics:")
print(df.describe()) # Gives mean, min, max, std deviation automatically!

# 4. Filtering Data (The AI way)
# Let's find everyone working in 'IT'
it_employees = df[df['Department'] == 'IT']
print("\nIT Employees:\n", it_employees)