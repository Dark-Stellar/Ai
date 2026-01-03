
# Step 1: Loading "Actual" Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- DATASET CREATION ---
# This simulates a real CSV file containing housing data.
csv_data = """Area_SqFt,Bedrooms,Age_Years,Location_Score,Price
1500,3,10,8,350000
1800,4,5,9,450000
2400,4,15,7,500000
1200,2,30,5,200000
3000,5,2,10,800000
1000,2,20,6,220000
2000,3,12,8,420000
1600,3,25,4,280000
2800,4,8,9,650000
1100,2,18,5,210000
1400,3,10,7,320000
1900,3,6,8,430000
2200,4,20,6,410000
3500,5,3,10,950000
900,1,40,3,150000
2100,4,14,7,440000
2600,5,9,8,600000
1300,3,22,5,260000
1750,3,11,9,400000
3100,5,5,9,820000
1550,3,15,6,310000
2300,4,7,8,550000
1250,2,28,4,205000
2900,4,4,10,750000
950,2,35,3,160000
"""

# Read the string as a CSV file
df = pd.read_csv(io.StringIO(csv_data))

print("--- Data Loaded ---")
print(df.head())

# Visualize the relationship: Area vs Price
plt.figure(figsize=(8, 5))
plt.scatter(df['Area_SqFt'], df['Price'], color='blue')
plt.title("House Size vs Price")
plt.xlabel("Area (SqFt)")
plt.ylabel("Price ($)")
plt.grid(True)
# plt.show()



#  Step 2: Data Preprocessing



# X = Features (Area, Bedrooms, Age, Location)
X = df[['Area_SqFt', 'Bedrooms', 'Age_Years', 'Location_Score']]

# y = Target (Price)
y = df['Price']

# Split Data: 80% for training the AI, 20% for testing it
# random_state=42 ensures we get the same split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Data Size: {len(X_train)} houses")
print(f"Testing Data Size: {len(X_test)} houses")





# Step 3: Model 1 - Linear Regression




# 1. Initialize
lin_reg = LinearRegression()

# 2. Train (Fit)
lin_reg.fit(X_train, y_train)

# 3. Predict
y_pred_lin = lin_reg.predict(X_test)

# 4. Evaluate
print("\n--- Linear Regression Results ---")
# Compare the first predicted price vs the real price
print(f"Predicted: ${y_pred_lin[0]:,.0f} | Actual: ${y_test.values[0]:,.0f}")

# R2 Score (1.0 is perfect, 0.0 is terrible)
accuracy_lin = r2_score(y_test, y_pred_lin)

print(f"Model Accuracy (R2 Score): {accuracy_lin:.2f}")




#   Step 4: Model 2 - Random Forest (Advanced)




# 1. Initialize
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 2. Train
rf_model.fit(X_train, y_train)

# 3. Predict
y_pred_rf = rf_model.predict(X_test)

# 4. Evaluate
print("\n--- Random Forest Results ---")
print(f"Predicted: ${y_pred_rf[0]:,.0f} | Actual: ${y_test.values[0]:,.0f}")

accuracy_rf = r2_score(y_test, y_pred_rf)
print(f"Model Accuracy (R2 Score): {accuracy_rf:.2f}")

# Comparison
if accuracy_rf > accuracy_lin:
    print("\n✅ Random Forest performed better!")
else:
    print("\n✅ Linear Regression performed better!")




#Part 3: Using your AI (Inference)



# Create the new data array matches the column structure:
# [Area, Bedrooms, Age, Location]
my_dream_house = [[2500, 4, 5, 8]]

# Predict using Random Forest
predicted_price = rf_model.predict(my_dream_house)

print("\n--- Final Prediction ---")
print(f"The AI estimates this house costs: ${predicted_price[0]:,.2f}")