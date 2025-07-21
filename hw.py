import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("BMW_Car_Sales_Classification.csv")

print(data.info())
print(data.head())

# Separate features and target
X = data.drop("Sales_Classification", axis=1)
y = data["Sales_Classification"]


le = LabelEncoder()
y = le.fit_transform(y) 
print(y)


cat_cols = ["Model", "Region", "Color", "Fuel_Type", "Transmission"]
num_cols = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "Sales_Volume"]


# Plotting
plt.figure(figsize=(10,8))

plt.subplot(221)
plt.scatter(X["Engine_Size_L"], y, s=10, c='green', marker='o')
plt.xlabel("Engine_Size_L")
plt.ylabel("Sales_Classification (encoded)")

plt.subplot(222)
plt.scatter(X["Mileage_KM"], y, s=10, c='red', marker='o')
plt.xlabel("Mileage_KM")

plt.subplot(223)
plt.scatter(X["Price_USD"], y, s=10, c='blue', marker='o')
plt.xlabel("Price_USD")

plt.subplot(224)
plt.scatter(X["Sales_Volume"], y, s=10, c='yellow', marker='o')
plt.xlabel("Sales_Volume")

plt.tight_layout()
plt.show()

