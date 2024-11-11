![image](https://github.com/user-attachments/assets/ecc7c5a5-4844-48f4-8c44-340a1371ed95)
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['house'] = data.target 


print("features in the dataset:", list(df.columns))
numerical_features = df.select_dtypes(include=[np.number]).columns
categorical_features = df.select_dtypes(include=['object', 'category']).columns
print("\nnumerical features count:", len(numerical_features))
print("categorical features count:", len(categorical_features))

plt.figure(figsize=(10, 6))
plt.plot(df['AveRooms'], df['house'], 'b.', alpha=0.1)
plt.xlabel("average roms (avroms)")
plt.ylabel("mdian hose vlue (house")
plt.title("plt between avrage rooms and house")
plt.show()

missing_values = df.isnull().sum()
print("\nmissingvalues in each field:\n", missing_values)

df.fillna(df.median(), inplace=True)
print("\nmissingvalues after substitution:\n", df.isnull().sum())

X = df.drop("house",axis = 1)
y = df["house"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

svm_reg = SVR(kernel='linear')
svm_reg.fit(X_train, y_train)
y_pred_svm = svm_reg.predict(X_test)
