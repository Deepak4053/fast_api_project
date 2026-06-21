from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error ,r2_score
import pandas as pd
import joblib

data = fetch_california_housing()
print("Loading dataset...")
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print("Dataset loaded.")

print("total rows and columns in the dataset:", x.shape[0], x.shape[1])
for col in x.columns:
    print("Column name:", col, "Data type:", x[col].dtype)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training dataset shape:", x_train.shape)
print("Target variable shape:", y_train.shape)

model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training the model...")
model.fit(x_train, y_train)
print("Model trained.")

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)
joblib.dump(model, 'california_housing_model.pkl')
print("Model saved as california_housing_model.pkl")

joblib.dump(x.columns, 'feature_names.pkl')
print("Feature names saved as feature_names.pkl")