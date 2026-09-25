from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the UCI Real Estate Valuation dataset
dataset = fetch_ucirepo(id=477)

X = dataset.data.features.copy()
y = dataset.data.targets.squeeze()

data = X.copy()
data["house_price_of_unit_area"] = y


# Explore the dataset
print("First 5 rows:")
print(data.head())

print("\nShape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nSummary statistics:")
print(data.describe())

print("\nModel features:")
print(X.columns.tolist())

assert "No" not in X.columns
print("\nConfirmed: 'No' is not included as a feature.")


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))


# Build preprocessing + model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])


# Train the model
model.fit(X_train, y_train)


# Predict on unseen test data
predictions = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")


# New property examples for customer predictions
new_properties = pd.DataFrame([
    {
        "X1 transaction date": 2013.500,
        "X2 house age": 10.0,
        "X3 distance to the nearest MRT station": 300.0,
        "X4 number of convenience stores": 8,
        "X5 latitude": 24.98000,
        "X6 longitude": 121.54000
    },
    {
        "X1 transaction date": 2013.250,
        "X2 house age": 25.0,
        "X3 distance to the nearest MRT station": 1200.0,
        "X4 number of convenience stores": 4,
        "X5 latitude": 24.97000,
        "X6 longitude": 121.52000
    },
    {
        "X1 transaction date": 2013.583,
        "X2 house age": 5.0,
        "X3 distance to the nearest MRT station": 100.0,
        "X4 number of convenience stores": 10,
        "X5 latitude": 24.98500,
        "X6 longitude": 121.54500
    }
])

new_predictions = model.predict(new_properties)

print("\nNEW PROPERTY ESTIMATES")
print("----------------------")

for i, prediction in enumerate(new_predictions, start=1):
    ntd_per_ping = prediction * 10000
    print(
        f"Property {i}: {prediction:.2f} units "
        f"(approximately NT${ntd_per_ping:,.0f} per Ping)"
    )