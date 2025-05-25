import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

data = pd.read_csv('winequality.csv')

data['type'] = data['type'].map({'red': 0, 'white': 1})

X = data.drop('quality', axis=1)  # all columns except quality
y = data['quality']               # just the quality column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42  # for reproducible results
)

print("Training the model...")
model = RandomForestRegressor(
    n_estimators=100,  # number of trees in the forest
    random_state=42    # for reproducibility
)
model.fit(X_train, y_train)

print("Evaluating the model...")
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"Model RMSE: {rmse:.2f}")

joblib.dump(model, 'wine_quality_model.joblib')
print("Model saved as wine_quality_model.joblib")

