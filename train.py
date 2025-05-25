import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    train_path = os.path.join(os.environ["SM_CHANNEL_TRAIN"], "train.csv")
    model_dir = os.environ["SM_MODEL_DIR"]
    
    data = pd.read_csv(train_path, header=None)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
