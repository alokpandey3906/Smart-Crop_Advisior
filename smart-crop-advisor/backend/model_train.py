import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("backend/crop_data.csv.csv")


X = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))


with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to crop_model.pkl")
