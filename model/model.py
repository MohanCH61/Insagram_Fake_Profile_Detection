import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os

# Load dataset
df = pd.read_csv("datasets/Insta_Fake_Profile_Detection/train.csv")

# Features and labels
X = df.drop("fake", axis=1)
y = df["fake"]

# Split train/test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler for Flask app
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# Build model
model = Sequential([
    Dense(50, input_dim=X_train.shape[1], activation='relu'),
    Dense(150, activation='relu'),
    Dropout(0.3),
    Dense(150, activation='relu'),
    Dropout(0.3),
    Dense(25, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=30, batch_size=16, validation_data=(X_val_scaled, y_val))

# Save model
model.save("model/fake_detection_model.h5")

print("✅ Model trained and saved as fake_detection_model.h5")
