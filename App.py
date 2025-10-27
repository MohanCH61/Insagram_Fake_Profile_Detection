import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib
from flask import Flask, render_template, request
import os

app = Flask(__name__, static_folder='static')

# Load datasets
def load_datasets():
    train_data_path = 'datasets/Insta_Fake_Profile_Detection/train.csv'
    test_data_path = 'datasets/Insta_Fake_Profile_Detection/test.csv'
    instagram_df_train = pd.read_csv(train_data_path)
    instagram_df_test = pd.read_csv(test_data_path)
    return instagram_df_train, instagram_df_test

instagram_df_train, instagram_df_test = load_datasets()

# Optional data visualization
def visualize_data():
    sns.countplot(x='fake', data=instagram_df_train)
    plt.show()
    sns.countplot(x='private', data=instagram_df_train)
    plt.show()
    sns.countplot(x='profile pic', data=instagram_df_train)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.hist(instagram_df_train['nums/length username'].values, bins=20, color='blue', alpha=0.7)
    plt.xlabel('nums/length username')
    plt.ylabel('Frequency')
    plt.title('Histogram of nums/length username')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 10))
    cm = instagram_df_train.corr()
    sns.heatmap(cm, annot=True, cmap='coolwarm')
    plt.show()

# Prepare Data
X_train = instagram_df_train.drop(columns=['fake'])
X_test = instagram_df_test.drop(columns=['fake'])
y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train.values)
X_test_scaled = scaler_x.transform(X_test.values)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Build and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs_hist = model.fit(X_train_scaled, y_train, epochs=50, verbose=1, validation_split=0.1)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/fake_detection_model.h5")

# Evaluate model
predicted = model.predict(X_test_scaled)
predicted_value = [np.argmax(i) for i in predicted]
test = [np.argmax(i) for i in y_test]

classification_rep = classification_report(test, predicted_value)
conf_matrix = confusion_matrix(test, predicted_value)

print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

# Flask routes
@app.route('/')
def index():
    return render_template('index99.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        profile_pic = float(request.form['profile_pic'])
        nums_length_username = float(request.form['nums_length_username'])
        fullname_words = float(request.form['fullname_words'])
        nums_length_fullname = float(request.form['nums_length_fullname'])
        name_equals_username = float(request.form['name_equals_username'])
        description_length = float(request.form['description_length'])
        external_url = float(request.form['external_url'])
        private = float(request.form['private'])
        posts = float(request.form['posts'])
        followers = float(request.form['followers'])
        follows = float(request.form['follows'])

        input_data = np.array([[profile_pic, nums_length_username, fullname_words,
                                nums_length_fullname, name_equals_username, description_length,
                                external_url, private, posts, followers, follows]])

        
        scaler = joblib.load("model/scaler.pkl")
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)
        prediction_label = "Fake Account 🚨" if np.argmax(prediction) == 1 else "Real Account ✅"

        return render_template('result99.html', prediction=prediction_label)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=9001)
