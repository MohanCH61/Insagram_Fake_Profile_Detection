# Instagram Fake Profile Detection

A Flask-based deep learning web application that predicts whether an Instagram account is **fake** or **real** using profile metrics such as username patterns, follower counts, post activity, and other parameters.

---

## 🚀 Project Overview

This project uses a **machine learning model** trained on synthetic Instagram profile data to detect fake accounts. It analyzes various factors such as profile picture availability, username structure, followers/following ratio, and account privacy status.

The app provides a simple web interface where users can input Instagram profile parameters manually to get a prediction.

---

## 🧠 Features

- Detects fake vs real Instagram accounts using ML model.
- Interactive web interface using Flask.
- User-friendly form for entering Instagram account data.
- Displays prediction results in an intuitive HTML page.
- Includes visualization-ready dataset (train.csv & test.csv).

---

## 🧩 Tech Stack

- **Python**
- **Flask**
- **Scikit-learn / TensorFlow (for model training)**
- **HTML, CSS (for frontend templates)**
- **Pandas, NumPy (for data processing)**

---

## ⚙️ Project Structure

```
Instagram_Fake_Profile_Detection/
│
├── datasets/
│   ├── train.csv
│   ├── test.csv
│   └── datatrain.py
│
├── model/
│   ├── fake_detection_model.h5
│   ├── model.py
│   └── scaler.pkl
│
├── static/
│   ├── background.pn
│   └── style.css
│
├── templates/
│   ├── index99.html
│   └── result99.html
│
├── App.py
├── requirements.txt
└── README.md
```

---

## 🧪 Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # For Windows
source venv/bin/activate  # For macOS/Linux
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Generate Dataset (if not already created)
```bash
python datasets/datatrain.py
```

### 4️⃣ Run the Flask App
```bash
python App.py
```

### 5️⃣ Access Web App
Open your browser and go to 👉 **http://127.0.0.1:5000/**

---

## 📊 Example Inputs

### Fake Account Example
| Feature | Value |
|----------|--------|
| Profile Picture | 0 |
| Nums/Length Username | 0.8 |
| Fullname Words | 1 |
| Nums/Length Fullname | 0.7 |
| Name Equals Username | 1 |
| Description Length | 10 |
| External URL | 1 |
| Private Account | 0 |
| Number of Posts | 10 |
| Followers | 30 |
| Follows | 4000 |

### Real Account Example
| Feature | Value |
|----------|--------|
| Profile Picture | 1 |
| Nums/Length Username | 0.2 |
| Fullname Words | 3 |
| Nums/Length Fullname | 0 |
| Name Equals Username | 0 |
| Description Length | 60 |
| External URL | 0 |
| Private Account | 1 |
| Number of Posts | 20 |
| Followers | 4200 |
| Follows | 800 |

---

## 🎯 Output

- The app predicts whether the account is **FAKE** or **REAL**.
- Displays results on `result99.html` with a clear status message.

---

## 📁 Before & After Proof

- **Before:** Manual checking of fake accounts based on visual judgment.
- **After:** Automated fake account detection using trained ML model.
- **Evidence:** Screenshots of prediction results (Fake / Real outputs).

---

## 💡 Future Improvements

- Integration with Instagram API for real-time profile data fetching.
- Add graph-based visualization of fake/real account distribution.
- Deploy on Render or Hugging Face Spaces for live demo.

---

## 👨‍💻 Author

**Mohan Cheekatla**  
2025 B.Tech (Electronics & Communication Engineering) graduate 
Passionate about **Full-Stack Projects**, **Web Development**, and **AI Projects**.

---
