# 🟡 Gold Price Prediction using Machine Learning

## 📌 Overview

This project builds a time-series machine learning model to predict future gold prices using historical monthly data.

The model uses **past price patterns (lag features)** to estimate future values and demonstrates core ML concepts like feature engineering, time-based splitting, and regression.

---

## 📊 Dataset

* Source: Kaggle (Historical Gold Prices)
* Time Range: 2000 – 2026
* Frequency: Monthly
* Features Used:

  * Previous prices (lag_1 to lag_5)
  * Price differences (trend & momentum)
  * Rolling mean (local smoothing)

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Converted `Date` column to datetime
* Set date as index for time-series structure

### 2. Feature Engineering

* Lag features to capture memory
* Difference features for trend/momentum
* Rolling mean for smoothing

### 3. Train-Test Split

* Time-based split (80% train, 20% test)
* Avoids data leakage (important for time-series)

### 4. Model

* Linear Regression

---

## 📈 Results

* Evaluation Metric: RMSE (Root Mean Squared Error)
* RMSE ≈ **~100 USD**

👉 The model captures overall trends but struggles with:

* Sudden spikes
* Strong non-linear movements

---

## 📉 Visualization

The model compares:

* Actual gold prices
* Predicted gold prices

Output graph shows that predictions are smoother than real data.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/gold-price-prediction.git
   cd gold-price-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:

   ```bash
   python main.py
   ```

---

## 📁 Project Structure

```
gold-price-prediction/
│
├── gold_prices_1995_2026_feb.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

* Linear model cannot fully capture non-linear financial behavior
* No external factors included (inflation, interest rates, etc.)

---

## 🔮 Future Improvements

* Add non-linear models (Decision Tree, Random Forest)
* Implement LSTM for sequence learning
* Use external economic indicators
* Add interactive visualization (Plotly)

---

## 🧠 Key Learnings

* Time-series requires ordered data splitting
* Feature engineering is critical for model performance
* Avoiding data leakage is essential
* Linear models have limitations on financial data

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Jeevanandam S
