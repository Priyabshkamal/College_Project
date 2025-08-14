<!-- Banner -->
<p align="center">
</p>

<h1 align="center">🚀 FutureStockAI — <i>AI-Based Stock Market Prediction System</i> 📈</h1>

---

## 🌟 About the Project
**FutureStockAI** is a **comprehensive AI-powered web app** designed to:
- 🔍 Analyze real-time stock market data  
- 🤖 Predict stock prices using AI  
- 📊 Optimize investment portfolios  

Built with **Streamlit**, combining **modern AI techniques** with **intuitive UI design** for both casual investors & professionals.

---

## ✨ Key Features
| 🚀 Feature | 💡 Details |
|------------|------------|
| 🧠 **AI-Driven Predictions** | LSTM-based time-series forecasting |
| ⚡ **Optimized Performance** | 2-year data window, 20 epochs for balance |
| 💾 **Data Caching** | `@st.cache_data` & `@st.cache_resource` for faster loads |
| 📈 **Live Tracking** | Simulated real-time price movements |
| 📰 **Sentiment Analysis** | News headlines processed with NLTK VADER |
| 💼 **Portfolio Optimization** | Suggests stock weights + Sharpe Ratio |
| 🎨 **Pro UI/UX** | Sleek, responsive Streamlit dashboard |

---

## 🔐 Authentication & Security
- **Database:** SQLite (`users.db`)
- **Table:** `users`
- **Passwords:** Securely hashed via `bcrypt`  
> 🔒 *No plain-text storage. Your data is safe.*

---

## 🛠 Getting Started
**Prerequisite:** Python `3.8+` 🐍

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/Priyabshkamal/College_Project.git
cd FutureStockAI
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
📦 **Libraries Used:**
```
streamlit
pandas
numpy
plotly
yfinance
bcrypt
ta
tensorflow
nltk
```

### 3️⃣ Run the Application
```bash
streamlit run app.py
```
💻 *Opens automatically in your default browser!*

---

## 📂 Project Structure
```plaintext
app.py                 # Main app controller
db_utils.py            # Authentication & DB logic
data_utils.py          # Data fetching & preprocessing
model_trainer.py       # LSTM model training & prediction
sentiment_analyzer.py  # News sentiment analysis
portfolio_optimizer.py # Portfolio optimization logic
requirements.txt       # Dependencies
```

---

## ⚙ How It Works
### 1️⃣ Authentication Flow
- Sign Up → Email & Password stored in `users.db` (hashed)
- Login → Session state activated → Dashboard unlocked

### 2️⃣ Dashboard Tabs
- 📈 Stock Prediction: Enter ticker → AI predicts prices
- 📊 Live Stock Tracking: Select ticker → Real-time chart updates
- 📰 Sentiment Analysis: Fetch news sentiment for selected stock
- 💼 Portfolio Optimization: Suggests stock weight allocation

---

## 📸 Screenshots & Previews
<p align="center">
  <br><i><b>Sample Dashboard View</b></i>
  
  <img width="1097" height="653" alt="image" src="https://github.com/user-attachments/assets/3ed5fe8e-abd4-4456-9e0c-21d8b5f3db09" />
</p>

---

## 📝 Notes
- 📡 *Live market prices are simulated (real APIs need licenses)*  
- 🔑 For sentiment analysis, get an API key from [NewsAPI.org](https://newsapi.org/) and add it to `sentiment_analyzer.py`

---

<p align="center">
  <p align="center">
  <img src="https://via.placeholder.com/300x150?text=Stock+Chart+Animation" alt="Stock Market Simulation" width="300px"><br>
    
  💡 <b>FutureStockAI — AI-Based Stock Market Prediction System</b> 💡
</p>







