<!-- Title with badges -->
<h1 align="center">🚀 FutureStockAI</h1>
<p align="center">
  <i>AI-Based Stock Market Prediction System 📈</i><br><br>
</p>

---

## 🌟 **About the Project**
**FutureStockAI** is a **comprehensive AI-powered web app** designed to:
- 🔍 Analyze real-time stock market data  
- 🤖 Predict stock prices using AI  
- 📊 Optimize investment portfolios  

Built using **Streamlit**, it merges **modern AI techniques** with **intuitive UI design** for both casual investors & professionals.

---

## ✨ **Key Features**
| Feature | Details |
|---------|---------|
| 🧠 AI-Driven Predictions | LSTM-based time-series forecasting |
| ⚡ Optimized Performance | 2-year data window, 50 epochs for balance |
| 💾 Data Caching | `@st.cache_data` & `@st.cache_resource` for faster loads |
| 📈 Live Tracking | Simulated real-time price movements |
| 📰 Sentiment Analysis | News headlines processed with NLTK VADER |
| 💼 Portfolio Optimization | Suggests stock weights + Sharpe Ratio |
| 🎨 Pro UI/UX | Sleek, responsive Streamlit dashboard |

---

## 🔐 **Authentication & Security**
- **Database:** SQLite (`users.db`)
- **Table:** `users`
- **Passwords:** Securely hashed via `bcrypt`  
> 🔒 *No plain-text storage. Your data is safe.*

---

## 🛠 **Getting Started**
**Prerequisite:** Python `3.8+` 🐍

---

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/Priyabshkamal/College_Project.git
cd FutureStockAI
```

---

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

📦 **Required Libraries:**
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

---

### 3️⃣ Run the Application
```bash
streamlit run app.py
```
💻 *The app will automatically open in your default browser!* 🌐

---

## 📂 **Project Structure**
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

## ⚙ **How It Works**

### 1️⃣ Authentication Flow
- **Sign Up:** Enter email & password → Stored in `users.db` (hashed)  
- **Login:** Session state activated → Dashboard unlocked  

### 2️⃣ Dashboard Tabs
- 📈 **Stock Prediction:** Enter ticker → AI predicts prices  
- 📊 **Live Stock Tracking:** Select ticker → Real-time chart updates  
- 📰 **Sentiment Analysis:** Fetch news sentiment for selected stock  
- 💼 **Portfolio Optimization:** Suggests stock weight allocation  

---

## 📝 **Important Notes**
- 📡 *Live market prices here are simulated (real APIs need licenses)*  
- 🔑 For news sentiment, get a free API key from [NewsAPI.org](https://newsapi.org/) and add it to `sentiment_analyzer.py`  

---

## 📸 **Screenshots**
<p align="center">
  <img src="https://via.placeholder.com/600x300?text=Dashboard+Preview" alt="Dashboard Screenshot" width="80%">
  <br><i>Sample Dashboard View</i>
</p>


---

<p align="center">💡 <b>FutureStockAI — Predict. Track. Optimize. Invest.</b> 💡</p>

