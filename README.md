<!-- Animated Banner -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=F70000&center=true&vCenter=true&width=800&lines=FutureStockAI+-+Stock+Market+Prediction;AI+Powered+Investment+Insights;Predict+Track+Optimize+Invest" alt="Typing SVG" />
</p>

---

## 🌟 About the Project
**FutureStockAI** is your **AI-powered crystal ball for the stock market** 🪄📈  
It brings **real-time insights**, **LSTM predictions**, **sentiment analysis**, and **portfolio optimization** into a **sleek, interactive dashboard**.  
Perfect for **casual investors** 💼 and **market pros** 💹.

---

## ✨ Key Features
| 🚀 Feature | 💡 Details |
|------------|------------|
| 🧠 **AI-Driven Predictions** | LSTM-based time-series forecasting |
| ⚡ **Optimized Performance** | 2-year data window, 20 epochs for balance |
| 💾 **Data Caching** | `@st.cache_data` & `@st.cache_resource` |
| 📈 **Live Tracking** | Simulated real-time price movements |
| 📰 **Sentiment Analysis** | News headlines processed with NLTK VADER |
| 💼 **Portfolio Optimization** | Suggests stock weights + Sharpe Ratio |
| 🎨 **Pro UI/UX** | Responsive, interactive Streamlit dashboard |

---

## 🔐 Authentication & Security
- **Database:** SQLite (`users.db`)
- **Passwords:** Securely hashed via `bcrypt`  
- 🔒 *No plain-text passwords — your data is safe.*

---

## 🛠 Getting Started
**Prerequisite:** Python `3.8+` 🐍

```bash
# 1️⃣ Clone the repo
git clone https://github.com/Priyabshkamal/College_Project.git
cd FutureStockAI

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the app
streamlit run app.py
```

📦 **Libraries Used:**  
`streamlit`, `pandas`, `numpy`, `plotly`, `yfinance`, `bcrypt`, `ta`, `tensorflow`, `nltk`

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
1️⃣ **Authentication Flow**  
   - Sign Up → Email & Password stored (hashed)  
   - Login → Unlocks personalized dashboard  

2️⃣ **Dashboard Tabs**  
   - 📈 Stock Prediction: Enter ticker → AI predicts prices  
   - 📊 Live Tracking: Real-time chart simulation  
   - 📰 Sentiment Analysis: NewsAPI + VADER analysis  
   - 💼 Portfolio Optimization: Allocations & Sharpe Ratio  

---

## 📸 Screenshots
<p align="center">
  <img width="1097" height="653" src="https://github.com/user-attachments/assets/3ed5fe8e-abd4-4456-9e0c-21d8b5f3db09" alt="Dashboard Screenshot" />
  <br><i>✨ Sleek AI-powered stock dashboard ✨</i>
</p>

---

## 📝 Notes
- 📡 Market prices are **simulated** for demo purposes.  
- 🔑 For news sentiment, get your API key from [NewsAPI.org](https://newsapi.org/).  

---

<!-- Cool Stock GIF at the Bottom -->
<p align="center">
  <img src="https://media.giphy.com/media/26tn33aiTi1jkl6H6/giphy.gif" width="400px" alt="Stock Market Animation">
  <br>💡 <b>FutureStockAI — Predict • Track • Optimize • Invest</b> 💡
</p>

