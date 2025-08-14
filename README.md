<!-- Permanent Project Name with Glow -->

<h1 align="center">
  <span style="color:#F70000; text-shadow: 0 0 10px #F70000, 0 0 20px #FF0000, 0 0 30px #FF4500;">
    🚀 FutureStockAI: AI-Based Stock Market Prediction System
</h1>

---

<!-- Animated Features -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=26&pause=1500&color=FFD700&center=true&vCenter=true&width=800&lines=Stock+Prediction;Sentiment+Analysis;Portfolio+Suggestion;Live+Stock+Tracking" alt="Typing Animation" />
</p>


---

## 🌟 About the Project

🚀 **FutureStockAI** is a **AI-powered stock market prediction platform** that blends advanced machine learning with sleek, interactive UI design.  
It’s not just another finance app — it’s your **personal AI market analyst**, capable of:

- 💹 **Stock Prediction** — Accurate price forecasting using **LSTM deep learning models**  
- 📰 **Sentiment Analysis** — Understand market mood via **real-time news scanning & VADER NLP**  
- 📊 **Portfolio Suggestion** — Get **optimized investment allocations** for maximum returns  
- 📈 **Live Stock Tracking** — Monitor simulated real-time price changes & market movements  

Built with **Streamlit** for an elegant dashboard, **TensorFlow** for powerful AI models, and **NLTK** for language sentiment processing, **FutureStockAI** is tailored for both **casual investors** and **serious traders**.  

🔥 *Think of it as your AI co-pilot for smarter, faster, and data-driven investing decisions.*


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
  <br>💡 <b>FutureStockAI — AI-Based Stock Market Prediction System </b> 💡
</p>






