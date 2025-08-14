<!-- Title with badges -->
<h1 align="center">🚀 FutureStockAI</h1>
<p align="center">
  <i>AI-Based Stock Market Prediction System 📈</i><br><br>
  <img src="https://img.shields.io/github/stars/YourUsername/FutureStockAI?color=yellow&style=for-the-badge">
  <img src="https://img.shields.io/github/forks/YourUsername/FutureStockAI?color=brightgreen&style=for-the-badge">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit">
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
| ⚡ Optimized Performance | 2-year data window, 20 epochs for balance |
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

### 1️⃣ Clone the Repo
```bash
git clone [your_repository_url_here]
cd FutureStockAI

### 1️⃣ Install Dependencies
pip install -r requirements.txt
### Libraries Used:
streamlit, pandas, numpy, plotly, yfinance,
bcrypt, ta, tensorflow, nltk


