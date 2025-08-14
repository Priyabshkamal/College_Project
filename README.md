# 🚀 **FutureStockAI**: *AI-Powered Stock Market Prediction System*

> 💹 *Real-time Market Analysis • Stock Price Forecasting • Data-Driven Investment Decisions*

---

## 🌟 **About the Project**
**FutureStockAI** is a **comprehensive AI-driven web application** built with [Streamlit](https://streamlit.io/), designed to empower users with:
- 📈 Real-time market insights  
- 🤖 AI-powered stock price forecasting  
- 📊 Data-backed portfolio optimization  

It demonstrates **modern data science libraries** + **full-stack development principles** for a **seamless user experience**.  

---

## ✨ **Key Features & Technical Highlights**

### 🧠 **Optimized AI-Driven Prediction Engine**
- ⚡ Uses **LSTM Deep Learning Model** for time-series forecasting.
- 📉 **Performance-Optimized:**  
  - Historical data window reduced to **2 years**  
  - Training epochs limited to **20**  
  - Balances **accuracy vs. speed**  

### 💾 **Data Caching**
- Utilizes `@st.cache_data` & `@st.cache_resource` to:  
  - 🚀 Speed up repeated loads  
  - 🗄 Cache **data** + **trained model**

### 📊 **Enhanced Real-Time Simulation**
- "Live Stock Tracking" with **interactive price charts**  
- Uses **yfinance delayed intraday data** + stochastic simulation for **realistic market feel**  

### 📰 **Sentiment Analysis**
- Fetches latest stock news 🗞  
- Analyzes headlines with **NLTK VADER** to show:  
  - ✅ Positive  
  - ⚠️ Negative  
  - ➖ Neutral sentiment  

### 📈 **Portfolio Optimization**
- Calculates **optimal stock weights** for given risk tolerance  
- Outputs 📊 **Expected Annual Return** + **Sharpe Ratio**  

### 🎨 **Professional UI/UX**
- Clean, responsive Streamlit interface  
- **User-friendly dashboard navigation**  

---

## 🔐 **Authentication & Data Storage**
- **Secure Login System** 🔑  
- **Database:** SQLite (`users.db`)  
- **Table:** `users`  
- **Password Security:** Hashed with `bcrypt` (no plain text!)  

---

## 🛠 **Getting Started**

### **Prerequisites**
- Python `3.8+` 🐍

---

### **1️⃣ Clone the Repository**
```bash
git clone [your_repository_url_here]
cd FutureStockAI
