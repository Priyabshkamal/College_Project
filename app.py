import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import asyncio
import json
import time # For simulating real-time updates
import yfinance as yf # For fetching initial stock data
import mysql.connector # Use the MySQL connector library
import bcrypt # For password hashing
import os # For environment variables, which is a better practice

# --- Streamlit Page Configuration (MUST BE AT THE VERY TOP) ---
st.set_page_config(layout="wide", page_title="FutureStockAI")

# --- MySQL Database Configuration ---
# FIX: Use Streamlit Secrets for cloud deployment and a fallback for local testing
try:
    DATABASE_HOST = st.secrets["mysql_db"]["host"]
    DATABASE_USER = st.secrets["mysql_db"]["user"]
    DATABASE_PASSWORD = st.secrets["mysql_db"]["password"]
    DATABASE_NAME = st.secrets["mysql_db"]["database"]
except KeyError:
    # This block runs when you are running locally without a secrets.toml file
    # You should ensure your MySQL is running with these credentials
    DATABASE_HOST = os.getenv("MYSQL_HOST", "localhost")
    DATABASE_USER = os.getenv("MYSQL_USER", "root")
    DATABASE_PASSWORD = os.getenv("MYSQL_PASSWORD", "Test@0987")
    DATABASE_NAME = os.getenv("MYSQL_DB", "futurestockai")


def get_db_connection():
    """
    Establishes a connection to the MySQL database.
    Returns the connection object.
    """
    try:
        conn = mysql.connector.connect(
            host=DATABASE_HOST,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            database=DATABASE_NAME
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

def init_db():
    """
    Initializes the MySQL database and creates the users table with the correct schema.
    """
    try:
        conn = mysql.connector.connect(
            host=DATABASE_HOST,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD
        )
        cursor = conn.cursor()
        
        # Create the database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
        conn.database = DATABASE_NAME
        
        # Drop the table to ensure a clean start with the new schema
        # This will remove all existing users, so they will need to sign up again.
        cursor.execute("DROP TABLE IF EXISTS users")
        
        # Create the users table with the 'password' column for bcrypt hash
        cursor.execute('''
            CREATE TABLE users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        st.error(f"Error initializing MySQL database: {e}")

def add_user(username, email, password) -> bool:
    """
    Adds a new user to the database after hashing the password.
    """
    conn = get_db_connection()
    if not conn: return False
    
    try:
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # The INSERT query is updated to use the 'password' column
        query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, email, hashed_password.decode('utf-8')))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        if e.errno == 1062: # 1062 is the error code for duplicate entry
            st.error("Username or Email already exists. Please choose a different one.")
        else:
            st.error(f"Error adding user: {e}")
        conn.close()
        return False

def verify_user(email, password) -> bool:
    """
    Verifies user credentials against the database.
    """
    conn = get_db_connection()
    if not conn: return False
    
    try:
        cursor = conn.cursor()
        # The SELECT query is updated to use the 'password' column
        query = "SELECT password FROM users WHERE email = %s"
        cursor.execute(query, (email,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()

        if result:
            stored_password_hash = result[0].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
                return True
        return False
    except mysql.connector.Error as e:
        st.error(f"Error verifying user: {e}")
        conn.close()
        return False


# Import data_utils and model_trainer outside the main flow
from data_utils import get_stock_data, prepare_data_for_lstm, create_dataset
from model_trainer import build_and_train_model, predict_future_prices
from sentiment_analyzer import analyze_sentiment, get_sentiment_label, fetch_news_headlines, NEWSAPI_KEY
from portfolio_optimizer import get_historical_prices_for_portfolio, optimize_portfolio, get_portfolio_performance


# --- Streamlit Page Configuration (MUST BE AT THE VERY TOP) ---
st.set_page_config(layout="wide", page_title="FutureStockAI")

# Initialize MySQL database for users (run once at app start)
init_db()

# --- Streamlit Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None # Will be set after login
if 'current_optimized_portfolio' not in st.session_state: # Only store current, not saved list
    st.session_state.current_optimized_portfolio = None
if 'live_chart_data' not in st.session_state:
    # Initialize live_chart_data with explicit dtypes to prevent future warnings
    st.session_state.live_chart_data = pd.DataFrame(columns=['Time', 'Price'], dtype=object)
if 'current_live_ticker' not in st.session_state:
    st.session_state.current_live_ticker = None
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df = None


# --- User Authentication Functions ---
def login_page():
    st.subheader("Login to FutureStockAI")
    email_input = st.text_input("Email", key="login_email_input")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_button"):
        if verify_user(email_input, password): # Verify using email and password
            st.session_state.logged_in = True
            st.session_state.username = email_input # Display email as username
            st.session_state.user_id = email_input # Use email as user_id for MySQL
            st.success(f"Welcome, {email_input}!")
            st.rerun() # Re-added st.rerun() to force dashboard load
        else:
            st.error("Invalid email or password.")

def signup_page():
    st.subheader("Create a New Account")
    new_username = st.text_input("Choose a Username", key="signup_username")
    new_email = st.text_input("Email", key="signup_email")
    new_password = st.text_input("Choose a Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")

    if st.button("Sign Up", key="signup_button"):
        if new_username and new_email and new_password and confirm_password:
            if new_password == confirm_password:
                if add_user(new_username, new_email, new_password):
                    st.success("Account created successfully! You can now login.")
            else:
                st.error("Passwords do not match.")
        else:
            st.warning("Please fill in all required fields (Username, Email, Password).")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.current_optimized_portfolio = None # Clear current portfolio on logout
    st.session_state.live_chart_data = pd.DataFrame(columns=['Time', 'Price'], dtype=object) # Clear live data on logout
    st.session_state.current_live_ticker = None
    st.session_state.prediction_df = None # Clear prediction data on logout
    st.success("Logged out successfully.")
    st.rerun()


# --- Main App Flow ---
if not st.session_state.logged_in:
    st.sidebar.title("Authentication")
    auth_option = st.sidebar.radio("Select an option:", ["Login", "Sign Up"], key="auth_option_radio")
    st.markdown("---")

    if auth_option == "Login":
        login_page()
    else:
        signup_page()
else:
    st.sidebar.info(f"**Logged in as:** `{st.session_state.username}`")
    st.sidebar.button("Logout", on_click=logout, key="logout_button_sidebar")
    st.sidebar.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Stock Prediction", "Sentiment Analysis", "Portfolio Suggestion", "Live Stock Tracking"])

    def run_stock_prediction(placeholder):
        """Handles the logic for running the stock prediction model."""
        with placeholder.container():
            ticker_symbol_pred = st.session_state.sidebar_ticker_pred
            num_prediction_days = st.session_state.num_prediction_days_pred_slider
            
            if not ticker_symbol_pred:
                st.error("Please enter a stock ticker symbol for prediction.")
                return

            st.subheader(f"Data and Prediction for {ticker_symbol_pred}")

            with st.spinner(f"Fetching historical data for '{ticker_symbol_pred}'..."):
                today = datetime.date.today()
                end_date = today
                start_date = today - datetime.timedelta(days=365 * 2)
                stock_data = get_stock_data(ticker_symbol_pred, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            if stock_data is not None and not stock_data.empty:
                latest_close_price = stock_data['Close'].iloc[-1].item()
                latest_close_date = stock_data.index[-1].strftime('%Y-%m-%d')
                st.info(f"**Latest Closing Price ({latest_close_date}):** ${latest_close_price:.2f}")
                st.markdown("*(Prediction model uses data up to this point.)*")

                with st.spinner("Preparing data..."):
                    scaled_data, scaler, last_60_days_for_prediction = prepare_data_for_lstm(stock_data, look_back=60)
                    training_data_length = int(len(scaled_data) * 0.8)
                    train_data_scaled = scaled_data[0:training_data_length, :]
                    X_train, y_train = create_dataset(train_data_scaled, look_back=60)

                with st.spinner("Building and training model..."):
                    model = build_and_train_model(X_train, y_train, epochs=50, batch_size=32)

                st.success("Stock Price Predicted Successfully!")

                with st.spinner(f"Generating predictions for the next {num_prediction_days} days..."):
                    predicted_future_prices = predict_future_prices(model, last_60_days_for_prediction, scaler, num_prediction_days)

                last_actual_date = stock_data.index[-1]
                prediction_dates = [last_actual_date + datetime.timedelta(days=i) for i in range(1, num_prediction_days + 1)]

                st.session_state.prediction_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Predicted Close': predicted_future_prices.flatten()
                })
                st.session_state.prediction_df.set_index('Date', inplace=True)
                st.session_state.last_stock_data_for_pred = stock_data
            else:
                st.error(f"Could not retrieve historical data for '{ticker_symbol_pred}'. Please check the ticker symbol or try a different date range.")
                st.session_state.prediction_df = None
                st.session_state.last_stock_data_for_pred = pd.DataFrame()

    with tab1:
        st.header("📊 Stock Price Prediction")
        st.markdown("Enter a stock ticker to view historical data and predict future prices using an LSTM model.")

        st.button("Run Stock Prediction", on_click=run_stock_prediction, args=(st.empty(),), key="run_stock_pred")

        st.sidebar.subheader("Stock Prediction Parameters")
        st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", "AAPL", key="sidebar_ticker_pred").upper()
        num_prediction_days = st.sidebar.slider("Number of days to predict?", 1, 30, 7, key="num_prediction_days_pred_slider")

        if st.session_state.prediction_df is not None and not st.session_state.prediction_df.empty:
            st.write("---")
            st.subheader(f"Price Prediction for {st.session_state.sidebar_ticker_pred} (Next {num_prediction_days} Days)")

            recent_actual_prices = st.session_state.last_stock_data_for_pred['Close'].tail(60).values if not st.session_state.last_stock_data_for_pred.empty else np.array([])
            recent_actual_dates = st.session_state.last_stock_data_for_pred.index[-60:] if not st.session_state.last_stock_data_for_pred.empty else pd.DatetimeIndex([])

            if len(recent_actual_prices) > 0 and len(st.session_state.prediction_df) > 0:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=recent_actual_dates, y=recent_actual_prices.flatten(),
                                             mode='lines', name='Actual Prices',
                                             line=dict(color='blue', width=2)))
                fig_pred.add_trace(go.Scatter(x=st.session_state.prediction_df.index, y=st.session_state.prediction_df['Predicted Close'].values.flatten(),
                                             mode='lines', name='Predicted Prices',
                                             line=dict(color='red', dash='dot', width=2)))

                all_prices = np.concatenate((recent_actual_prices.flatten(), st.session_state.prediction_df['Predicted Close'].values.flatten()))
                min_all_price = np.min(all_prices)
                max_all_price = np.max(all_prices)

                fig_pred.update_layout(
                    title_text=f'{st.session_state.sidebar_ticker_pred} Price Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=500,
                    template="plotly_dark",
                    hovermode="x unified",
                    yaxis_range=[min_all_price * 0.95, max_all_price * 1.05]
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.warning("Not enough data to display prediction chart. Please click 'Run Stock Prediction'.")

            st.write("---")
            st.subheader("Predicted Prices Table")
            st.dataframe(st.session_state.prediction_df.style.format({"Predicted Close": "{:.2f}"}), use_container_width=True)

            csv_data = st.session_state.prediction_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Prediction Data as CSV",
                data=csv_data,
                file_name=f"{st.session_state.sidebar_ticker_pred}_predictions.csv",
                mime="text/csv",
                key="download_predictions_csv"
            )
        else:
            st.info("Click 'Run Stock Prediction' to generate and view predictions.")

    with tab2:
        st.header("💬 Sentiment Analysis")
        st.markdown("Get real-time market sentiment by analyzing recent news headlines for a stock.")

        if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
            st.warning("⚠️ **NewsAPI Key Missing!** Please get your free API key from [NewsAPI.org](https://newsapi.org/) and replace 'YOUR_NEWSAPI_KEY' in `sentiment_analyzer.py` to enable this feature.")
        else:
            sentiment_ticker_input = st.text_input("Enter Stock Ticker for News Sentiment (e.g., TSLA, MSFT):", "TSLA", key="sentiment_ticker_input").upper()
            num_news_days_back = st.slider("Look back for news (days):", 1, 30, 7, key="num_news_days_back")

            if st.button("Fetch & Analyze News Sentiment", key="fetch_analyze_sentiment_btn"):
                if sentiment_ticker_input:
                    with st.spinner(f"Fetching news for '{sentiment_ticker_input}' and analyzing sentiment..."):
                        articles = fetch_news_headlines(sentiment_ticker_input, NEWSAPI_KEY, num_news_days_back)

                        if articles:
                            st.subheader(f"Recent News Headlines for {sentiment_ticker_input}")
                            all_compound_scores = []

                            for i, article in enumerate(articles[:10]):
                                title = article.get('title', 'No Title')
                                description = article.get('description', 'No Description')
                                url = article.get('url', '#')
                                source = article.get('source', {}).get('name', 'Unknown Source')
                                published_at = article.get('publishedAt', 'N/A')

                                sentiment_scores = analyze_sentiment(title + " " + (description if description else ""))
                                compound_score = sentiment_scores['compound']
                                sentiment_label = get_sentiment_label(compound_score)
                                all_compound_scores.append(compound_score)

                                st.markdown(f"**[{title}]({url})**")
                                st.write(f"Source: {source} | Published: {published_at}")
                                st.write(f"Sentiment: **{sentiment_label}** (Compound: {compound_score:.2f})")
                                st.markdown(f"*{description}*")
                                st.markdown("---")

                            if all_compound_scores:
                                average_compound_score = np.mean(all_compound_scores)
                                overall_sentiment_label = get_sentiment_label(average_compound_score)
                                st.subheader("Overall Sentiment Summary")
                                st.write(f"**Average Compound Score:** {average_compound_score:.2f}%")
                                st.write(f"**Overall Market Sentiment:** {overall_sentiment_label}")

                                if overall_sentiment_label == "Positive":
                                    st.success("😊 The overall news sentiment for this stock is positive.")
                                elif overall_sentiment_label == "Negative":
                                    st.error("😠 The overall news sentiment for this stock is negative.")
                                else:
                                    st.info("😐 The overall news sentiment for this stock is neutral.")
                            else:
                                st.info(f"No headlines found for '{sentiment_ticker_input}' in the last {num_news_days_back} days or unable to analyze.")
                        else:
                            st.info(f"No news headlines found for '{sentiment_ticker_input}' in the last {num_news_days_back} days.")
                else:
                    st.warning("Please enter a stock ticker for news sentiment analysis.")

    with tab3:
        st.header("💰 Portfolio Suggestion (Optimized)")
        st.markdown("Enter a list of stocks you are interested in, and we will suggest optimal weights to maximize returns for a given risk level.")

        portfolio_tickers_input = st.text_input(
            "Enter stock tickers (comma-separated, e.g., AAPL, MSFT, GOOGL, AMZN):",
            "AAPL, MSFT, GOOGL",
            key="portfolio_tickers_input"
        )
        portfolio_tickers_list = [ticker.strip().upper() for ticker in portfolio_tickers_input.split(',') if ticker.strip()]

        risk_tolerance_level = st.slider(
            "Your Risk Tolerance (1 = Very Low Risk, 10 = Very High Risk):",
            1, 10, 5, key="risk_tolerance_level_slider"
        )

        col_optimize, col_save = st.columns([0.7, 0.3])

        with col_optimize:
            if st.button("Optimize Portfolio", key="optimize_portfolio_btn"):
                if not portfolio_tickers_list or len(portfolio_tickers_list) < 2:
                    st.warning("Please enter at least two stock tickers to optimize a portfolio.")
                else:
                    portfolio_end_date = datetime.date.today()
                    portfolio_start_date = portfolio_end_date - datetime.timedelta(days=365 * 3)
                    prices_df = get_historical_prices_for_portfolio(
                        portfolio_tickers_list,
                        portfolio_start_date.strftime('%Y-%m-%d'),
                        portfolio_end_date.strftime('%Y-%m-%d')
                    )

                    if prices_df is not None and not prices_df.empty:
                        available_tickers = prices_df.columns.tolist()
                        if len(available_tickers) < len(portfolio_tickers_list):
                            st.warning(f"Could not fetch data for some tickers. Optimizing with: {', '.join(available_tickers)}")
                        if len(available_tickers) < 2:
                            st.error("Not enough valid tickers with historical data to perform optimization. Please check the tickers.")
                        else:
                            optimized_weights = optimize_portfolio(prices_df, risk_aversion=risk_tolerance_level)

                            if optimized_weights is not None and not optimized_weights.empty:
                                st.session_state.current_optimized_portfolio = {
                                    'weights': optimized_weights,
                                    'performance': get_portfolio_performance(optimized_weights, prices_df),
                                    'tickers': portfolio_tickers_list,
                                    'risk_tolerance': risk_tolerance_level
                                }
                                st.success("Portfolio optimized!")
                            else:
                                st.warning("Portfolio optimization failed. Check your ticker symbols or try different ones.")
                    else:
                        st.error("Could not retrieve sufficient historical data for portfolio optimization. Please check the tickers.")

        if 'current_optimized_portfolio' in st.session_state and st.session_state.current_optimized_portfolio:
            current_weights = st.session_state.current_optimized_portfolio['weights']
            current_performance = st.session_state.current_optimized_portfolio['performance']
            current_tickers = st.session_state.current_optimized_portfolio['tickers']
            current_risk_tolerance = st.session_state.current_optimized_portfolio['risk_tolerance']

            st.write("---")
            st.subheader(f"Current Optimized Portfolio for {current_tickers} (Risk: {current_risk_tolerance}/10)")

            if current_weights is not None:
                if isinstance(current_weights, pd.Series):
                    weights_dict_for_display = current_weights.to_dict()
                elif isinstance(current_weights, dict):
                    weights_dict_for_display = current_weights
                else:
                    st.warning("Optimized weights are in an unexpected format. Cannot display.")
                    weights_dict_for_display = {}

                if weights_dict_for_display:
                    weights_df = pd.DataFrame(list(weights_dict_for_display.items()), columns=['Ticker', 'Weight'])
                    weights_df['Weight (%)'] = (weights_df['Weight'] * 100).round(2)
                    st.dataframe(weights_df[['Ticker', 'Weight (%)']], use_container_width=True)
                else:
                    st.info("No optimized portfolio weights to display.")

                if current_performance:
                    expected_return, annual_volatility, sharpe_ratio = current_performance
                    st.subheader("Performance Metrics (Annualized)")
                    st.markdown(f"**Expected Annual Return:** {expected_return * 100:.2f}%")
                    st.markdown(f"**Annual Volatility (Risk):** {annual_volatility * 100:.2f}%")
                    st.markdown(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
                    st.info("Higher Sharpe Ratio indicates better risk-adjusted returns.")
                else:
                    st.warning("Could not calculate portfolio performance for the current optimized portfolio.")

            else:
                st.info("Optimize a portfolio first to see its details.")
        else:
            st.info("Optimize a portfolio to see its details.")

    with tab4:
        st.header("📈 Live Stock Tracking")
        st.markdown("Track real-time (simulated) stock prices and visualize movements for any stock ticker.")


        col_select, col_refresh_rate = st.columns([0.7, 0.3])

        with col_select:
            # Dropdown for popular tickers
            default_tickers = [
                "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "GOOG", "NFLX",
                "BRK-B", "JPM", "V", "PG", "JNJ", "UNH", "XOM", "CVX",
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "ITC.NS",
                "TATASTEEL.NS", "LT.NS", "MARUTI.NS", "ASIANPAINT.NS", "ADANIENT.NS",
                "HSBC", "SAP", "TM", "SNEJF", "BABA", "TCEHY", "NSRGY", "RY.TO",
                "SPY", "QQQ", "DIA", "GLD", "SLV"
            ]
            default_tickers.sort()

            selected_ticker = st.selectbox(
                "Select a stock ticker:",
                options=default_tickers,
                index=default_tickers.index(st.session_state.current_live_ticker) if st.session_state.current_live_ticker in default_tickers else 0,
                key="live_ticker_select"
            )
            
            final_ticker = selected_ticker

        with col_refresh_rate:
            refresh_interval = st.slider("Refresh Interval (seconds):", 1, 10, 5, key="refresh_interval_slider")

        if st.button("Start Live Tracking", key="start_live_tracking_btn"):
            if not final_ticker:
                st.warning("Please select a stock ticker to start tracking.")
            else:
                st.session_state.current_live_ticker = final_ticker
                st.session_state.live_chart_data = pd.DataFrame(columns=['Time', 'Price'], dtype=object)

                st.subheader(f"Live Price for {final_ticker}")
                price_metric_placeholder = st.empty()
                chart_placeholder = st.empty()

                try:
                    initial_data = yf.download(final_ticker, period="2d", interval="1m", auto_adjust=True)
                    if not initial_data.empty:
                        last_price = initial_data['Close'].iloc[-1].item()
                        if initial_data.index.tz is not None:
                            initial_data.index = initial_data.index.tz_localize(None)

                        new_initial_data = pd.DataFrame({
                            'Time': initial_data.index,
                            'Price': initial_data['Close'].values.flatten()
                        })
                        st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_initial_data], ignore_index=True)

                        if len(st.session_state.live_chart_data) > 100:
                            st.session_state.live_chart_data = st.session_state.live_chart_data.tail(100).reset_index(drop=True)
                    else:
                        last_price = 100.0
                        st.warning(f"Could not retrieve initial data for '{final_ticker}'. Starting with a default price.")
                except Exception as e:
                    last_price = 100.0
                    st.error(f"Error retrieving initial data for '{final_ticker}': {e}. Starting with a default price.")
                    st.warning("Please note: Live data is simulated. For real-time data, a dedicated API key and backend are needed.")

                while True:
                    current_time = datetime.datetime.now()
                    new_price = last_price

                    try:
                        latest_data = yf.download(final_ticker, period="1d", interval="1m", auto_adjust=True)
                        if not latest_data.empty:
                            if latest_data.index.tz is not None:
                                latest_data.index = latest_data.index.tz_localize(None)

                            if not st.session_state.live_chart_data.empty and latest_data.index[-1] > st.session_state.live_chart_data['Time'].max():
                                new_price = latest_data['Close'].iloc[-1].item()
                                new_data_point = pd.DataFrame([{'Time': latest_data.index[-1], 'Price': new_price}])
                                st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_data_point], ignore_index=True)
                            else:
                                price_change = np.random.uniform(-0.5, 0.5)
                                new_price = last_price + price_change
                                if new_price < 0: new_price = 0.01
                                new_data_point = pd.DataFrame([{'Time': current_time, 'Price': new_price}])
                                st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_data_point], ignore_index=True)
                        else:
                            price_change = np.random.uniform(-0.5, 0.5)
                            new_price = last_price + price_change
                            if new_price < 0: new_price = 0.01
                            new_data_point = pd.DataFrame([{'Time': current_time, 'Price': new_price}])
                            st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_data_point], ignore_index=True)

                    except Exception as e:
                        st.warning(f"Error retrieving latest data for '{final_ticker}': {e}. Simulating price.")
                        price_change = np.random.uniform(-0.5, 0.5)
                        new_price = last_price + price_change
                        if new_price < 0: new_price = 0.01
                        new_data_point = pd.DataFrame([{'Time': current_time, 'Price': new_price}])
                        st.session_state.live_chart_data = pd.concat([st.session_state.live_chart_data, new_data_point], ignore_index=True)

                    price_metric_placeholder.metric(label=f"Current Price ({final_ticker})", value=f"${new_price:.2f}", delta=f"{new_price - last_price:.2f}")

                    if len(st.session_state.live_chart_data) > 100:
                        st.session_state.live_chart_data = st.session_state.live_chart_data.tail(100).reset_index(drop=True)

                    chart_placeholder.line_chart(st.session_state.live_chart_data.set_index('Time'))

                    last_price = new_price
                    time.sleep(refresh_interval)
