from flask import Flask, render_template, request, redirect, session, jsonify
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, time, timedelta
import pytz
import requests
from textblob import TextBlob
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import os
import random

app = Flask(__name__)
app.secret_key = "secret123"

NEWS_API_KEY = "a4c9743a64d347fa8e23a65271b65a5a"

# ---------------- DATABASE ----------------
def init_db():
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS info (
            user TEXT,
            name TEXT,
            email TEXT,
            mobile TEXT,
            password TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

# ---------------- MARKET STATUS ----------------
def get_market_status():
    india = pytz.timezone('Asia/Kolkata')
    now = datetime.now(india)

    open_time = now.replace(hour=9, minute=15, second=0)
    close_time = now.replace(hour=15, minute=30, second=0)

    if now < open_time:
        return "PRE-OPEN", "Opens at 09:15 AM"
    elif open_time <= now <= close_time:
        return "OPEN", "Closes at 03:30 PM"
    else:
        return "CLOSED", "Closed at 03:30 PM"

# ---------------- CLEAN DATA ----------------
def clean_stock_data(df):
    if df.empty:
        return df

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df:
        return pd.DataFrame()

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    df = df[df['Close'] > 0]

    return df

# ---------------- FILTER MARKET HOURS ----------------
def filter_intraday(df):
    if df.empty:
        return df

    india = pytz.timezone('Asia/Kolkata')

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    df.index = df.index.tz_convert(india)

    today = datetime.now(india).date()
    df = df[df.index.date == today]

    start = time(9, 15)
    end = time(15, 30)

    df = df[(df.index.time >= start) & (df.index.time <= end)]

    return df

# ---------------- LIVE DATA ----------------
@app.route('/live-data/<symbol>')
def live_data(symbol):
    try:
        data = yf.download(symbol, period="5d", interval="1m", progress=False)

        data = clean_stock_data(data)
        filtered = filter_intraday(data)

        if filtered.empty:
            data = data.tail(120)
        else:
            data = filtered

        if data.empty or 'Close' not in data:
            raise Exception("No data")

        times = data.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        prices = data["Close"].fillna(0).round(2).tolist()

        return jsonify({"time": times, "price": prices})

    except Exception as e:
        print("LIVE ERROR:", e)

        india = pytz.timezone('Asia/Kolkata')
        now = datetime.now(india)

        times = [
            (now - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(60)
        ][::-1]

        prices = [100 + random.uniform(-1, 1) for _ in range(60)]

        return jsonify({"time": times, "price": prices})

# ---------------- MODELS ----------------
def lstm_predict(prices, stock):
    try:
        if len(prices) < 60:
            return prices[-1]

        model_path = f"models/{stock}_lstm.h5"
        scaler_path = f"models/{stock}_scaler.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return prices[-1]

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        data = np.array(prices).reshape(-1, 1)
        scaled = scaler.transform(data)

        last_60 = scaled[-60:].reshape(1, 60, 1)
        pred = model.predict(last_60, verbose=0)

        return float(scaler.inverse_transform(pred)[0][0])

    except:
        return prices[-1]

def lr_predict(prices, stock):
    try:
        if len(prices) < 3:
            return prices[-1]

        model_path = f"models/{stock}_lr.pkl"
        if not os.path.exists(model_path):
            return prices[-1]

        model = joblib.load(model_path)

        X = pd.DataFrame([[prices[-1], prices[-2], prices[-3]]],
                         columns=['lag1', 'lag2', 'lag3'])

        return float(model.predict(X)[0])

    except:
        return prices[-1]

def arima_predict(prices):
    try:
        if len(prices) < 10:
            return prices[-1]

        model = ARIMA(prices, order=(5, 1, 0))
        model_fit = model.fit()
        return float(model_fit.forecast(1)[0])

    except:
        return prices[-1]

# ---------------- NEWS ----------------
def get_stock_news(stock):
    try:
        url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={NEWS_API_KEY}"
        res = requests.get(url).json()

        articles = res.get("articles", [])
        news_list = []
        scores = []

        for a in articles[:8]:
            title = a.get("title", "")
            link = a.get("url", "#")

            if title:
                news_list.append({"title": title, "link": link})
                scores.append(TextBlob(title).sentiment.polarity)

        score = round(sum(scores) / len(scores), 3) if scores else 0

        if score > 0.15:
            sentiment = "Positive 📈"
        elif score < -0.15:
            sentiment = "Negative 📉"
        else:
            sentiment = "Neutral ➖"

        return news_list, sentiment, score

    except:
        return [], "Neutral ➖", 0

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect('/signin')

    market_status, market_time = get_market_status()

    return render_template('index.html',
                           market_status=market_status,
                           market_time=market_time)

# ---------------- SIGNIN FIXED ----------------
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        user = request.form.get('username')
        password = request.form.get('password')

        # ✅ FIX
        user = user.strip()
        password = password.strip()

        con = sqlite3.connect('database.db')
        cur = con.cursor()

        cur.execute("SELECT * FROM info WHERE user=? AND password=?", (user, password))
        data = cur.fetchone()
        con.close()

        if data:
            session['user'] = user
            return redirect('/index')
        else:
            return render_template('signin.html', error="Invalid Credentials")

    return render_template('signin.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/signin')

# ---------------- SIGNUP FIXED ----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        user = request.form.get('username').strip()
        name = request.form.get('name')
        email = request.form.get('email')
        mobile = request.form.get('mobile')
        password = request.form.get('password').strip()

        try:
            con = sqlite3.connect('database.db')
            cur = con.cursor()

            cur.execute("SELECT * FROM info WHERE user=?", (user,))
            existing = cur.fetchone()

            if existing:
                con.close()
                return render_template('signup.html', error="User already exists")

            cur.execute(
                "INSERT INTO info (user, name, email, mobile, password) VALUES (?, ?, ?, ?, ?)",
                (user, name, email, mobile, password)
            )

            con.commit()
            con.close()

            return redirect('/signin')

        except Exception as e:
            print("SIGNUP ERROR:", e)
            return render_template('signup.html', error="Something went wrong")

    return render_template('signup.html')

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():

    if 'user' not in session:
        return redirect('/signin')

    try:
        stock = request.form.get('nm')

        if not stock:
            return redirect('/index')

        stock = stock.strip().upper()

        data = yf.download(stock, period="5d", interval="1m", progress=False)

        if data.empty:
            stock_alt = stock + ".NS"
            data = yf.download(stock_alt, period="5d", interval="1m", progress=False)
            if not data.empty:
                stock = stock_alt

        data = clean_stock_data(data)

        filtered = filter_intraday(data)
        data = filtered if not filtered.empty else data.tail(120)

        if data.empty or 'Close' not in data:
            raise Exception("No data")

        prices = data['Close'].fillna(0).tolist()
        timestamps = data.index.strftime("%Y-%m-%d %H:%M:%S").tolist()

        if len(prices) < 2:
            raise Exception("Not enough data")

        last_price = prices[-1]

        lstm_val = round(lstm_predict(prices, stock), 2)
        lr_val = round(lr_predict(prices, stock), 2)
        arima_val = round(arima_predict(prices), 2)

        final_val = round((lstm_val + lr_val + arima_val) / 3, 2)

        errors = {
            "ARIMA": abs(arima_val - last_price),
            "LSTM": abs(lstm_val - last_price),
            "Linear": abs(lr_val - last_price)
        }

        best_model = min(errors, key=errors.get)

        avg_error = np.mean(list(errors.values()))
        confidence = max(50, min(95, 100 - (avg_error / max(last_price, 1) * 100)))

        change = ((final_val - last_price) / last_price) * 100

        if change > 1:
            signal = "BUY"
        elif change < -1:
            signal = "SELL"
        else:
            signal = "HOLD"

        trend = "Uptrend" if prices[-1] > prices[0] else "Downtrend"

        buy = max(0, min(100, (change + 1) * 50))
        sell = 100 - buy

        risk = "Low" if confidence > 85 else "Medium" if confidence > 70 else "High"

        news, sentiment, sentiment_score = get_stock_news(stock)

        india = pytz.timezone('Asia/Kolkata')
        now = datetime.now(india)

        market_status, market_time = get_market_status()

        return render_template(
            'results.html',
            stock=stock,
            prices=prices,
            timestamps=timestamps,
            arima=arima_val,
            lstm=lstm_val,
            linear=lr_val,
            final=final_val,
            best_model=best_model,
            confidence=round(confidence, 2),
            signal=signal,
            trend=trend,
            move=round(change, 2),
            news=news,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            buy=round(buy, 2),
            sell=round(sell, 2),
            risk=risk,
            strength=round(abs(change), 2),
            market_status=market_status,
            market_time=market_time,
            current_time=now.strftime("%I:%M %p"),
            today_date=now.strftime("%d %B %Y")
        )

    except Exception as e:
        print("ERROR:", e)

        market_status, market_time = get_market_status()

        return render_template('index.html',
                               error="Something went wrong",
                               market_status=market_status,
                               market_time=market_time)

if __name__ == "__main__":
    app.run(debug=True)