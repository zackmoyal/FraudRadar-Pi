import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from pi_python.pi_python import PiNetwork

# Define relative paths
base_dir = os.path.dirname(__file__)
data_file = os.path.join(base_dir, "data", "transactions.csv")
anomaly_file = os.path.join(base_dir, "data", "anomalous_transactions.csv")

# Initialize Pi SDK
api_key = "Enter Your API Key Here"
wallet_private_seed = "Enter Your Wallet Private Seed Here"
pi = PiNetwork()
pi.initialize(api_key, wallet_private_seed, "Pi Testnet")

# Load transaction data
def load_data():
    return pd.read_csv(data_file)

def load_anomalies():
    return pd.read_csv(anomaly_file)

# Streamlit UI Setup
st.set_page_config(page_title="FraudRadar Pi - Dashboard", layout="wide")
st.title("🚀 FraudRadar Pi - Transaction Dashboard")

# Load Data
data = load_data()
anomalies = load_anomalies()

# Sidebar Filters
st.sidebar.header("🏠 Filter Transactions")

# Date filter
if 'created_at' in data.columns:
    data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce', utc=True)
    start_date, end_date = st.sidebar.date_input("📅 Select Date Range", [data['created_at'].min().date(), data['created_at'].max().date()])
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    data = data[(data['created_at'] >= start_date) & (data['created_at'] <= end_date)]

# Fee filter
if 'fee_charged' in data.columns:
    min_fee, max_fee = data['fee_charged'].min(), data['fee_charged'].max()
    fee_range = st.sidebar.slider("💰 Select Fee Range", float(min_fee), float(max_fee), (float(min_fee), float(max_fee)))
    data = data[(data['fee_charged'] >= fee_range[0]) & (data['fee_charged'] <= fee_range[1])]

# Wallet filter
if 'source_account' in data.columns:
    unique_wallets = data['source_account'].unique()
    wallet_filter = st.sidebar.selectbox("🏦 Select Wallet ID", ["All"] + list(unique_wallets))
    if wallet_filter != "All":
        data = data[data['source_account'] == wallet_filter]

# Summary Statistics
st.sidebar.header("📊 Summary Statistics")
st.sidebar.write(f"**Total Transactions:** {len(data)}")
st.sidebar.write(f"**Total Anomalies:** {len(anomalies)}")
st.sidebar.write(f"**Average Fee:** {data['fee_charged'].mean() if not data.empty else 'N/A'}")

# Transaction Data Overview
st.subheader("📄 Transaction Data Overview")
if data.empty:
    st.warning("⚠️ No transactions match the selected filters.")
st.dataframe(data[['id', 'successful', 'hash', 'fee_charged']])

# Transaction Fee Distribution
st.subheader("📊 Transaction Fee Distribution")
if not data.empty:
    fig, ax = plt.subplots()
    data['fee_charged'].hist(bins=50, ax=ax)
    ax.set_xlabel("Fee Charged")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
else:
    st.warning("⚠️ No data available for fee distribution.")

# Anomalous Transactions
st.subheader("⚠️ Anomalous Transactions")
st.dataframe(anomalies[['id', 'successful', 'hash', 'fee_charged']])

# Daily Total Fee Charged
if 'created_at' in data.columns:
    st.subheader("📅 Daily Total Fee Charged")
    daily_fee = data.groupby(data['created_at'].dt.date)['fee_charged'].sum()
    st.line_chart(daily_fee) if not daily_fee.empty else st.warning("⚠️ No data available for daily fee analysis.")

# Example of Pi Payment Integration
st.subheader("🔗 Pi Payment Integration")
user_uid = "SAMPLE-USER-UID"  # Replace with actual user UID
if st.button("Create Payment"):
    payment_data = {
        "amount": 3.14,
        "memo": "Test - Greetings from MyApp",
        "metadata": {"product_id": "test-product"},
        "uid": user_uid
    }
    payment_id = pi.create_payment(payment_data)
    st.write(f"Payment Created! Payment ID: {payment_id}")

if st.button("Submit Payment"):
    payment_id = "PASTE_PAYMENT_ID_HERE"  # Replace with actual payment ID
    txid = pi.submit_payment(payment_id, False)
    st.write(f"Payment Submitted! Transaction ID: {txid}")

if st.button("Complete Payment"):
    payment_id = "PASTE_PAYMENT_ID_HERE"  # Replace with actual payment ID
    txid = "PASTE_TRANSACTION_ID_HERE"  # Replace with actual transaction ID
    payment = pi.complete_payment(payment_id, txid)
    st.write("Payment Completed!", payment)
