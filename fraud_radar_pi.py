import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os  # For relative paths and directory creation
import logging  # For logging

# Setup logging
def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
    log_file = os.path.join(log_dir, "fraud_detection.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return log_file

log_file = setup_logging()
logging.info("Logging setup complete. Logs will be saved to '%s'.", log_file)

# Define relative file paths
base_dir = os.path.dirname(__file__)  # The directory where the script is located
data_file = os.path.join(base_dir, "data", "transactions.csv")
save_file = os.path.join(base_dir, "data", "anomalous_transactions.csv")
visualizations_dir = os.path.join(base_dir, "visualizations")
os.makedirs(visualizations_dir, exist_ok=True)  # Ensure visualizations directory exists

# Load the transaction data
try:
    data = pd.read_csv(data_file)
    logging.info("Transaction data loaded successfully.")
except Exception as e:
    logging.error("Error loading transaction data: %s", e)
    raise

# Display the first few rows and column names
print("Data Preview:")
print(data.head())
logging.info("Data Columns: %s", data.columns.tolist())
if len(data.columns) < 20:  # Limit output if too many columns
    print("Column Names:", data.columns.tolist())

# Check unique values in 'fee_charged'
if 'fee_charged' in data.columns:
    unique_fees = data['fee_charged'].unique()
    print("Unique values in 'fee_charged':", unique_fees)
    logging.info("Unique values in 'fee_charged': %s", unique_fees)

# Rule-based checks for low fees and frequent small transactions
low_fee_threshold = 5000
if 'fee_charged' in data.columns:
    low_fee_transactions = data[data['fee_charged'] < low_fee_threshold]
    print("Low Fee Transactions:")
    print(low_fee_transactions)
    logging.info("Low Fee Transactions Found: %d", len(low_fee_transactions))

if 'source_account' in data.columns and 'fee_charged' in data.columns:
    small_fee_threshold = 10000
    small_tx = data[data['fee_charged'] < small_fee_threshold]
    small_tx_by_wallet = small_tx.groupby('source_account').size()
    frequent_small_tx = small_tx_by_wallet[small_tx_by_wallet > 5]
    print("Wallets with Frequent Small Transactions:")
    print(frequent_small_tx)
    logging.info("Frequent Small Transactions Found: %d wallets", len(frequent_small_tx))

# Analyze anomalies further
if 'anomaly' in data.columns:
    anomalies = data[data['anomaly'] == -1]

    # Check source accounts for multiple anomalies
    anomalous_accounts = anomalies['source_account'].value_counts()
    print("Accounts with Multiple Anomalies:")
    print(anomalous_accounts)
    logging.info("Accounts with Multiple Anomalies: %s", anomalous_accounts.to_dict())

    # Check timestamps for unusual transaction times
    if 'created_at' in anomalies.columns:
        anomalies['created_at'] = pd.to_datetime(anomalies['created_at'])
        anomalous_times = anomalies['created_at'].dt.hour.value_counts()
        print("Anomalous Transaction Times (Hour of Day):")
        print(anomalous_times)
        logging.info("Anomalous Transaction Times: %s", anomalous_times.to_dict())

# Check if 'fee_charged' column exists
if 'fee_charged' in data.columns:
    # Ensure 'fee_charged' is numeric
    data['fee_charged'] = pd.to_numeric(data['fee_charged'], errors='coerce')
    data = data.dropna(subset=['fee_charged'])
    
    # Analyze 'fee_charged' with a histogram
    data['fee_charged'].hist(bins=50)
    plt.title("Transaction Fee Distribution")
    plt.xlabel("Fee Charged")
    plt.ylabel("Frequency")

    # Save the histogram to the visualizations directory
    histogram_path = os.path.join(visualizations_dir, "transaction_fee_distribution.png")
    plt.savefig(histogram_path)
    plt.show()
    logging.info("Histogram saved to '%s'.", histogram_path)

    # Perform anomaly detection using Isolation Forest
    X = data[['fee_charged']].dropna()  # Drop missing values if any

    # Train Isolation Forest with configurable contamination
    contamination_level = 0.05  # Default to 5%
    clf = IsolationForest(contamination=contamination_level, random_state=42)
    data['anomaly'] = clf.fit_predict(X)

    # Filter and display anomalies
    anomalies = data[data['anomaly'] == -1]
    print("Anomalous Transactions:")
    print(anomalies)
    logging.info("Anomalous Transactions Found: %d", len(anomalies))

    # Ensure the "data" directory exists
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Save anomalies to a file with error handling
    try:
        anomalies.to_csv(save_file, index=False)
        print(f"Anomalous transactions saved to '{save_file}'.")
        logging.info("Anomalous transactions saved to '%s'.", save_file)
    except Exception as e:
        logging.error("Error saving anomalous transactions: %s", e)
else:
    print("Error: 'fee_charged' column not found. Cannot perform anomaly detection.")
    logging.error("'fee_charged' column not found in data. Cannot perform anomaly detection.")
    print("Available Columns:", data.columns.tolist())

# Additional visualization: Daily fee chart
if 'created_at' in data.columns:
    data['created_at'] = pd.to_datetime(data['created_at'])
    daily_fee = data.groupby(data['created_at'].dt.date)['fee_charged'].sum()
    daily_fee.plot(kind='line', title="Daily Total Fee Charged")
    plt.xlabel("Date")
    plt.ylabel("Total Fee Charged")

    daily_fee_path = os.path.join(visualizations_dir, "daily_fee_charged.png")
    plt.savefig(daily_fee_path)
    plt.show()
    logging.info("Daily fee chart saved to '%s'.", daily_fee_path)
