import os
os.environ['OPENELECTRICITY_API_KEY'] = 'oe_3ZK9Yrx1bVBunwPJZXGLJFuG'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from scipy.stats import genpareto
import requests_cache
import openmeteo_requests
from retry_requests import retry
import requests
import csv
from datetime import datetime, timedelta, timezone
import pytz
import time
from openelectricity.types import DataMetric, MarketMetric

# Constants for time-series forecasting
WINDOW_HOURS = 24
TOKEN_SIZE = 2  # Hours per token
LOCATIONS = 7   # Number of locations
WINDOW_TOKENS = WINDOW_HOURS // TOKEN_SIZE

def get_live_price():
    
    # Compute naive UTC timestamps
    tz = pytz.timezone("Australia/Sydney")
    now_local = datetime.now(tz)
    end_minute = (now_local.minute // 5) * 5
    end_local = now_local.replace(minute=end_minute, second=0, microsecond=0)
    start_local = end_local - timedelta(minutes=30)
    end_utc = end_local.astimezone(timezone.utc).replace(tzinfo=None)
    start_utc = start_local.astimezone(timezone.utc).replace(tzinfo=None)

    API_KEY = os.getenv("OPENELECTRICITY_API_KEY")
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # --- Market Data ---
    market_url = "https://api.openelectricity.org.au/v4/market/network/NEM"
    market_params = {
        "metrics": [MarketMetric.DEMAND, MarketMetric.PRICE, MarketMetric.DEMAND_ENERGY],
        "interval": "5m",
        "date_start": start_utc.isoformat(),
        "date_end": end_utc.isoformat(),
        # "primary_grouping": "network_region"
    }
    market_resp = requests.get(market_url, params=market_params, headers=headers)
    market_resp.raise_for_status()
    market_data = market_resp.json()

    # --- Network Data ---
    network_url = "https://api.openelectricity.org.au/v4/data/network/NEM"

    network_params = {
        "metrics": [
            DataMetric.POWER,
            DataMetric.ENERGY,
            DataMetric.MARKET_VALUE,
            DataMetric.EMISSIONS,
        ],
        "interval": "5m",
        # "primary_grouping": "network_region",
        "date_start": start_utc.isoformat(),
        "date_end": end_utc.isoformat(),
    }

    network_resp = requests.get(network_url, params=network_params, headers=headers)
    network_resp.raise_for_status()
    network_data = network_resp.json()

    # --- Process and combine data by timestamp ---
    combined_data = {}

    # Process market data
    for metric_data in market_data.get("data", []):
        metric_name = metric_data.get("metric")
        unit = metric_data.get("unit")

        for result in metric_data.get("results", []):
            for entry in result.get("data", []):
                timestamp = entry[0]
                value = entry[1]

                if timestamp not in combined_data:
                    combined_data[timestamp] = {}

                combined_data[timestamp][f"{metric_name}_{unit}"] = value

    # Process network data
    for metric_data in network_data.get("data", []):
        metric_name = metric_data.get("metric")
        unit = metric_data.get("unit")

        for result in metric_data.get("results", []):
            result_name = result.get("name", "")

            for entry in result.get("data", []):
                timestamp = entry[0]
                value = entry[1]

                if timestamp not in combined_data:
                    combined_data[timestamp] = {}

                column_name = f"{result_name}_{unit}"
                combined_data[timestamp][column_name] = value


    # --- Save to CSV ---
    csv_filename = "nem_snapshot.csv"

    # Get all possible column names to ensure consistent CSV structure
    all_columns = set()
    for data in combined_data.values():
        all_columns.update(data.keys())

    excluded_regions = ("VIC", "TAS", "QLD", "SA")
    filtered_columns = [col for col in all_columns if not any(region in col for region in excluded_regions)]

    # Sort timestamps
    sorted_timestamps = sorted(combined_data.keys())

    # Write to CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp'] + sorted(filtered_columns)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for timestamp in sorted_timestamps:
            row_data = {
                key: value for key, value in combined_data[timestamp].items()
                if key in filtered_columns
            }
            row_data['timestamp'] = timestamp
            writer.writerow(row_data)

    print(f"Saved combined market + network data to {csv_filename}")

    # --- Print first data row with GMT+10 time ---
    with open(csv_filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        first_row = next(reader)
        # pirce = first_row['price_$/MWh']
        print(first_row['price_$/MWh'])
        return float(first_row['price_$/MWh'])
    


def preprocess_data(csv_path):
    """
    Preprocess the energy market data with weather features
    """
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert timestamp to datetime
    if 'SETTLEMENTDATE' in df.columns:
        try:
            df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
            df = df.sort_values('SETTLEMENTDATE').reset_index(drop=True)
        except Exception as e:
            print(f"Warning: Could not convert SETTLEMENTDATE to datetime: {e}")

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Identify price column
    price_cols = [col for col in df.columns if 'RRP' in col.upper() or 'PRICE' in col.upper()]
    if not price_cols:
        price_cols = [col for col in df.columns if 'price' in col.lower()]

    if not price_cols:
        # Try to find any numeric column that might be price
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"No obvious price column found. Numeric columns available: {numeric_cols}")
        if numeric_cols:
            price_col = numeric_cols[0]  # Use first numeric column
            print(f"Using first numeric column as price: {price_col}")
        else:
            raise ValueError("No price column found. Please ensure your data has a numeric price column.")
    else:
        price_col = price_cols[0]
        print(f"Using price column: {price_col}")

    # Ensure price column is numeric
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

    # Create lag features for past 24 hours
    for i in range(1, WINDOW_HOURS + 1):
        df[f'price_lag_{i}h'] = df[price_col].shift(i)

    # Create token features aggregating all 24 lags
    lag_cols = [f'price_lag_{i}h' for i in range(1, WINDOW_HOURS + 1)]
    df['token_0_avg'] = df[lag_cols].mean(axis=1)
    df['token_0_min'] = df[lag_cols].min(axis=1)
    df['token_0_max'] = df[lag_cols].max(axis=1)

    # Create target variable (next price - 5 minutes ahead)
    df['next_RRP'] = df[price_col].shift(-1)  # Next period price

    # Time-based features
    if 'SETTLEMENTDATE' in df.columns and df['SETTLEMENTDATE'].dtype == 'datetime64[ns]':
        df["hour_sin"] = np.sin(2 * np.pi * df['SETTLEMENTDATE'].dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df['SETTLEMENTDATE'].dt.hour / 24)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df['SETTLEMENTDATE'].dt.dayofweek / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df['SETTLEMENTDATE'].dt.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * df['SETTLEMENTDATE'].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df['SETTLEMENTDATE'].dt.month / 12)

        # Additional time features
        df['is_weekend'] = (df['SETTLEMENTDATE'].dt.dayofweek >= 5).astype(int)
        df['hour_cat'] = df['SETTLEMENTDATE'].dt.hour

        # Special date features
        df['is_christmas_day'] = ((df['SETTLEMENTDATE'].dt.month == 12) &
                                 (df['SETTLEMENTDATE'].dt.day == 25)).astype(int)
        df['is_new_years_eve'] = ((df['SETTLEMENTDATE'].dt.month == 12) &
                                 (df['SETTLEMENTDATE'].dt.day == 31)).astype(int)
        df['is_anzac_day'] = ((df['SETTLEMENTDATE'].dt.month == 4) &
                             (df['SETTLEMENTDATE'].dt.day == 25)).astype(int)
        df['is_boxing_day'] = ((df['SETTLEMENTDATE'].dt.month == 12) &
                              (df['SETTLEMENTDATE'].dt.day == 26)).astype(int)
    else:
        # Create default time features if no datetime column
        print("No valid datetime column found, creating default time features")
        df["hour_sin"] = 0
        df["hour_cos"] = 1
        df["day_of_week_sin"] = 0
        df["day_of_week_cos"] = 1
        df["month_sin"] = 0
        df["month_cos"] = 1
        df['is_weekend'] = 0
        df['hour_cat'] = 12  # Default to noon
        df['is_christmas_day'] = 0
        df['is_new_years_eve'] = 0
        df['is_anzac_day'] = 0
        df['is_boxing_day'] = 0

    # Process weather and demand features
    weather_cols = [col for col in df.columns if any(x in col.lower() for x in
                   ['temp', 'humidity', 'wind', 'solar', 'weather'])]

    # Scale key variables if they exist
    key_vars = []

    # Add common weather variables to scaling list
    potential_vars = ['temperature_2m', 'temperature_80m', 'temperature_120m',
                     'apparent_temperature', 'dew_point_2m', 'humidity', 'demand']

    for var in potential_vars:
        matching_cols = [col for col in df.columns if var.lower() in col.lower()]
        if matching_cols:
            key_vars.extend(matching_cols[:1])  # Take first match

    # Scale identified variables
    for var in key_vars:
        if var in df.columns and df[var].dtype in ['float64', 'int64']:
            try:
                scaler = StandardScaler()
                var_data = df[[var]].fillna(df[var].median())
                df[f"{var}_s"] = scaler.fit_transform(var_data).flatten()
            except Exception as e:
                print(f"Warning: Could not scale variable {var}: {e}")
                df[f"{var}_s"] = 0.0

    # Handle location encoding if exists
    if 'location_id' in df.columns:
        try:
            le = LabelEncoder()
            df['location_id_encoded'] = le.fit_transform(df['location_id'].fillna('unknown'))
        except Exception as e:
            print(f"Warning: Could not encode location_id: {e}")
            df['location_id_encoded'] = 0

    # Remove rows with NaN target - but check if target exists first
    if 'next_RRP' in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=['next_RRP'])
        print(f"Removed {initial_len - len(df)} rows with missing target values")

    # Remove rows where we don't have enough lag features
    if all(col in df.columns for col in ['token_0_avg', 'token_0_min', 'token_0_max']):
        initial_len = len(df)
        df = df.dropna(subset=['token_0_avg', 'token_0_min', 'token_0_max'])
        print(f"Removed {initial_len - len(df)} rows with missing lag features")

    print(f"Final processed data shape: {df.shape}")
    return df, price_col

def train_model(df):
    """
    Train XGBoost model using time-series features
    """
    print("Training model...")

    # Check if target variable exists
    if 'next_RRP' not in df.columns:
        raise ValueError("Target variable 'next_RRP' not found in dataframe")

    # Define feature columns (exclude non-feature columns)
    exclude_cols = ['SETTLEMENTDATE', 'next_RRP'] + [col for col in df.columns if 'price_lag' in col]

    # Also exclude any object columns that might cause issues
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols.extend(object_cols)

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Ensure we have the essential token features
    essential_features = ['token_0_avg', 'token_0_min', 'token_0_max']
    for feat in essential_features:
        if feat not in feature_cols and feat in df.columns:
            feature_cols.append(feat)

    # Remove any duplicate columns
    feature_cols = list(set(feature_cols))

    print(f"Using {len(feature_cols)} features")
    print(f"Features: {feature_cols[:10]}...")  # Show first 10 features

    # Check if we have any features at all
    if len(feature_cols) == 0:
        raise ValueError("No valid features found for training")

    # Prepare feature matrix and target
    X = df[feature_cols].copy()
    y = df['next_RRP'].copy()

    # Convert all features to numeric, filling NaNs with 0
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Remove any rows where target is NaN
    valid_idx = ~y.isna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(X) == 0:
        raise ValueError("No valid training samples after removing NaN targets")

    print(f"Training with {len(X)} samples")

    # Split data - maintain time ordering
    split_idx = max(1, int(len(X) * 0.7))  # Ensure at least 1 sample in train
    X_train, X_temp = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_temp = y.iloc[:split_idx], y.iloc[split_idx:]

    # Further split temp into validation and test
    if len(X_temp) > 1:
        val_split_idx = max(1, int(len(X_temp) * 0.5))
        X_val, X_test = X_temp.iloc[:val_split_idx], X_temp.iloc[val_split_idx:]
        y_val, y_test = y_temp.iloc[:val_split_idx], y_temp.iloc[val_split_idx:]
    else:
        # If temp data is too small, use it all for validation
        X_val, X_test = X_temp, X_temp
        y_val, y_test = y_temp, y_temp

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Train XGBoost with error handling
    try:
        model = XGBRegressor(
            n_estimators=min(500, len(X_train) * 2),  # Adjust based on data size
            learning_rate=0.01,
            max_depth=min(5, max(2, len(feature_cols) // 10)),  # Adjust depth based on features
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=min(50, len(X_train) // 10),
            eval_metric='mae',
            random_state=42,
            verbosity=1  # Reduce verbosity
        )

        # Fit with evaluation set if we have validation data
        if len(X_val) > 0 and len(set(y_val)) > 1:  # Check we have varied validation targets
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     verbose=False)  # Reduce output
        else:
            print("Warning: Limited validation data, training without early stopping")
            model.fit(X_train, y_train)

    except Exception as e:
        print(f"Error training XGBoost: {e}")
        print("Trying with simpler parameters...")

        # Fallback with simpler parameters
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

    # Evaluate if we have test data
    if len(X_test) > 0:
        try:
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            print(f"\nModel MAE: {mae:.4f}")
        except Exception as e:
            print(f"Warning: Could not evaluate model: {e}")

    # Feature importance
    try:
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 15 most important features:")
        print(feature_importance.head(15))
    except Exception as e:
        print(f"Warning: Could not calculate feature importance: {e}")

    return model, X_test, y_test, X_train, feature_cols

def save_model_and_artifacts(model, feature_cols, scalers=None, encoders=None,
                           model_path='energy_price_model.pkl'):
    """
    Save the trained model and associated artifacts
    """
    print(f"Saving model to {model_path}")

    model_artifacts = {
        'model': model,
        'feature_columns': feature_cols,
        'scalers': scalers,
        'encoders': encoders,
        'timestamp': datetime.now(),
        'model_type': 'XGBRegressor'
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_artifacts, f)

    print(f"Model saved successfully!")
    return model_path

def load_model(model_path='/content/energy_price_model.pkl'):
    """
    Load the trained model and associated artifacts
    """
    print(f"Loading model from {model_path}")

    with open(model_path, 'rb') as f:
        model_artifacts = pickle.load(f)

    print(f"Model loaded successfully! Trained on: {model_artifacts['timestamp']}")
    return model_artifacts

def update_csv_with_processed_data(original_df, processed_df, csv_path):
    """
    Add processed features to the original CSV
    """
    print("Updating CSV with processed features...")

    # Create backup first
    backup_path = csv_path.replace('.csv', '_backup.csv')
    original_df.to_csv(backup_path, index=False)
    print(f"Backup saved to: {backup_path}")

    # Handle datetime conversion and merging
    if 'SETTLEMENTDATE' in original_df.columns and 'SETTLEMENTDATE' in processed_df.columns:
        # Ensure both SETTLEMENTDATE columns are datetime
        original_df = original_df.copy()  # Avoid modifying original

        # Convert SETTLEMENTDATE to datetime in both dataframes
        if original_df['SETTLEMENTDATE'].dtype == 'object':
            original_df['SETTLEMENTDATE'] = pd.to_datetime(original_df['SETTLEMENTDATE'])
        if processed_df['SETTLEMENTDATE'].dtype == 'object':
            processed_df['SETTLEMENTDATE'] = pd.to_datetime(processed_df['SETTLEMENTDATE'])

        # Reset index to ensure clean merge
        original_df = original_df.reset_index(drop=True)
        processed_df = processed_df.reset_index(drop=True)

        try:
            # Merge on timestamp
            updated_df = original_df.merge(processed_df, on='SETTLEMENTDATE', how='left', suffixes=('', '_processed'))

            # Remove duplicate columns that were created with '_processed' suffix
            cols_to_drop = [col for col in updated_df.columns if col.endswith('_processed')]
            if cols_to_drop:
                print(f"Dropping duplicate columns: {cols_to_drop}")
                updated_df = updated_df.drop(columns=cols_to_drop)

        except Exception as merge_error:
            print(f"Merge failed: {merge_error}")
            print("Falling back to index-based merge...")
            # Fallback: merge by index if timestamps don't match perfectly
            updated_df = original_df.copy()

            # Add new columns from processed_df that don't exist in original
            for col in processed_df.columns:
                if col not in updated_df.columns and col != 'SETTLEMENTDATE':
                    # Align by index
                    if len(processed_df) == len(updated_df):
                        updated_df[col] = processed_df[col].values
                    else:
                        # Handle length mismatch
                        min_len = min(len(processed_df), len(updated_df))
                        updated_df.loc[:min_len-1, col] = processed_df[col].iloc[:min_len].values
    else:
        # If no timestamp column, merge by index
        updated_df = original_df.copy()
        for col in processed_df.columns:
            if col not in updated_df.columns:
                if len(processed_df) == len(updated_df):
                    updated_df[col] = processed_df[col].values
                else:
                    # Handle length mismatch
                    min_len = min(len(processed_df), len(updated_df))
                    updated_df.loc[:min_len-1, col] = processed_df[col].iloc[:min_len].values

    # Save updated CSV
    updated_df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to: {csv_path}")
    print(f"Original shape: {original_df.shape}, Updated shape: {updated_df.shape}")

    return updated_df

def predict_next_5min_price(model_artifacts, latest_data_row):
    """
    Predict the price 5 minutes ahead using the trained model
    """
    model = model_artifacts['model']
    feature_cols = model_artifacts['feature_columns']
    

    # Prepare features for prediction
    if isinstance(latest_data_row, pd.Series):
        # Check if all required features are available
        missing_features = [col for col in feature_cols if col not in latest_data_row.index]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Create a complete feature vector with zeros for missing features
            feature_vector = pd.Series(0.0, index=feature_cols)
            # Fill available features
            available_features = [col for col in feature_cols if col in latest_data_row.index]
            feature_vector[available_features] = latest_data_row[available_features]
        else:
            feature_vector = latest_data_row[feature_cols]

        features = feature_vector.fillna(0).values.reshape(1, -1)
    else:
        # Handle DataFrame input
        missing_features = [col for col in feature_cols if col not in latest_data_row.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing columns with zeros
            for col in missing_features:
                latest_data_row[col] = 0.0

        features = latest_data_row[feature_cols].fillna(0)
        if len(features.shape) == 1:
            features = features.values.reshape(1, -1)
    print("Features used for prediction:", features)
    # Make prediction
    try:
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        print(f"Features shape: {features.shape}")
        print(f"Expected features: {len(feature_cols)}")
        return 0.0

def run_complete_pipeline(csv_path='combined_NSW_nem_weather_forecast.xlsx - Sheet1.csv'):
    """
    Run the complete pipeline: preprocess, train, save, and predict
    """
    print("="*60)
    print("ENERGY PRICE PREDICTION PIPELINE")
    print("="*60)

    try:
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        original_df = pd.read_csv(csv_path)
        processed_df, price_col = preprocess_data(csv_path)

        # Step 2: Update CSV with processed features
        print("\n2. Updating CSV with processed features...")
        updated_df = update_csv_with_processed_data(original_df, processed_df, csv_path)

        # Step 3: Train model
        print("\n3. Training model...")
        model, X_test, y_test, X_train, feature_cols = train_model(processed_df)

        # Step 4: Save model
        print("\n4. Saving model...")
        model_path = save_model_and_artifacts(model, feature_cols)

        # Step 5: Test prediction on latest data
        print("\n5. Testing prediction on latest available data...")
        if len(processed_df) > 0:
            latest_row = processed_df.iloc[-1]

            # Load model (to test loading functionality)
            model_artifacts = load_model(model_path)

            # Make prediction
            prediction = predict_next_5min_price(model_artifacts, latest_row)
            actual_price = latest_row.get('next_RRP', 'N/A')
            current_price = latest_row.get(price_col, 'N/A')

            # Generate trading recommendation
            if current_price != 'N/A' and isinstance(current_price, (int, float)):
                if prediction > current_price:
                    recommendation = "BUY"
                elif prediction < current_price:
                    recommendation = "SELL"
                else:
                    recommendation = "HOLD"
            else:
                recommendation = "HOLD"

            print(f"\nPREDICTION RESULTS:")
            print(f"Current Price: ${current_price:.2f}/MWh" if current_price != 'N/A' else f"Current Price: {current_price}")
            print(f"Predicted Next Price: ${prediction:.2f}/MWh")
            print(f"Trading Recommendation: {recommendation}")

            # Save prediction results to CSV
            prediction_results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': current_price if current_price != 'N/A' else None,
                'predicted_price': prediction,
                'actual_price': actual_price if actual_price != 'N/A' else None,
                'recommendation': recommendation,
                'price_difference': prediction - current_price if current_price != 'N/A' else None
            }

            prediction_df = pd.DataFrame([prediction_results])
            prediction_csv_path = csv_path.replace('.csv', '_predictions.csv')

            # Append to existing predictions file or create new one
            if os.path.exists(prediction_csv_path):
                existing_predictions = pd.read_csv(prediction_csv_path)
                updated_predictions = pd.concat([existing_predictions, prediction_df], ignore_index=True)
            else:
                updated_predictions = prediction_df

            updated_predictions.to_csv(prediction_csv_path, index=False)
            print(f"Prediction results saved to: {prediction_csv_path}")
            if actual_price != 'N/A':
                print(f"Actual Next Price: ${actual_price:.2f}/MWh")
                error = abs(prediction - actual_price)
                print(f"Prediction Error: ${error:.2f}/MWh")

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"- Updated CSV: {csv_path}")
        print(f"- Saved Model: {model_path}")
        print(f"- Predictions CSV: {prediction_csv_path if len(processed_df) > 0 else 'No predictions made'}")
        print(f"- Model Features: {len(feature_cols)}")
        print(f"- Training Data: {len(X_train)} samples")
        print(f"- Test Data: {len(X_test)} samples")

        return model_artifacts, processed_df

    except Exception as e:
        print(f"\nERROR in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def make_live_prediction(model_path='energy_price_model.pkl',
                        csv_path='combined_NSW_nem_weather_forecast.xlsx - Sheet1.csv'):
    """
    Make a live prediction using the saved model
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None

        # Load model
        model_artifacts = load_model(model_path)

        # Load latest data
        if not os.path.exists(csv_path):
            print(f"Data file not found: {csv_path}")
            return None

        df = pd.read_csv(csv_path)

        if len(df) == 0:
            print("No data available for prediction")
            return None

        # Get latest row
        latest_row = df.iloc[-1]

        # Make prediction
        prediction = predict_next_5min_price(model_artifacts, latest_row)

        print(f"\nLIVE PREDICTION:")
        print(f"Predicted Next Price: ${prediction:.2f}/MWh")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return prediction

    except Exception as e:
        print(f"Error making live prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
if __name__ == "__main__":
    # Run the complete pipeline
    model_artifacts, processed_df = run_complete_pipeline()

    # If successful, demonstrate live prediction capability
    if model_artifacts is not None:
        print("\n" + "="*40)
        print("TESTING LIVE PREDICTION CAPABILITY")
        print("="*40)
        live_prediction = make_live_prediction()


# # def preprocess_data(csv_path):
#     """
#     Preprocesses the raw data for time series forecasting.
#     - Loads and renames columns.
#     - Creates target variable (next_RRP).
#     - Engineers time-based features (raw and cyclical).
#     - Creates lag and token-based features for price.
#     - Scales numerical features.
#     - Encodes categorical features.
#     """
#     # Load data
#     df = pd.read_csv(csv_path, parse_dates=['SETTLEMENTDATE'], index_col='SETTLEMENTDATE')

#     # Rename columns for clarity
#     df = df.rename(columns={
#         'TOTALDEMAND': 'demand_MW',
#         'RRP': 'price',
#         'temperature_2m (°C)': 'temp', # Example weather column
#         'relative_humidity_2m (%)': 'rh', # Example weather column
#         'apparent_temperature (°C)': 'app_temp' # Example weather column
#     })

#     # Create next price target (for predicting next settlement period)
#     # Ensure that grouping by location_id is done correctly if multiple locations exist
#     if 'location_id' in df.columns:
#         df['next_RRP'] = df.groupby('location_id')['price'].shift(-1)
#     else:
#         # Assuming a single location if 'location_id' is not present
#         df['location_id'] = 'default_location' # Add a default location_id
#         df['next_RRP'] = df['price'].shift(-1)


#     # --- Create time features ---
#     df['hour'] = df.index.hour
#     df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
#     df['month'] = df.index.month
#     df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

#     # --- Implement Cyclical Time Features ---
#     # Hour
#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

#     # Day of week
#     df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
#     df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

#     # Month
#     df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
#     df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

#     special_dates_md = [
#         (12, 25), # Christmas Day
#         (12, 31), # New Year's Eve
#         (4, 25),  # Anzac Day
#         (12, 26)  # Boxing Day
#     ]

#     # Create binary columns based on month and day
#     df['is_christmas_day'] = ((df.index.month == 12) & (df.index.day == 25)).astype(int)
#     df['is_new_years_eve'] = ((df.index.month == 12) & (df.index.day == 31)).astype(int)
#     df['is_anzac_day'] = ((df.index.month == 4) & (df.index.day == 25)).astype(int)
#     df['is_boxing_day'] = ((df.index.month == 12) & (df.index.day == 26)).astype(int)


#     # Select initial relevant columns
#     weather_cols = [col for col in df.columns if 'temp' in col.lower() or 'humidity' in col.lower() or 'rh' in col.lower()]

#     base_cols = ['price', 'next_RRP', 'demand_MW', 'is_weekend', 'location_id',
#                  'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
#                  'month_sin', 'month_cos',
#                  'is_christmas_day', 'is_new_years_eve', 'is_anzac_day', 'is_boxing_day']

#     if 'hour' not in base_cols:
#          base_cols.append('hour')


#     working_cols = base_cols + weather_cols

#     # Ensure we have all necessary columns before proceeding
#     # Filter out columns that might not exist if weather data is missing
#     existing_working_cols = [col for col in working_cols if col in df.columns]
#     df = df[existing_working_cols].copy()


#     # Drop rows with NaN in target variable (must be done before lag features if lags depend on it indirectly)
#     df = df.dropna(subset=['next_RRP'])

#     print("Creating time-series features (lags and tokens)...")

#     # Group by location_id and create lag features
#     # Using a list to store processed groups before concatenation
#     processed_groups = []
#     for location_id_val, group in df.groupby('location_id'):
#         group = group.copy() # Work on a copy to avoid SettingWithCopyWarning
#         # Create lag features for past 24 hours
#         for i in range(1, WINDOW_HOURS + 1):
#             col_name = f'price_lag_{i}h'
#             group[col_name] = group['price'].shift(i)

#         # Create token-based features (aggregating hours into tokens)
#         for i in range(WINDOW_TOKENS):
#             start_lag = i * TOKEN_SIZE + 1
#             end_lag = start_lag + TOKEN_SIZE # end_lag is exclusive in range

#             # Ensure lags are present before trying to aggregate them
#             cols_to_agg_present = []
#             for j in range(start_lag, end_lag):
#                 lag_col_name = f'price_lag_{j}h'
#                 if lag_col_name in group.columns:
#                     cols_to_agg_present.append(lag_col_name)

#             if not cols_to_agg_present: # Skip if no relevant lag columns exist for this token
#                 # Optionally, fill with a default value like 0 or NaN, or skip creating token feature
#                 group[f'token_{i}_avg'] = np.nan
#                 group[f'token_{i}_min'] = np.nan
#                 group[f'token_{i}_max'] = np.nan
#                 continue

#             # Average price for the token period
#             token_col_avg = f'token_{i}_avg'
#             group[token_col_avg] = group[cols_to_agg_present].mean(axis=1)

#             # Min and max price for the token period
#             group[f'token_{i}_min'] = group[cols_to_agg_present].min(axis=1)
#             group[f'token_{i}_max'] = group[cols_to_agg_present].max(axis=1)
#         processed_groups.append(group)

#     if processed_groups:
#         df = pd.concat(processed_groups)
#     else: # Handle case with no data after grouping (e.g. if location_id was missing and not handled)
#         print("Warning: No data after grouping by location_id. Check 'location_id' column and data.")
#         # Fallback to original df if it's empty or handle error appropriately
#         if df.empty:
#             # If df is already empty, return it with an empty date_index or raise error
#             return pd.DataFrame(), pd.Series(dtype='datetime64[ns]')


#     # Drop rows with NaN in lag/token features (critical after creating them)
#     df = df.dropna()
#     if df.empty:
#         print("Warning: DataFrame is empty after dropping NaNs from lag/token features.")
#         return pd.DataFrame(), pd.Series(dtype='datetime64[ns]')

#     # Scale numerical features
#     demand_scaler = StandardScaler()
#     # Ensure 'demand_MW' exists and is not all NaN before scaling
#     if 'demand_MW' in df.columns and not df['demand_MW'].isnull().all():
#         df['demand_s'] = demand_scaler.fit_transform(df[['demand_MW']])
#     else:
#         df['demand_s'] = np.nan # Or handle as error/default value

#     # Scale weather features if available
#     scaled_weather_cols = []
#     for col in weather_cols:
#         if col in df.columns and not df[col].isnull().all():
#             col_scaler = StandardScaler()
#             scaled_col_name = f'{col}_s'
#             df[scaled_col_name] = col_scaler.fit_transform(df[[col]])
#             scaled_weather_cols.append(scaled_col_name)

#     # Encode categorical feature (hour) - can be useful even with cyclical hour features
#     if 'hour' in df.columns:
#         hour_encoder = LabelEncoder()
#         df['hour_cat'] = hour_encoder.fit_transform(df['hour'])
#     else:
#         df['hour_cat'] = np.nan # Or handle as error/default value


#     # Reset index for easier handling while maintaining the datetime information
#     df = df.reset_index().sort_values(['SETTLEMENTDATE', 'location_id'])

#     # --- Select features for the model ---
#     feature_cols = []
#     # Token features
#     for i in range(WINDOW_TOKENS):
#         feature_cols.extend([f'token_{i}_avg', f'token_{i}_min', f'token_{i}_max'])

#     # Base demand and time features
#     # Ensure these columns exist in df before adding to feature_cols
#     potential_base_features = [
#         'demand_s', 'hour_cat', 'is_weekend', 'location_id',
#         'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
#         'month_sin', 'month_cos'
#     ]
#     for f_col in potential_base_features:
#         if f_col in df.columns:
#             feature_cols.append(f_col)

#     # Add scaled weather features
#     feature_cols.extend(scaled_weather_cols)

#     # Ensure all selected feature_cols actually exist in df, remove if not
#     feature_cols = [col for col in feature_cols if col in df.columns]

#     # Remove target from features if accidentally included
#     if 'next_RRP' in feature_cols:
#         feature_cols.remove('next_RRP')
#     if 'price' in feature_cols: # Original price should not be a direct feature for predicting next_RRP
#         feature_cols.remove('price')


#     # Create final dataframe with datetime index and selected features
#     date_index = df['SETTLEMENTDATE']
#     df_final = df.set_index('SETTLEMENTDATE')

#     # Ensure target variable 'next_RRP' is present
#     if 'next_RRP' not in df_final.columns:
#         print("Error: Target variable 'next_RRP' is missing from the final DataFrame.")
#         # Return empty DataFrame or handle error
#         return pd.DataFrame(columns=feature_cols + ['next_RRP']), pd.Series(dtype='datetime64[ns]')

#     # Select only the feature columns and the target variable for the final df
#     # This also ensures the order of columns if that's important for the model later
#     final_cols_for_model = feature_cols + ['next_RRP']
#     df_final = df_final[final_cols_for_model]

#     print(f"Prepared dataset with {len(df_final)} rows and {len(feature_cols)} features (excluding target).")
#     return df_final, date_index # date_index might be useful for later analysis or plotting

# # def train_model(df):
#     """
#     Train XGBoost model using time-series features
#     to the positive residuals (underpredicted spikes).
#     """
#     X = df.drop(['next_RRP'], axis=1)
#     y = df['next_RRP']

#     # Split data - maintain time ordering
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
#     X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

#     # Train XGBoost
#     model = XGBRegressor(
#         n_estimators=500,
#         learning_rate=0.01,
#         max_depth=5,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         early_stopping_rounds=5,
#         eval_metric='mae'
#     )
#     model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

#     # Evaluate
#     predictions = model.predict(X_test)
#     residuals = y_test - predictions
#     mae = mean_absolute_error(y_test, predictions)
#     print(f"\nModel MAE: {mae:.4f}")

#     # Feature importance
#     feature_importance = pd.DataFrame({
#         'Feature': X_train.columns,
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False)
#     print("\nTop 10 important features:")
#     print(feature_importance.head(10))


#     return model, X_test, y_test, X_train

# # def predict_next_5min_with_signal(model, df, location_id,
#                                   profit_target_pct_entry=0.05,
#                                   neg_price_entry_target=5):
#     """
#     Predict the next 5-minute price and suggest a trading signal.

#     Parameters:
#     -----------
#     model : XGBRegressor
#         The trained model.
#     df : pd.DataFrame
#         Feature set with 'SETTLEMENTDATE' as index, includes 'price', 'location_id', and tokens.
#     location_id : int
#         Location ID to filter.
#     profit_target_pct_entry : float
#         Minimum predicted increase to trigger a buy.
#     neg_price_entry_target : float
#         Target price to buy when current price is negative.

#     Returns:
#     --------
#     tuple (float, str)
#         Predicted price and trading action: "Buy", "Sell", or "Hold".
#     """
#     if 'location_id' not in df or location_id not in df['location_id'].values:
#         raise ValueError(f"Invalid or missing location_id: {location_id}")

#     loc_df = df[df['location_id'] == location_id]
#     if not isinstance(loc_df.index, pd.DatetimeIndex):
#         raise ValueError("Index must be a DatetimeIndex (SETTLEMENTDATE).")

#     latest_row = loc_df.loc[loc_df.index.max()]
#     if latest_row.empty:
#         raise ValueError("No recent data found for prediction.")

#     # Prepare input features
#     features = latest_row.drop('next_RRP', errors='ignore')
#     if hasattr(model, 'feature_names_in_'):
#         features = features[model.feature_names_in_]

#     predicted_price = float(model.predict(features)[0])
#     current_price = latest_row.get('price', np.nan)
#     if not np.isfinite(current_price):
#         return predicted_price, "Hold"

#     # Compute recent token average (token_0_avg, token_1_avg, ...)
#     token_cols = [col for col in latest_row.index if col.startswith('token_') and col.endswith('_avg')]
#     token_values = latest_row[token_cols[:3]].dropna()
#     token_avg = token_values.mean() if not token_values.empty else current_price

#     # Compute predicted change %
#     if current_price == 0:
#         predicted_change_pct = np.inf if abs(predicted_price) > 1e-6 else 0
#     else:
#         predicted_change_pct = (predicted_price - current_price) / abs(current_price)

#     # Determine action
#     if current_price < token_avg and predicted_change_pct > profit_target_pct_entry:
#         action = "Buy"
#     elif current_price < 0 and predicted_price > neg_price_entry_target:
#         action = "Buy (Negative Price Opportunity)"
#     elif predicted_change_pct < -profit_target_pct_entry:
#         action = "Sell"
#     else:
#         action = "Hold"

#     return predicted_price, action