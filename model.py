import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from scipy.stats import genpareto

# Constants for time-series forecasting
WINDOW_HOURS = 24
TOKEN_SIZE = 2  # Hours per token
LOCATIONS = 7   # Number of locations
WINDOW_TOKENS = WINDOW_HOURS // TOKEN_SIZE

def preprocess_data(csv_path):
    """
    Preprocesses the raw data for time series forecasting.
    - Loads and renames columns.
    - Creates target variable (next_RRP).
    - Engineers time-based features (raw and cyclical).
    - Creates lag and token-based features for price.
    - Scales numerical features.
    - Encodes categorical features.
    """
    # Load data
    df = pd.read_csv(csv_path, parse_dates=['SETTLEMENTDATE'], index_col='SETTLEMENTDATE')

    # Rename columns for clarity
    df = df.rename(columns={
        'TOTALDEMAND': 'demand_MW',
        'RRP': 'price',
        'temperature_2m (°C)': 'temp', # Example weather column
        'relative_humidity_2m (%)': 'rh', # Example weather column
        'apparent_temperature (°C)': 'app_temp' # Example weather column
    })

    # Create next price target (for predicting next settlement period)
    # Ensure that grouping by location_id is done correctly if multiple locations exist
    if 'location_id' in df.columns:
        df['next_RRP'] = df.groupby('location_id')['price'].shift(-1)
    else:
        # Assuming a single location if 'location_id' is not present
        df['location_id'] = 'default_location' # Add a default location_id
        df['next_RRP'] = df['price'].shift(-1)


    # --- Create time features ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # --- Implement Cyclical Time Features ---
    # Hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # Day of week
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

    # Month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    special_dates_md = [
        (12, 25), # Christmas Day
        (12, 31), # New Year's Eve
        (4, 25),  # Anzac Day
        (12, 26)  # Boxing Day
    ]

    # Create binary columns based on month and day
    df['is_christmas_day'] = ((df.index.month == 12) & (df.index.day == 25)).astype(int)
    df['is_new_years_eve'] = ((df.index.month == 12) & (df.index.day == 31)).astype(int)
    df['is_anzac_day'] = ((df.index.month == 4) & (df.index.day == 25)).astype(int)
    df['is_boxing_day'] = ((df.index.month == 12) & (df.index.day == 26)).astype(int)


    # Select initial relevant columns
    weather_cols = [col for col in df.columns if 'temp' in col.lower() or 'humidity' in col.lower() or 'rh' in col.lower()]

    base_cols = ['price', 'next_RRP', 'demand_MW', 'is_weekend', 'location_id',
                 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                 'month_sin', 'month_cos',
                 'is_christmas_day', 'is_new_years_eve', 'is_anzac_day', 'is_boxing_day']

    if 'hour' not in base_cols:
         base_cols.append('hour')


    working_cols = base_cols + weather_cols

    # Ensure we have all necessary columns before proceeding
    # Filter out columns that might not exist if weather data is missing
    existing_working_cols = [col for col in working_cols if col in df.columns]
    df = df[existing_working_cols].copy()


    # Drop rows with NaN in target variable (must be done before lag features if lags depend on it indirectly)
    df = df.dropna(subset=['next_RRP'])

    print("Creating time-series features (lags and tokens)...")

    # Group by location_id and create lag features
    # Using a list to store processed groups before concatenation
    processed_groups = []
    for location_id_val, group in df.groupby('location_id'):
        group = group.copy() # Work on a copy to avoid SettingWithCopyWarning
        # Create lag features for past 24 hours
        for i in range(1, WINDOW_HOURS + 1):
            col_name = f'price_lag_{i}h'
            group[col_name] = group['price'].shift(i)

        # Create token-based features (aggregating hours into tokens)
        for i in range(WINDOW_TOKENS):
            start_lag = i * TOKEN_SIZE + 1
            end_lag = start_lag + TOKEN_SIZE # end_lag is exclusive in range

            # Ensure lags are present before trying to aggregate them
            cols_to_agg_present = []
            for j in range(start_lag, end_lag):
                lag_col_name = f'price_lag_{j}h'
                if lag_col_name in group.columns:
                    cols_to_agg_present.append(lag_col_name)

            if not cols_to_agg_present: # Skip if no relevant lag columns exist for this token
                # Optionally, fill with a default value like 0 or NaN, or skip creating token feature
                group[f'token_{i}_avg'] = np.nan
                group[f'token_{i}_min'] = np.nan
                group[f'token_{i}_max'] = np.nan
                continue

            # Average price for the token period
            token_col_avg = f'token_{i}_avg'
            group[token_col_avg] = group[cols_to_agg_present].mean(axis=1)

            # Min and max price for the token period
            group[f'token_{i}_min'] = group[cols_to_agg_present].min(axis=1)
            group[f'token_{i}_max'] = group[cols_to_agg_present].max(axis=1)
        processed_groups.append(group)

    if processed_groups:
        df = pd.concat(processed_groups)
    else: # Handle case with no data after grouping (e.g. if location_id was missing and not handled)
        print("Warning: No data after grouping by location_id. Check 'location_id' column and data.")
        # Fallback to original df if it's empty or handle error appropriately
        if df.empty:
            # If df is already empty, return it with an empty date_index or raise error
            return pd.DataFrame(), pd.Series(dtype='datetime64[ns]')


    # Drop rows with NaN in lag/token features (critical after creating them)
    df = df.dropna()
    if df.empty:
        print("Warning: DataFrame is empty after dropping NaNs from lag/token features.")
        return pd.DataFrame(), pd.Series(dtype='datetime64[ns]')

    # Scale numerical features
    demand_scaler = StandardScaler()
    # Ensure 'demand_MW' exists and is not all NaN before scaling
    if 'demand_MW' in df.columns and not df['demand_MW'].isnull().all():
        df['demand_s'] = demand_scaler.fit_transform(df[['demand_MW']])
    else:
        df['demand_s'] = np.nan # Or handle as error/default value

    # Scale weather features if available
    scaled_weather_cols = []
    for col in weather_cols:
        if col in df.columns and not df[col].isnull().all():
            col_scaler = StandardScaler()
            scaled_col_name = f'{col}_s'
            df[scaled_col_name] = col_scaler.fit_transform(df[[col]])
            scaled_weather_cols.append(scaled_col_name)

    # Encode categorical feature (hour) - can be useful even with cyclical hour features
    if 'hour' in df.columns:
        hour_encoder = LabelEncoder()
        df['hour_cat'] = hour_encoder.fit_transform(df['hour'])
    else:
        df['hour_cat'] = np.nan # Or handle as error/default value


    # Reset index for easier handling while maintaining the datetime information
    df = df.reset_index().sort_values(['SETTLEMENTDATE', 'location_id'])

    # --- Select features for the model ---
    feature_cols = []
    # Token features
    for i in range(WINDOW_TOKENS):
        feature_cols.extend([f'token_{i}_avg', f'token_{i}_min', f'token_{i}_max'])

    # Base demand and time features
    # Ensure these columns exist in df before adding to feature_cols
    potential_base_features = [
        'demand_s', 'hour_cat', 'is_weekend', 'location_id',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos'
    ]
    for f_col in potential_base_features:
        if f_col in df.columns:
            feature_cols.append(f_col)

    # Add scaled weather features
    feature_cols.extend(scaled_weather_cols)

    # Ensure all selected feature_cols actually exist in df, remove if not
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Remove target from features if accidentally included
    if 'next_RRP' in feature_cols:
        feature_cols.remove('next_RRP')
    if 'price' in feature_cols: # Original price should not be a direct feature for predicting next_RRP
        feature_cols.remove('price')


    # Create final dataframe with datetime index and selected features
    date_index = df['SETTLEMENTDATE']
    df_final = df.set_index('SETTLEMENTDATE')

    # Ensure target variable 'next_RRP' is present
    if 'next_RRP' not in df_final.columns:
        print("Error: Target variable 'next_RRP' is missing from the final DataFrame.")
        # Return empty DataFrame or handle error
        return pd.DataFrame(columns=feature_cols + ['next_RRP']), pd.Series(dtype='datetime64[ns]')

    # Select only the feature columns and the target variable for the final df
    # This also ensures the order of columns if that's important for the model later
    final_cols_for_model = feature_cols + ['next_RRP']
    df_final = df_final[final_cols_for_model]

    print(f"Prepared dataset with {len(df_final)} rows and {len(feature_cols)} features (excluding target).")
    return df_final, date_index # date_index might be useful for later analysis or plotting

def train_model(df):
    """
    Train XGBoost model using time-series features
    to the positive residuals (underpredicted spikes).
    """
    X = df.drop(['next_RRP'], axis=1)
    y = df['next_RRP']

    # Split data - maintain time ordering
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    # Train XGBoost
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=5,
        eval_metric='mae'
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # Evaluate
    predictions = model.predict(X_test)
    residuals = y_test - predictions
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nModel MAE: {mae:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop 10 important features:")
    print(feature_importance.head(10))


    return model, X_test, y_test, X_train
    
def predict_next_5min_with_signal(model, df, location_id,
                                  profit_target_pct_entry=0.05,
                                  neg_price_entry_target=5):
    """
    Predict the next 5-minute price and suggest a trading signal.

    Parameters:
    -----------
    model : XGBRegressor
        The trained model.
    df : pd.DataFrame
        Feature set with 'SETTLEMENTDATE' as index, includes 'price', 'location_id', and tokens.
    location_id : int
        Location ID to filter.
    profit_target_pct_entry : float
        Minimum predicted increase to trigger a buy.
    neg_price_entry_target : float
        Target price to buy when current price is negative.

    Returns:
    --------
    tuple (float, str)
        Predicted price and trading action: "Buy", "Sell", or "Hold".
    """
    if 'location_id' not in df or location_id not in df['location_id'].values:
        raise ValueError(f"Invalid or missing location_id: {location_id}")

    loc_df = df[df['location_id'] == location_id]
    if not isinstance(loc_df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex (SETTLEMENTDATE).")

    latest_row = loc_df.loc[loc_df.index.max()]
    if latest_row.empty:
        raise ValueError("No recent data found for prediction.")

    # Prepare input features
    features = latest_row.drop('next_RRP', errors='ignore').to_frame().T
    if hasattr(model, 'feature_names_in_'):
        features = features[model.feature_names_in_]

    predicted_price = float(model.predict(features)[0])
    current_price = latest_row.get('price', np.nan)
    if not np.isfinite(current_price):
        return predicted_price, "Hold"

    # Compute recent token average (token_0_avg, token_1_avg, ...)
    token_cols = [col for col in latest_row.index if col.startswith('token_') and col.endswith('_avg')]
    token_values = latest_row[token_cols[:3]].dropna()
    token_avg = token_values.mean() if not token_values.empty else current_price

    # Compute predicted change %
    if current_price == 0:
        predicted_change_pct = np.inf if abs(predicted_price) > 1e-6 else 0
    else:
        predicted_change_pct = (predicted_price - current_price) / abs(current_price)

    # Determine action
    if current_price < token_avg and predicted_change_pct > profit_target_pct_entry:
        action = "Buy"
    elif current_price < 0 and predicted_price > neg_price_entry_target:
        action = "Buy (Negative Price Opportunity)"
    elif predicted_change_pct < -profit_target_pct_entry:
        action = "Sell"
    else:
        action = "Hold"

    return predicted_price, action