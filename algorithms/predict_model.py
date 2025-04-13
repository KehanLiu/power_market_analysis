"""random forest algorithm"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def random_forest(X_train, y_train):
    """Create a random forest model"""
    # Create the random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train the model
    model.fit(X_train, y_train)
    return model

def split_train_and_test(data, data_of_yesterday=False):
    """Split the data into training and testing sets"""
    # Create features from the timestamp
    data['hour'] = data['start_timestamp'].dt.hour
    data['day_of_week'] = data['start_timestamp'].dt.dayofweek

    if data_of_yesterday:
        # Create a new feature for the spot_price of the previous day
        data['prev_day_price'] = data['spot_price'].shift(24)

    # Sort the data by 'start_timestamp'
    data.sort_values('start_timestamp', inplace=True)

    data = data.dropna()

    # Split the data into training and testing sets
    train_data = data[data['start_timestamp'] < data['start_timestamp'].max() - pd.DateOffset(months=1)]
    test_data = data[data['start_timestamp'] >= data['start_timestamp'].max() - pd.DateOffset(months=1)]

    if not data_of_yesterday:
        X_train = train_data[['hour', 'day_of_week']]
        X_test = test_data[['hour', 'day_of_week']]
    else:
        X_train = train_data[['hour', 'day_of_week', 'prev_day_price']]
        X_test = test_data[['hour', 'day_of_week', 'prev_day_price']]
    y_train = train_data['spot_price']
    y_test = test_data['spot_price']

    return train_data, test_data, X_train, X_test, y_train, y_test

def predict_next_24_hours_spot_price(df, model, data_of_yesterday=False):
    """Predict the spot price for the next 24 hours."""
    next_day = pd.date_range(start=df['start_timestamp'].max() + pd.DateOffset(days=1), periods=24, freq='H')
    next_day_features = pd.DataFrame({'start_timestamp': next_day})
    next_day_features['hour'] = next_day_features['start_timestamp'].dt.hour
    next_day_features['day_of_week'] = next_day_features['start_timestamp'].dt.dayofweek
    if data_of_yesterday:
        next_day_features['prev_day_price'] = df['spot_price'].iloc[-24:].values
        next_day_predictions = model.predict(next_day_features[['hour', 'day_of_week', 'prev_day_price']])
    else:
        next_day_predictions = model.predict(next_day_features[['hour', 'day_of_week']])
    return next_day_predictions


