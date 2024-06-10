import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pandas as pd
from prophet import Prophet

# Load the data
train = pd.read_parquet('path/to/train.parquet')
test = pd.read_parquet('path/to/test.parquet')

train['ds'] = pd.to_datetime(train['week'].astype(str) + '-1', format='%Y-%W-%w')
train = train.rename(columns={'turnover': 'y'})

# Merge the forecasts with the test dataset
test['week'] = pd.to_datetime(test['week'].astype(str) + '-1', format='%Y-%W-%w')
merged_test = test.merge(forecast_df, left_on=['but_num_business_unit', 'dpt_num_department', 'week'], right_on=['but_num_business_unit', 'dpt_num_department', 'ds'], how='left')

# Save the predictions to a file
merged_test[['but_num_business_unit', 'dpt_num_department', 'week', 'y']].to_csv('prophet_forecasts.csv', index=False)


# Separate forecasts for each store-department combination
forecasts = []
'but_num_business_unit', 'dpt_num_department'
for store_id in train['but_num_business_unit'].unique():
    for department_id in train['dpt_num_department'].unique():
        # Filter data for the current store and department
        store_dept_data = train[(train['but_num_business_unit'] == store_id) & (train['dpt_num_department'] == department_id)]
        
        # Check if there's enough data to train the model
        if len(store_dept_data) > 8:  # Ensure at least some data points
            # Initialize and fit the model
            model = Prophet()
            model.fit(store_dept_data[['ds', 'y']])
            
            # Create a dataframe for future dates
            future = model.make_future_dataframe(periods=8, freq='W')
            
            # Predict future turnover
            forecast = model.predict(future)
            
            # Extract the relevant part of the forecast
            forecast = forecast[['ds', 'yhat']].tail(8)
            forecast['but_num_business_unit'] = store_id
            forecast['department_id'] = department_id
            
            forecasts.append(forecast)

# Combine all forecasts into a single dataframe
forecast_df = pd.concat(forecasts)

# Merge the forecasts with the test dataset
test['week'] = pd.to_datetime(test['week'].astype(str) + '-1', format='%Y-%W-%w')
merged_test = test.merge(forecast_df, left_on=['but_num_business_unit', 'dpt_num_department', 'week'], right_on=['but_num_business_unit', 'dpt_num_department', 'ds'], how='left')

# Save the predictions to a file
merged_test[['but_num_business_unit', 'dpt_num_department', 'week', 'y']].to_csv('prophet_forecasts.csv', index=False)



def train_model(X_train, y_train):
    # Split the training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train a Random Forest model as an example
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = model.predict(X_val)
    
    # Evaluate the model
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    
    return model

if __name__ == "__main__":
    X_train = pd.read_csv('data/processed/train_features.csv')
    y_train = pd.read_csv('data/processed/train_target.csv')
    model = train_model(X_train, y_train)
    # Save the model
    import joblib
    joblib.dump(model, 'models/prophet_model.pkl')