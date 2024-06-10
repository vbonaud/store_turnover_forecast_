import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pandas as pd
from prophet import Prophet


def prohet_model(train, test) :
    train['ds'] = pd.to_datetime(train['week'].astype(str) + '-1', format='%Y-%W-%w')
    train = train.rename(columns={'turnover': 'y'})
    forecasts = [] # Separate forecasts for each store-department combination
    for bu_id in train['but_num_business_unit'].unique():
        for department_id in train['dpt_num_department'].unique():
            # Filter data for the current store and department
            bu_dept_data = train[(train['but_num_business_unit'] == store_id) & (train['dpt_num_department'] == department_id)]

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
                forecast['but_num_business_unit'] = but_num_business_unit
                forecast['dpt_num_department'] = dpt_num_department

                forecasts.append(forecast)

    # Combine all forecasts into a single dataframe
    forecast_df = pd.concat(forecasts)


