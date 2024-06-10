import pandas as pd
import numpy as np

def create_datetime_features(df, datetime_col):
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['week'] = df[datetime_col].dt.isocalendar().week
    df['day'] = df[datetime_col].dt.day
    df['dayofweek'] = df[datetime_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

def create_cyclical_features(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def create_interaction_features(df, store_col, dept_col):
    df['bu_depart_interaction'] = df[store_col].astype(str) + '_' + df[dept_col].astype(str)
    return df

def feature_engineering(train_df, test_df):
    # Create datetime features
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    train_df = create_datetime_features(train_df, 'date')
    test_df = create_datetime_features(test_df, 'date')
    
    train_df = create_cyclical_features(train_df, 'dayofweek', 7)
    test_df = create_cyclical_features(test_df, 'dayofweek', 7)
    
    train_df = create_interaction_features(train_df, 'store_id', 'department_id')
    test_df = create_interaction_features(test_df, 'store_id', 'department_id')
    
    # Drop columns not needed in the test set
    drop_cols = ['turnover']  # Specify columns to drop as needed
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['turnover']
    X_test = test_df[X_train.columns]
    
    return X_train, y_train, X_test

if __name__ == "__main__":
    train_path = 'data/processed/train_preprocessed.csv'
    test_path = 'data/processed/test_preprocessed.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train, X_test = feature_engineering(train_df, test_df)
    X_train.to_csv('../data/processed/train_features.csv', index=False)
    y_train.to_csv('../data/processed/train_target.csv', index=False)
    X_test.to_csv('../data/processed/test_features.csv', index=False)
