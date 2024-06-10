import pandas as pd

def load_data(train_path, test_path, bu_feat_path):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    bu_feat_df = pd.read_parquet(bu_feat_path)
    return train_df, test_df, bu_feat_df

def merge_store_info(train_df, test_df, bu_feat_df):
    train_df = train_df.merge(bu_feat_df, on='but_num_business_unit', how='left')
    test_df = test_df.merge(bu_feat_df, on='but_num_business_unit', how='left')
    return train_df, test_df

def handle_missing_values(df):
    df.fillna(method='ffill', inplace=True)
    return df

def preprocess_data(train_path, test_path, bu_feat_path):
    train_df, test_df, bu_feat_df = load_data(train_path, test_path, bu_feat_path)
    train_df, test_df = merge_store_info(train_df, test_df, bu_feat_df)
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)
    return train_df, test_df

if __name__ == "__main__":
    train_path = 'data/raw/train.parquet'
    test_path = 'data/raw/test.parquet'
    bu_feat_path = 'data/raw/bu_feat.parquet'
    train_df, test_df = preprocess_data(train_path, test_path, bu_feat_path)
    train_df.to_csv('data/processed/train_preprocessed.csv', index=False)
    test_df.to_csv('data/processed/test_preprocessed.csv', index=False)