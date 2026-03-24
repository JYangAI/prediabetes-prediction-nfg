import pandas as pd
import joblib


def scaler(df, target, scaler_file_path):
    features = df.drop(columns=target)
    df_target = df[target]
    scaler_loaded = joblib.load(scaler_file_path)
    features_scaled = scaler_loaded.transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    final_df = pd.concat([features_scaled_df, df_target], axis=1)
    return final_df


def inference_scaler(features_df, scaler_file_path):
    scaler_loaded = joblib.load(scaler_file_path)
    features_scaled = scaler_loaded.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns)
    return features_scaled_df
