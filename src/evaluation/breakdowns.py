import pandas as pd


def breakdown_tables(df: pd.DataFrame):
    by_method = df.groupby('method').mean(numeric_only=True).reset_index()
    by_model = df.groupby('model').mean(numeric_only=True).reset_index()
    by_disruption = df.groupby('disruption_type').mean(numeric_only=True).reset_index() if 'disruption_type' in df.columns else pd.DataFrame()
    by_source = df.groupby('source_dataset').mean(numeric_only=True).reset_index() if 'source_dataset' in df.columns else pd.DataFrame()
    return by_method, by_model, by_disruption, by_source
