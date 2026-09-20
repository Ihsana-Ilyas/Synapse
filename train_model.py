import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET_PATH = "train_delay_master_dataset_zero_leakage.csv"
MODEL_OUTPUT = "future_delay_model.pkl"


def parse_time_to_hours(time_val):
    if pd.isna(time_val) or str(time_val).strip() in ['—', '-', '']:
        return np.nan
    try:
        parts = str(time_val).split(':')
        return float(parts[0]) + float(parts[1]) / 60.0
    except Exception:
        return np.nan


def train_and_save():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"'{DATASET_PATH}' not found in your folder!")

    print(f"Loading '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    df['journey_start_date'] = pd.to_datetime(df['journey_start_date'])

    # 1. Timetable Scheduled Hour
    df['sched_arr_hour'] = df['scheduled_arrival'].apply(parse_time_to_hours)
    df['sched_dep_hour'] = df['scheduled_departure'].apply(parse_time_to_hours)
    df['sched_hour'] = df['sched_arr_hour'].fillna(
        df['sched_dep_hour']).fillna(12.0)

    # 2. Date and Calendar Features
    df['month_num'] = df['journey_start_date'].dt.month
    df['day_of_week_num'] = df['journey_start_date'].dt.weekday
    df['day_of_year'] = df['journey_start_date'].dt.dayofyear
    df['quarter'] = (df['month_num'] - 1) // 3 + 1
    df['is_weekend_flag'] = df['day_of_week_num'].isin([5, 6]).astype(int)

    # 3. Categorical Encodings
    train_cat = df['train_number'].astype('category')
    station_cat = df['station_code'].astype('category')

    df['train_num_cat'] = train_cat.cat.codes
    df['station_code_cat'] = station_cat.cat.codes

    # 4. Bottlenecks
    flag_cols = ['is_junction', 'is_major_junction', 'is_crossing_bottleneck']
    for c in flag_cols:
        df[c] = df[c].fillna(0).astype(int) if c in df.columns else 0

    feature_cols = [
        'train_num_cat',
        'station_code_cat',
        'distance_from_source_km',
        'distance_from_previous_km',
        'distance_to_destination_km',
        'route_completion_ratio',
        'sched_hour',
        'month_num',
        'day_of_week_num',
        'day_of_year',
        'quarter',
        'is_weekend_flag'
    ] + flag_cols

    target_col = 'target_arrival_delay_min'

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    print(f"Training Model on {len(X_train):,} records...")
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)

    print("\n" + "=" * 55)
    print("           MODEL TEST PERFORMANCE")
    print("=" * 55)
    print(f"  Mean Absolute Error (MAE) : {mae:.2f} minutes")
    print(f"  Root Mean Sq Error (RMSE) : {rmse:.2f} minutes")
    print(f"  R² Score                  : {r2:.4f}")
    print("=" * 55)

    # Timetable reference catalog to automatically resolve scheduled arrival & departure
    timetable_ref = df[[
        'train_number', 'train_name', 'station_code', 'station_name',
        'distance_from_source_km', 'distance_from_previous_km', 'distance_to_destination_km',
        'route_completion_ratio', 'scheduled_arrival', 'scheduled_departure',
        'sched_hour', 'is_junction', 'is_major_junction', 'is_crossing_bottleneck'
    ]].drop_duplicates(subset=['train_number', 'station_code']).reset_index(drop=True)

    artifacts = {
        'model': model,
        'feature_cols': feature_cols,
        'timetable_ref': timetable_ref,
        'train_categories': list(train_cat.cat.categories),
        'station_categories': list(station_cat.cat.categories)
    }

    joblib.dump(artifacts, MODEL_OUTPUT)
    print(f"\nModel and timetable catalog saved to '{MODEL_OUTPUT}'")


if __name__ == "__main__":
    train_and_save()
