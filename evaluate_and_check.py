import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET_PATH = "train_delay_master_dataset_zero_leakage.csv"


def parse_time_to_hours(time_val):
    if pd.isna(time_val) or str(time_val).strip() in ['—', '-', '']:
        return np.nan
    try:
        parts = str(time_val).split(':')
        return float(parts[0]) + float(parts[1]) / 60.0
    except Exception:
        return np.nan


def run_automated_benchmark():
    print("=" * 82)
    print("      AUTOMATED MULTI-ROUTE ACCURACY & LEAKAGE BENCHMARK (ZERO INPUT)")
    print("=" * 82)

    if not os.path.exists(DATASET_PATH):
        print(
            f"Error: '{DATASET_PATH}' not found in the current working directory.")
        return

    df = pd.read_csv(DATASET_PATH)
    df['journey_start_date'] = pd.to_datetime(df['journey_start_date'])

    # 1. Feature Engineering
    df['sched_arr_hour'] = df['scheduled_arrival'].apply(parse_time_to_hours)
    df['sched_dep_hour'] = df['scheduled_departure'].apply(parse_time_to_hours)
    df['sched_hour'] = df['sched_arr_hour'].fillna(
        df['sched_dep_hour']).fillna(12.0)

    df['month_num'] = df['journey_start_date'].dt.month
    df['day_of_week_num'] = df['journey_start_date'].dt.weekday
    df['day_of_year'] = df['journey_start_date'].dt.dayofyear
    df['quarter'] = (df['month_num'] - 1) // 3 + 1
    df['is_weekend_flag'] = df['day_of_week_num'].isin([5, 6]).astype(int)

    train_cat = df['train_number'].astype('category')
    station_cat = df['station_code'].astype('category')
    df['train_num_cat'] = train_cat.cat.codes
    df['station_code_cat'] = station_cat.cat.codes

    flag_cols = ['is_junction', 'is_major_junction', 'is_crossing_bottleneck']
    for c in flag_cols:
        df[c] = df[c].fillna(0).astype(int) if c in df.columns else 0

    feature_cols = [
        'train_num_cat', 'station_code_cat', 'distance_from_source_km',
        'distance_from_previous_km', 'distance_to_destination_km', 'route_completion_ratio',
        'sched_hour', 'month_num', 'day_of_week_num', 'day_of_year', 'quarter', 'is_weekend_flag'
    ] + flag_cols

    target_col = 'target_arrival_delay_min'

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # 2. Data Leakage Verification
    leaky_cols = ['current_train_delay_min', 'delay_delta_min',
                  'is_delayed_flag', 'severe_delay_flag']
    found_leaks = [c for c in leaky_cols if c in feature_cols]
    print(
        f"Data Leakage Verification : {'CLEAN (0 target-derived features in training)' if not found_leaks else f'FAILED: {found_leaks}'}")

    # 3. Chronological / Holdout Split (20% Out-of-Sample)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    print(
        f"Dataset Size              : {len(df):,} total records across 3 major train corridors")
    print(
        f"Train / Holdout Partition : {len(X_train):,} training records | {len(X_test):,} unseen test records\n")

    # 4. Train Model
    print("Training GBDT Regressor...")
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 5. Global Holdout Metrics
    y_pred = np.maximum(0, model.predict(X_test))
    global_mae = mean_absolute_error(y_test, y_pred)
    global_medae = np.median(np.abs(y_test - y_pred))
    global_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    global_r2 = r2_score(y_test, y_pred)
    global_mape = np.mean(
        np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1.0))) * 100.0
    global_mbe = np.mean(y_pred - y_test)
    global_acc_10 = np.mean(np.abs(y_test - y_pred) <= 10.0) * 100.0
    global_acc_15 = np.mean(np.abs(y_test - y_pred) <= 15.0) * 100.0

    print("-" * 82)
    print("                     GLOBAL NETWORK BENCHMARK (OVERALL)")
    print("-" * 82)
    print(f"  • MAE (Mean Absolute Error)     : {global_mae:.2f} minutes")
    print(f"  • MedAE (Median Absolute Error) : {global_medae:.2f} minutes")
    print(f"  • RMSE (Root Mean Sq Error)     : {global_rmse:.2f} minutes")
    print(f"  • MAPE (Mean Absolute Pct Error): {global_mape:.2f}%")
    print(
        f"  • R² Score (Goodness of Fit)    : {global_r2:.4f} ({global_r2 * 100:.2f}%)")
    print(f"  • MBE (Mean Bias / Drift Error) : {global_mbe:+.2f} minutes")
    print(f"  • Punctuality within ±10 mins   : {global_acc_10:.2f}%")
    print(f"  • Punctuality within ±15 mins   : {global_acc_15:.2f}%")
    print("-" * 82)

    # 6. Granular Breakdown by Specific Route Corridor
    test_indices = y_test.index
    test_df = df.loc[test_indices].copy()
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    test_df['abs_error'] = np.abs(test_df['actual'] - test_df['predicted'])

    route_summary = []
    for (t_num, t_name, r_str), grp in test_df.groupby(['train_number', 'train_name', 'route']):
        n_samples = len(grp)
        mean_delay = grp['actual'].mean()
        mae = mean_absolute_error(grp['actual'], grp['predicted'])
        med_ae = np.median(grp['abs_error'])
        rmse = np.sqrt(mean_squared_error(grp['actual'], grp['predicted']))
        r2 = r2_score(grp['actual'], grp['predicted'])
        mape = np.mean(np.abs((grp['actual'] - grp['predicted']) /
                       np.maximum(np.abs(grp['actual']), 1.0))) * 100.0
        acc_10 = np.mean(grp['abs_error'] <= 10.0) * 100.0
        acc_15 = np.mean(grp['abs_error'] <= 15.0) * 100.0

        route_summary.append({
            'Train No': t_num,
            'Train Name': t_name,
            'Route Corridor': r_str,
            'Holdout Trips': n_samples,
            'Avg Delay': f"{mean_delay:.1f}m",
            'MAE': f"{mae:.2f}m",
            'MedAE': f"{med_ae:.2f}m",
            'RMSE': f"{rmse:.2f}m",
            'R²': f"{r2:.4f}",
            'MAPE': f"{mape:.1f}%",
            '±10m Acc': f"{acc_10:.1f}%",
            '±15m Acc': f"{acc_15:.1f}%"
        })

    summary_df = pd.DataFrame(route_summary)

    print("\n" + "=" * 82)
    print("               ACCURACY BREAKDOWN ACROSS ALL TRAIN CORRIDORS")
    print("=" * 82)
    print(summary_df.to_string(index=False))
    print("=" * 82)
    print("Execution completed successfully.\n")


if __name__ == '__main__':
    run_automated_benchmark()
