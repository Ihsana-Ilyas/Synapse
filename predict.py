import os
import joblib
import pandas as pd
from datetime import datetime, timedelta

MODEL_FILE = "future_delay_model.pkl"


def add_minutes_to_time(time_str, delay_mins):
    """Calculates actual clock time by adding delay to scheduled time."""
    if pd.isna(time_str) or str(time_str).strip() in ['—', '-', '']:
        return "N/A"
    try:
        t = datetime.strptime(str(time_str).strip(), "%H:%M")
        actual_t = t + timedelta(minutes=float(delay_mins))
        return actual_t.strftime("%H:%M")
    except Exception:
        return "N/A"


def run_prediction():
    if not os.path.exists(MODEL_FILE):
        print(
            f"Error: '{MODEL_FILE}' not found! Run 'python train_model.py' first.")
        return

    # Load artifacts
    artifacts = joblib.load(MODEL_FILE)
    model = artifacts['model']
    feature_cols = artifacts['feature_cols']
    timetable = artifacts['timetable_ref']
    train_cats = artifacts['train_categories']
    station_cats = artifacts['station_categories']

    print("\n" + "=" * 65)
    print("      INDIAN RAILWAYS: FUTURE DELAY & ACTUAL TIME PREDICTOR")
    print("=" * 65)

    # 1. SELECT TRAIN
    trains = timetable[['train_number', 'train_name']
                       ].drop_duplicates().reset_index(drop=True)
    print("\nAvailable Trains:")
    for idx, r in trains.iterrows():
        print(f"  [{idx + 1}] Train {r['train_number']}: {r['train_name']}")

    t_choice = int(
        input("\nSelect train (1-3, default 1): ").strip() or "1") - 1
    sel_train_num = trains.iloc[t_choice]['train_number']
    sel_train_name = trains.iloc[t_choice]['train_name']

    # 2. SELECT STATION
    stations = timetable[timetable['train_number']
                         == sel_train_num].reset_index(drop=True)
    print(f"\nStations for {sel_train_name} ({sel_train_num}):")
    for idx, r in stations.iterrows():
        print(
            f"  [{idx + 1:2d}] {r['station_code']:<5} - {r['station_name']:<30} ({r['distance_from_source_km']} km)")

    s_choice = int(input(
        f"\nSelect Station index (1 to {len(stations)}, default 2): ").strip() or "2") - 1
    sel_station = stations.iloc[s_choice]
    station_code = sel_station['station_code']
    station_name = sel_station['station_name']

    # 3. ENTER DATE
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    date_input = input(
        f"\nEnter Journey Date (YYYY-MM-DD, default tomorrow '{tomorrow}'): ").strip() or tomorrow

    try:
        dt = datetime.strptime(date_input, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Using default tomorrow's date.")
        dt = datetime.strptime(tomorrow, "%Y-%m-%d")
        date_input = tomorrow

    # 4. PREDICT
    train_code = train_cats.index(
        sel_train_num) if sel_train_num in train_cats else 0
    station_code_idx = station_cats.index(
        station_code) if station_code in station_cats else 0

    record = {
        'train_num_cat': train_code,
        'station_code_cat': station_code_idx,
        'distance_from_source_km': sel_station['distance_from_source_km'],
        'distance_from_previous_km': sel_station['distance_from_previous_km'],
        'distance_to_destination_km': sel_station['distance_to_destination_km'],
        'route_completion_ratio': sel_station['route_completion_ratio'],
        'sched_hour': sel_station['sched_hour'],
        'month_num': dt.month,
        'day_of_week_num': dt.weekday(),
        'day_of_year': dt.timetuple().tm_yday,
        'quarter': (dt.month - 1) // 3 + 1,
        'is_weekend_flag': 1 if dt.weekday() in [5, 6] else 0,
        'is_junction': sel_station['is_junction'],
        'is_major_junction': sel_station['is_major_junction'],
        'is_crossing_bottleneck': sel_station['is_crossing_bottleneck']
    }

    input_df = pd.DataFrame([record])[feature_cols]
    predicted_delay = max(0.0, round(float(model.predict(input_df)[0]), 1))

    sched_arr = sel_station['scheduled_arrival']
    sched_dep = sel_station['scheduled_departure']

    actual_arr = add_minutes_to_time(sched_arr, predicted_delay)
    actual_dep = add_minutes_to_time(sched_dep, predicted_delay)

    # 5. OUTPUT RESULTS
    print("\n" + "=" * 65)
    print(f"             TRAVEL REPORT FOR {date_input}")
    print("=" * 65)
    print(f"  Train                    : {sel_train_name} ({sel_train_num})")
    print(f"  Target Station           : {station_name} [{station_code}]")
    print(f"  Day of Week              : {dt.strftime('%A')}")
    print(
        f"  Distance from Origin     : {sel_station['distance_from_source_km']} km")
    print("-" * 65)
    print(f"  Scheduled Arrival Time   : {sched_arr}")
    print(f"  Scheduled Departure Time : {sched_dep}")
    print("-" * 65)
    print(f"  PREDICTED ARRIVAL DELAY  : {predicted_delay} MINUTES LATE")
    print(f"  EXPECTED ACTUAL ARRIVAL  : {actual_arr}")
    print(f"  EXPECTED ACTUAL DEPART   : {actual_dep}")
    print("-" * 65)

    if predicted_delay > 60:
        print("  Status Advisory          : SEVERE DELAY (> 1 hr). Plan buffer time.")
    elif predicted_delay > 20:
        print("  Status Advisory          : MODERATE DELAY. Typical for this section.")
    else:
        print("  Status Advisory          : ON TIME / MINIMAL DELAY (< 20 mins).")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_prediction()
