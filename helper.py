import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
import warnings
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error

def generate_fips_code(row):
    state_ansi_code = str(row["state_ansi"]).zfill(2)
    county_ansi_code = str(row["county_ansi"]).zfill(3)
    return f"{state_ansi_code}{county_ansi_code}"

def calculate_acre_from_production(row):
    if row["commodity_desc"] == "COTTON":
        return row["PRODUCTION, MEASURED IN 480 LB BALES"]/row["YIELD, MEASURED IN LB / ACRE"]
    return row["PRODUCTION, MEASURED IN BU"]/row["YIELD, MEASURED IN BU / ACRE"]


def load_all_crop_production_data(crop_data_path, states_producer, crop):
    years = [2017, 2018, 2019, 2020, 2021, 2022]
    
    all_years_crop_data = []
    crop_records = 0
    for year in years:
        file_path = f"{crop_data_path}/{crop}/{year}"
        try:
            df = pd.read_csv(f"{file_path}/USDA_{crop}_County_{year}.csv")
            df = df[df["state_name"].isin(states_producer)]
            all_years_crop_data.append(df)
            crop_records += len(df)
        except FileNotFoundError:
            print(f" {crop} {year}: FILE NOT FOUND")
            continue
        except Exception as e:
            print(f" {crop} {year}: ERROR - {e}")
            continue
    
    
    if all_years_crop_data:
        master_crop_data = pd.concat(all_years_crop_data, ignore_index=True)
        total_loaded_states = len(master_crop_data["state_name"].unique())
        print(f"Loaded {crop} production data from {total_loaded_states} states")
        return master_crop_data
    
    else:
        print("ERROR: No crop data loaded!")
        return None
    
def crop_data_missing_values_check(crop_data):
    crop_type = crop_data["commodity_desc"].iloc[0]
    
        
    # Corn Production Missing Values Analysis

    # Assuming your corn dataset is loaded as 'crop_data'
    # Replace with your actual variable name

    print("=" * 60)
    print(f"{crop_type} PRODUCTION MISSING VALUES ANALYSIS")
    print("=" * 60)

    # 1. Dataset Overview
    print(f"\n1. DATASET OVERVIEW:")
    print(f"Total records: {len(crop_data)}")
    print(f"Columns: {list(crop_data.columns)}")
    print(f"Shape: {crop_data.shape}")

    # 2. Missing Values Count
    print(f"\n2. MISSING VALUES COUNT:")
    missing_counts = crop_data.isnull().sum()
    print(missing_counts)

    # 3. Focus on Key Columns
    yield_col = 'YIELD, MEASURED IN BU / ACRE'
    production_col = 'PRODUCTION, MEASURED IN BU'
    if crop_type == "COTTON":
        production_col = "PRODUCTION, MEASURED IN 480 LB BALES"
        yield_col = "YIELD, MEASURED IN LB / ACRE"

    print(f"\n3. KEY COLUMNS ANALYSIS:")
    print(f"Missing values in YIELD column: {crop_data[yield_col].isnull().sum()}")
    print(f"Missing values in PRODUCTION column: {crop_data[production_col].isnull().sum()}")

    # 4. Zero Values Check
    print(f"\n4. ZERO VALUES CHECK:")
    yield_zeros = (crop_data[yield_col] == 0).sum()
    production_zeros = (crop_data[production_col] == 0).sum()
    print(f"Zero values in YIELD: {yield_zeros}")
    print(f"Zero values in PRODUCTION: {production_zeros}")

    print(f"\n5. MISSING VALUES BY STATE:")
    state_missing = crop_data.groupby('state_name').apply(lambda x: x[yield_col].isnull().sum())
    print("Missing YIELD values by state:")
    print(state_missing)

    # 6. Missing Values by Year
    print(f"\n6. MISSING VALUES BY YEAR:")
    year_missing = crop_data.groupby('year').apply(lambda x: x[yield_col].isnull().sum())
    print("Missing YIELD values by year:")
    print(year_missing)

    # 7. Sample Data Inspection
    print(f"\n7. SAMPLE DATA INSPECTION:")
    print("First 5 rows of key columns:")
    key_cols = ['state_name', 'year', yield_col, production_col]
    print(crop_data[key_cols].head())

    # 8. Basic Statistics
    print(f"\n8. YIELD STATISTICS (excluding missing values):")
    yield_stats = crop_data[yield_col].describe()
    print(yield_stats)

    # 9. Outlier Detection (Simple approach)
    print(f"\n9. POTENTIAL OUTLIERS:")
    Q1 = crop_data[yield_col].quantile(0.25)
    Q3 = crop_data[yield_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = crop_data[
        (crop_data[yield_col] < lower_bound) | 
        (crop_data[yield_col] > upper_bound)
    ]
    print(f"Number of potential outliers: {len(outliers)}")
    if len(outliers) > 0:
        print("Outlier examples:")
        print(outliers[key_cols].head(10))

    # 10. Complete vs Incomplete Records
    print(f"\n10. DATA COMPLETENESS SUMMARY:")
    complete_records = crop_data[yield_col].notna().sum()
    total_records = len(crop_data)
    completeness_rate = (complete_records / total_records) * 100
    print(f"Complete records: {complete_records}/{total_records} ({completeness_rate:.1f}%)")

    # 11. State-Year Completeness Matrix
    print(f"\n11. STATE-YEAR COMPLETENESS MATRIX:")
    completeness_matrix = crop_data.pivot_table(
        values=yield_col, 
        index='state_name', 
        columns='year', 
        aggfunc='count',
        fill_value=0
    )
    print("Records count by State x Year:")
    print(completeness_matrix)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - REVIEW RESULTS ABOVE")
    print("=" * 60)
    return 

def generate_month_number_string(month_number):
    return f"0{month_number}" if month_number < 10 else f"{month_number}"

def _retrieve_weather_data(weather_data_directory, selected_states, selected_counties=[], month = "04", year="2017", type="Monthly"):
    year_dir = os.path.join(weather_data_directory, year)
    file_template = "{}-{}.csv"
    targeted_month = file_template.format(year, month)

    all_data = []
    states_dirs = [state for state in os.listdir(year_dir) if state in selected_states]
    
    for state_dir in states_dirs:
        if not state_dir in selected_states:
            continue
        state_path = os.path.join(year_dir, state_dir)
        # iteratore over the files in the dir
        for file in os.listdir(state_path):
            if file.endswith(targeted_month):
            # full path to file
                file_path = os.path.join(state_path, file)
                df = pd.read_csv(file_path)

                # only consifer the monthly data for visualization
                df = df[df["Daily/Monthly"] == type]
                # hanlde the fips code
                df["FIPS Code"] = df["FIPS Code"].astype(str).str.zfill(5)
                if len(selected_counties) > 1:
                    df = df[df["FIPS Code"].isin(selected_counties)]
                all_data.append(df)

    # concatenate all dfs into a single one
    monthly_weather_data = pd.concat(all_data)
    return monthly_weather_data

def load_monthly_weather_data(weather_data_directory, selected_states, selected_counties=[], year="2017"):
    monthly_weather = {}
    for month in range(1, 13, 1):
        month_string = generate_month_number_string(month)
        
        weather = _retrieve_weather_data(weather_data_directory, selected_states, selected_counties, month=month_string, year=f"{year}")
        monthly_weather[month_string] = weather

    return monthly_weather

def load_daily_weather_data(weather_data_directory, selected_states, selected_counties=[], year="2017"):
    monthly_weather = {}
    for month in range(1, 13, 1):
        month_string = generate_month_number_string(month)
        weather = _retrieve_weather_data(weather_data_directory, selected_states, selected_counties, month=month_string, year=f"{year}", type="Daily")
        monthly_weather[month_string] = weather

    return monthly_weather


def split_dataset_by_year(full_dataset, test_set_year=2022, val_set_year=None):
    
    test_set = full_dataset[full_dataset["Year"] == test_set_year]
    train_set = full_dataset[full_dataset["Year"] != test_set_year ]
    if val_set_year is not None:
        train_set = full_dataset[full_dataset["Year"] != test_set_year and full_dataset["Year"] != val_set_year]
        val_set = full_dataset[full_dataset["Year"] == val_set_year] 
        return train_set, val_set, test_set
    
    return train_set, test_set


def cross_validation_split_by_year(full_train_set, val_set_year, target_column):
    
    val_set = full_train_set[full_train_set["Year"] == val_set_year]
    train_set = full_train_set[full_train_set["Year"] != val_set_year ]
    
    X_train = train_set.drop(columns=[target_column, "Year"])
    y_train = train_set[target_column]
    
    X_val = val_set.drop(columns=[target_column, "Year"])
    y_val = val_set[target_column]
    
    return X_train, X_val, y_train, y_val

def number_formatter(x, pos):
    if abs(x) >= 1_000_000:
        return f'{x * 1e-6:.1f}M'
    elif abs(x) >= 1_000:
        return f'{x * 1e-3:.1f}K'
    else:
        return f'{x:.0f}'
    
def flexible_format(metric_name, value):
    abs_value = abs(value)
    
    if metric_name == 'Avg R2 Score':
        return f"{value:.2f}"
    
    elif 'MAPE' in metric_name:
        return f"{value * 100:.2f}%"  # Convert to percentage
    
    elif 'Error' in metric_name or 'MAE' in metric_name or 'RMSE' in metric_name:
        if abs_value >= 1_000_000:
            return f"{value / 1e6:.2f}M"
        elif abs_value >= 1_000:
            return f"{value / 1e3:.2f}K"
        else:
            return f"{value:.0f}"
    
    # Default fallback
    return f"{value:.2f}"


def plot_cross_validation_result(results, title="", show_title=True):
    

    years = [r['val_set_year'] for r in results]
    
    r2_val = [r['r2_val'] for r in results]
    r2_train = [r['r2_train'] for r in results]
    
    mae_val = [r['mae_val'] for r in results]
    mae_train = [r['mae_train'] for r in results]
    
    rmse_train = [r['rmse_train'] for r in results]
    rmse_val = [r['rmse_val'] for r in results]

    mape_train = [r['mape_train'] for r in results]
    mape_val = [r['mape_val'] for r in results]

    max_error_train = [r['max_error_train'] for r in results]
    max_error_val = [r['max_error_val'] for r in results]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(years, r2_val, marker='o', label='R² Validation')
    ax1.plot(years, r2_train, marker='s', label='R² Training')
    ax1.set_title('R² Score Across Validation Years')
    ax1.set_xlabel('Validation Year')
    ax1.set_ylabel('R² Score')
    ax1.set_xticks(years)
    ax1.legend()
    ax1.grid(True)
    
    max_val = max(mae_val + mae_train)  # max of all MAE values

    if max_val >= 1_000_000:
        y_label = 'MAE (Millions of Bushels)'
    elif max_val >= 1_000:
        y_label = 'MAE (Thousands of Bushels)'
    else:
        y_label = 'MAE (Bushels)'

    ax2.plot(years, mae_val, marker='o', label='MAE Validation')
    ax2.plot(years, mae_train, marker='s', label='MAE Training')
    ax2.set_title('Mean Absolute Error Across Validation Years')
    ax2.set_xlabel('Validation Year')
    ax2.set_ylabel(y_label)
    ax2.set_xticks(years)
    ax2.yaxis.set_major_formatter(number_formatter)  # Apply custom formatter
    ax2.legend()
    ax2.grid(True)

    if show_title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()
    
    # Calculate averages
    avg_r2_val = np.mean(r2_val)
    avg_r2_train = np.mean(r2_train)
    avg_mae_val = np.mean(mae_val)
    avg_mae_train = np.mean(mae_train)
    
    avg_rmse_train = np.mean(rmse_train)
    avg_rmse_val = np.mean(rmse_val)

    avg_mape_train = np.mean(mape_train)
    avg_mape_val = np.mean(mape_val)

    avg_max_error_train = np.mean(max_error_train)
    avg_max_error_val = np.mean(max_error_val)
    

    # Create a DataFrame
    df_avg = pd.DataFrame({
    'Metric': [
        'Avg R2 Score',
        'Avg Mean Absolute Error (MAE)',
        'Avg Root Mean Squared Error (RMSE)',
        'Avg Mean Absolute Percentage Error (MAPE)',
        'Avg Max Error'
    ],
    'Train': [
        avg_r2_train,
        avg_mae_train,
        avg_rmse_train,
        avg_mape_train,
        avg_max_error_train
    ],
    'Validation': [
        avg_r2_val,
        avg_mae_val,
        avg_rmse_val,
        avg_mape_val,
        avg_max_error_val
    ]
})

    # Format columns with flexible formatting
    df_avg['Train'] = df_avg.apply(lambda row: flexible_format(row['Metric'], row['Train']), axis=1)
    df_avg['Validation'] = df_avg.apply(lambda row: flexible_format(row['Metric'], row['Validation']), axis=1)

    print(df_avg)
    
def get_skewed_features(df, exclude_columns=None, threshold=1.0):
    """
    Returns list of numerical feature names with skewness > threshold.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Ensure it's a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame.")
    
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=exclude_columns, errors='ignore')
    skewness = numeric_df.skew()
    skewed_features = skewness[skewness > threshold].index.tolist()
    
    return skewed_features


def perform_cross_validation(
    train_set, 
    target_column, 
    scaler=StandardScaler(), 
    model=RandomForestRegressor(random_state=42),
):
    cross_val_result = []
    year_list = train_set["Year"].unique()

    for val_year in year_list:
   
        X_train, X_val, y_train, y_val = cross_validation_split_by_year(train_set, val_year, target_column)

        # === Feature scaling ===
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # === Fit model ===
        model.fit(X_train_scaled, y_train)

        # === Predict ===
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
  
        # === Evaluation ===
        r2_train = r2_score(y_train, y_train_pred)
        r2_val = r2_score(y_val, y_val_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

        mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
        mape_val = mean_absolute_percentage_error(y_val, y_val_pred)

        max_error_train = max_error(y_train, y_train_pred)
        max_error_val = max_error(y_val, y_val_pred)

        cross_val_result.append({
            "val_set_year": int(val_year),
            "r2_val": r2_val,
            "r2_train": r2_train,
            "mae_val": mae_val,
            "mae_train": mae_train,
            'rmse_train': rmse_train,
            'rmse_val': rmse_val,
            'mape_train': mape_train,
            'mape_val': mape_val,
            'max_error_train': max_error_train,
            'max_error_val': max_error_val,
            "model": model
        })

    return cross_val_result
    
def precompute_transformations_by_fold(
    train_set: pd.DataFrame,
    target_column: str,
    feature_skew_threshold: float = 1.0,
    target_skew_threshold: float = 1.0,
    verbose: bool = False
):
    """
    Pre-compute all transformations for each fold without data leakage.
    
    For each validation year:
    1. Exclude that year from training data
    2. Detect skewed features using remaining training data only
    3. Apply transformations and store results
    
    Returns a dictionary with pre-computed transformations for each fold.
    """
    year_list = train_set["Year"].unique()
    feature_columns = [col for col in train_set.columns if col not in ["Year", target_column]]
    
    # Dictionary to store pre-computed transformations
    precomputed_data = {}
    
    if verbose:
        print("=== Pre-computing leak-free transformations ===")
    
    for val_year in year_list:
        if verbose:
            print(f"\nProcessing fold: validation year = {val_year}")
        
        # Split data (training = all years except val_year)
        train_mask = train_set["Year"] != val_year
        val_mask = train_set["Year"] == val_year
        
        train_data = train_set[train_mask].copy()
        val_data = train_set[val_mask].copy()
        
        # === FEATURE SKEWNESS DETECTION (training data only) ===
        skewed_features = []
        feature_skewness = {}
        
        for feature in feature_columns:
            if feature in train_data.columns:
                feature_skew = skew(train_data[feature])
                feature_skewness[feature] = feature_skew
                
                if abs(feature_skew) > feature_skew_threshold:
                    skewed_features.append(feature)
        
        if verbose:
            print(f"  Skewed features detected: {skewed_features}")
        
        # === TARGET SKEWNESS DETECTION (training data only) ===
        target_skewness = skew(train_data[target_column])
        apply_target_log = target_skewness > target_skew_threshold
        
        if verbose:
            print(f"  Target skewness: {target_skewness:.3f} - {'applying' if apply_target_log else 'not applying'} log transform")
        
        # === APPLY TRANSFORMATIONS ===
        train_transformed = train_data.copy()
        val_transformed = val_data.copy()
        
        # Transform features
        if skewed_features:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for feature in skewed_features:
                    # Transform training data
                    train_feature_data = np.clip(train_transformed[feature].values, 0, 1e50)
                    train_transformed[feature] = np.log1p(train_feature_data)
                    
                    # Apply same transformation to validation data
                    val_feature_data = np.clip(val_transformed[feature].values, 0, 1e50)
                    val_transformed[feature] = np.log1p(val_feature_data)
        
        # Transform target
        if apply_target_log:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # Transform training targets
                train_target_data = np.clip(train_transformed[target_column].values, 0, 1e50)
                train_transformed[target_column + '_log'] = np.log1p(train_target_data)
                
                # Transform validation targets
                val_target_data = np.clip(val_transformed[target_column].values, 0, 1e50)
                val_transformed[target_column + '_log'] = np.log1p(val_target_data)
        
        # === STORE PRE-COMPUTED DATA ===
        precomputed_data[val_year] = {
            'train_data': train_transformed,
            'val_data': val_transformed,
            'original_train_data': train_data,
            'original_val_data': val_data,
            'skewed_features': skewed_features,
            'feature_skewness': feature_skewness,
            'target_skewness': target_skewness,
            'apply_target_log': apply_target_log,
            'target_column_transformed': target_column + '_log' if apply_target_log else target_column
        }
    
    if verbose:
        print(f"\n=== Pre-computation complete for {len(year_list)} folds ===")
        
        # Summary statistics
        all_skewed_features = set()
        target_transform_count = 0
        
        for fold_data in precomputed_data.values():
            all_skewed_features.update(fold_data['skewed_features'])
            if fold_data['apply_target_log']:
                target_transform_count += 1
        
        print(f"Unique skewed features across all folds: {sorted(all_skewed_features)}")
        print(f"Folds with target transformation: {target_transform_count}/{len(year_list)}")
    
    return precomputed_data


def perform_cv_with_precomputed_transforms(
    precomputed_data,
    target_column,
    scaler=StandardScaler(),
    model=RandomForestRegressor(random_state=42),
    verbose: bool = False,
    max_safe_exp: float = 50
):
    """
    Perform cross-validation using pre-computed transformations.
    This is very fast since all transformations are already done.
    """
    cross_val_result = []
    
    for val_year, fold_data in precomputed_data.items():
        if verbose:
            print(f"\n=== Training fold: validation year = {val_year} ===")
        
        # Extract pre-computed data
        train_transformed = fold_data['train_data']
        val_transformed = fold_data['val_data']
        train_original = fold_data['original_train_data']
        val_original = fold_data['original_val_data']
        
        apply_target_log = fold_data['apply_target_log']
        target_col_transformed = fold_data['target_column_transformed']
        skewed_features = fold_data['skewed_features']
        
        if verbose:
            print(f"  Transformed features: {skewed_features}")
            print(f"  Target transformed: {apply_target_log}")
        
        # Prepare features and targets
        feature_cols = [col for col in train_transformed.columns if col not in ["Year", target_column, target_column + '_log']]
        
        X_train = train_transformed[feature_cols]
        X_val = val_transformed[feature_cols]
        y_train_transformed = train_transformed[target_col_transformed]
        y_val_transformed = val_transformed[target_col_transformed]
        
        # Original targets for evaluation
        y_train_original = train_original[target_column]
        y_val_original = val_original[target_column]
        
        # === FEATURE SCALING (still per fold) ===
        fold_scaler = type(scaler)(**scaler.get_params()) if hasattr(scaler, 'get_params') else StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_val_scaled = fold_scaler.transform(X_val)
        
        # === MODEL FITTING ===
        fold_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else RandomForestRegressor(random_state=42)
        fold_model.fit(X_train_scaled, y_train_transformed)
        
        # === PREDICTION ===
        y_train_pred = fold_model.predict(X_train_scaled)
        y_val_pred = fold_model.predict(X_val_scaled)
        
        # === INVERSE TRANSFORM (if needed) ===
        if apply_target_log:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y_train_pred = np.expm1(np.clip(y_train_pred, None, max_safe_exp))
                y_val_pred = np.expm1(np.clip(y_val_pred, None, max_safe_exp))
                
                # Safety clipping
                y_max_reasonable = y_train_original.max() * 100
                y_train_pred = np.clip(y_train_pred, 0, y_max_reasonable)
                y_val_pred = np.clip(y_val_pred, 0, y_max_reasonable)
        
        # === EVALUATION (on original scale) ===
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r2_train = np.clip(r2_score(y_train_original, y_train_pred), -1e6, 1.0)
            r2_val = np.clip(r2_score(y_val_original, y_val_pred), -1e6, 1.0)
            mae_train = mean_absolute_error(y_train_original, y_train_pred)
            mae_val = mean_absolute_error(y_val_original, y_val_pred)
            
            rmse_train = np.sqrt(mean_squared_error(y_train_original, y_train_pred))
            rmse_val = np.sqrt(mean_squared_error(y_val_original, y_val_pred))
            
            mape_train = mean_absolute_percentage_error(y_train_original, y_train_pred)
            mape_val = mean_absolute_percentage_error(y_val_original, y_val_pred)

            max_error_train = max_error(y_train_original, y_train_pred)
            max_error_val = max_error(y_val_original, y_val_pred)
        
        cross_val_result.append({
            "val_set_year": int(val_year),
            "r2_val": r2_val,
            "r2_train": r2_train,
            "mae_val": mae_val,
            "mae_train": mae_train,
            'rmse_train': rmse_train,
            'rmse_val': rmse_val,
            'mape_train': mape_train,
            'mape_val': mape_val,
            'max_error_train': max_error_train,
            'max_error_val': max_error_val,
            "skewed_features": skewed_features,
            "target_transformed": apply_target_log,
            "model": fold_model
        })
        
        if verbose:
            print(f"  R² validation: {r2_val:.4f}")
            print(f"  MAE validation: {mae_val:.4f}")
    
    return cross_val_result

def train_final_model_and_predict_log_transformed(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    target_column: str,
    feature_skew_threshold: float = 1.0,
    target_skew_threshold: float = 1.0,
    scaler=StandardScaler(),
    model=RandomForestRegressor(random_state=42),
    max_safe_exp: float = 50,
):
    
    
    # Get feature columns
    feature_columns = [col for col in train_set.columns if col not in ["Year", target_column]]
    
    # Make copies to avoid modifying original data
    train_data = train_set.copy()
    test_data = test_set.copy()
    
    # === FEATURE SKEWNESS DETECTION (full training data) ===
    skewed_features = []
    feature_skewness = {}
    
    for feature in feature_columns:
        if feature in train_data.columns:
            feature_skew = skew(train_data[feature])
            feature_skewness[feature] = feature_skew
            
            if abs(feature_skew) > feature_skew_threshold:
                skewed_features.append(feature)
    
    # === TARGET SKEWNESS DETECTION (full training data) ===
    target_skewness = skew(train_data[target_column])
    apply_target_log = target_skewness > target_skew_threshold
    
   
    
    # === APPLY TRANSFORMATIONS ===
    train_transformed = train_data.copy()
    test_transformed = test_data.copy()
    
    # Transform skewed features
    if skewed_features:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            for feature in skewed_features:
                # Transform training data
                train_feature_data = np.clip(train_transformed[feature].values, 0, 1e50)
                train_transformed[feature] = np.log1p(train_feature_data)
                
                # Apply same transformation to test data
                test_feature_data = np.clip(test_transformed[feature].values, 0, 1e50)
                test_transformed[feature] = np.log1p(test_feature_data)
                
    
    
    # Transform target (training data only)
    target_col_transformed = target_column
    if apply_target_log:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            # Transform training targets
            train_target_data = np.clip(train_transformed[target_column].values, 0, 1e50)
            train_transformed[target_column + '_log'] = np.log1p(train_target_data)
            target_col_transformed = target_column + '_log'
            

    
    # === PREPARE FEATURES FOR MODELING ===
    # Exclude non-feature columns
    exclude_cols = ["Year", target_column]
    if apply_target_log:
        exclude_cols.append(target_column + '_log')
    
    feature_cols = [col for col in train_transformed.columns if col not in exclude_cols]
    
    X_train = train_transformed[feature_cols]
    X_test = test_transformed[feature_cols]
    y_train = train_transformed[target_col_transformed]
    
    # === FEATURE SCALING ===
    final_scaler = type(scaler)(**scaler.get_params()) if hasattr(scaler, 'get_params') else StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    # === MODEL TRAINING ===
    final_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else RandomForestRegressor(random_state=42)
    final_model.fit(X_train_scaled, y_train)
    
    # === MAKE PREDICTIONS ===
    # Predictions on training set (for diagnostics)
    y_train_pred_transformed = final_model.predict(X_train_scaled)
    
    # Predictions on test set
    y_test_pred_transformed = final_model.predict(X_test_scaled)
    
    # === INVERSE TRANSFORM PREDICTIONS ===
    if apply_target_log:
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            # Inverse transform training predictions
            y_train_pred = np.expm1(np.clip(y_train_pred_transformed, None, max_safe_exp))
            
            # Inverse transform test predictions
            y_test_pred = np.expm1(np.clip(y_test_pred_transformed, None, max_safe_exp))
            
            # Safety clipping to reasonable values
            y_max_reasonable = train_data[target_column].max() * 100
            y_train_pred = np.clip(y_train_pred, 0, y_max_reasonable)
            y_test_pred = np.clip(y_test_pred, 0, y_max_reasonable)
            
    else:
        y_train_pred = y_train_pred_transformed
        y_test_pred = y_test_pred_transformed
    
    # === TRAINING SET EVALUATION (for diagnostics) ===
    y_train_original = train_data[target_column]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.metrics import mean_absolute_percentage_error, max_error
        
        r2_train = np.clip(r2_score(y_train_original, y_train_pred), -1e6, 1.0)
        mae_train = mean_absolute_error(y_train_original, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train_original, y_train_pred))
        mape_train = mean_absolute_percentage_error(y_train_original, y_train_pred)
        max_error_train = max_error(y_train_original, y_train_pred)
    
    # === TEST SET EVALUATION (if target column exists) ===
    test_metrics = None
    if target_column in test_data.columns:
        y_test_original = test_data[target_column]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            r2_test = np.clip(r2_score(y_test_original, y_test_pred), -1e6, 1.0)
            mae_test = mean_absolute_error(y_test_original, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test_original, y_test_pred))
            mape_test = mean_absolute_percentage_error(y_test_original, y_test_pred)
            max_error_test = max_error(y_test_original, y_test_pred)
            
            test_metrics = {
                'r2': r2_test,
                'mae': mae_test,
                'rmse': rmse_test,
                'mape': mape_test,
                'max_error': max_error_test
            }
    # === RETURN RESULTS ===
    results = {
        'test_predictions': y_test_pred,
        'train_predictions': y_train_pred,
        'model': final_model,
        'scaler': final_scaler,
        'transformation_info': {
            'skewed_features': skewed_features,
            'feature_skewness': feature_skewness,
            'target_skewness': target_skewness,
            'apply_target_log': apply_target_log,
            'target_column_transformed': target_col_transformed,
            'feature_columns': feature_cols
        },
        'training_metrics': {
            'r2': r2_train,
            'mae': mae_train,
            'rmse': rmse_train,
            'mape': mape_train,
            'max_error': max_error_train
        },
        'test_metrics': test_metrics  # None if no target in test set
    }
    
    return results

def plot_error_distribution(y_true, y_pred, title=""):
    """
    Plots the distribution of prediction errors (y_pred - y_true).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    - title: str, title of the plot
    """
    errors = np.array(y_pred) - np.array(y_true)

    plt.figure(figsize=(8, 5))
    sns.histplot(errors, kde=True, bins=30, color='skyblue')
    plt.title(title)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.legend()
    plt.tight_layout()
    plt.show()

def format_number(x):
    abs_x = abs(x)
    if abs_x >= 1e6:
        return f"{x / 1e6:.2f}M"
    elif abs_x >= 1e3:
        return f"{x / 1e3:.2f}K"
    else:
        return f"{x:.0f}"

def print_metrics(metrics_dict, title="Model Evaluation Metrics"):
    print(title)
    print(f"  R² Score       : {metrics_dict['r2']:.2f}")
    print(f"  MAE            : {format_number(metrics_dict['mae'])}")
    print(f"  RMSE           : {format_number(metrics_dict['rmse'])}")
    print(f"  MAPE           : {metrics_dict['mape'] * 100:.2f}%")
    print(f"  Max Error      : {format_number(metrics_dict['max_error'])}")
    
def train_final_model_and_predict(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    target_column: str,
    scaler=StandardScaler(),
    model=RandomForestRegressor(random_state=42)
):

    # === PREPARE FEATURES ===
    feature_columns = [col for col in train_set.columns if col not in ["Year", target_column]]
    
    X_train = train_set[feature_columns]
    y_train = train_set[target_column]
    X_test = test_set[feature_columns]
    
    # === FEATURE SCALING ===
    final_scaler = type(scaler)(**scaler.get_params()) if hasattr(scaler, 'get_params') else StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    # === MODEL TRAINING ===
    final_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else RandomForestRegressor(random_state=42)
    final_model.fit(X_train_scaled, y_train)
    
    # === MAKE PREDICTIONS ===
    y_train_pred = final_model.predict(X_train_scaled)
    y_test_pred = final_model.predict(X_test_scaled)
    
    # === TRAINING SET EVALUATION ===
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    max_error_train = max_error(y_train, y_train_pred)
    
    # === TEST SET EVALUATION (if target exists) ===
    test_metrics = None
    if target_column in test_set.columns:
        y_test = test_set[target_column]
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
        max_error_test = max_error(y_test, y_test_pred)
        
        test_metrics = {
            'r2': r2_test,
            'mae': mae_test,
            'rmse': rmse_test,
            'mape': mape_test,
            'max_error': max_error_test
        }
    
    # === RETURN RESULTS ===
    return {
        'test_predictions': y_test_pred,
        'train_predictions': y_train_pred,
        'model': final_model,
        'scaler': final_scaler,
        'feature_columns': feature_columns,
        'training_metrics': {
            'r2': r2_train,
            'mae': mae_train,
            'rmse': rmse_train,
            'mape': mape_train,
            'max_error': max_error_train
        },
        'test_metrics': test_metrics
    }
