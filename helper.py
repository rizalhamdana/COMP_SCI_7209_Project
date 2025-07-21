import pandas as pd
import os

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