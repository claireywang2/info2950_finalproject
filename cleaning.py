#!/usr/bin/env python3
"""
Board Game Geek Dataset Cleaning Script

This script performs initial data cleaning on the BoardGameGeek dataset:
- Removes duplicate game titles
- Filters out invalid/missing data (including Amazon price)
- Creates derived features (Game Age, Log Playtime, etc.)
- Normalizes features by creating NEW columns (preserving originals)
- Exports cleaned dataset

Final dataset spans 1990-2025.
"""

import pandas as pd
import numpy as np

RAW_DATA_PATH = "/content/boardgame-geek-dataset_organized.csv"
CLEANED_DATA_PATH = "/content/bgg_clean.csv"

MIN_YEAR = 1990
MAX_YEAR = 2025
MIN_RATINGS = 0

def load_raw_data(filepath):
    """Load the raw BoardGameGeek dataset and display initial info."""
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("RAW DATA LOADED")
    print("=" * 60)
    print(f"Initial shape: {df.shape}")
    return df


def remove_duplicates_and_missing(df):
    """
    Remove duplicate games and drop rows with missing critical fields.
    """
    print("\n" + "=" * 60)
    print("REMOVING DUPLICATES AND MISSING VALUES")
    print("=" * 60)

    # Remove duplicate game titles
    initial_count = len(df)
    df = df.drop_duplicates(subset=["boardgame"])
    print(f"Duplicates removed: {initial_count - len(df)}")

    # Drop rows with missing critical fields 
    critical_cols = ["avg_rating", "num_ratings", "release_year"]
    cols_to_drop = [c for c in critical_cols if c in df.columns]

    df = df.dropna(subset=cols_to_drop)
    print(f"Shape after removing NAs (checking {cols_to_drop}): {df.shape}")

    return df


def filter_valid_records(df):
    """
    Filter dataset to valid year ranges and positive rating counts.
    """
    print("\n" + "=" * 60)
    print("FILTERING VALID YEARS AND RATINGS")
    print("=" * 60)

    # Filter to reasonable year range
    df = df[df["release_year"].between(MIN_YEAR, MAX_YEAR)]
    print(f"After year filter ({MIN_YEAR}-{MAX_YEAR}): {len(df)} games")

    # Keep only games with positive rating counts
    df = df[pd.to_numeric(df["num_ratings"], errors="coerce") > MIN_RATINGS]
    print(f"After positive ratings filter: {len(df)} games")

    return df


def create_derived_features(df):
    """
    Engineer helper variables.
    """
    print("\n" + "=" * 60)
    print("CREATING DERIVED FEATURES")
    print("=" * 60)

    # 1. Game Age
    df['game_age'] = MAX_YEAR - df['release_year']

    # 2. Player Counts
    df["players_mid"] =(
        pd.to_numeric(df["min_players"], errors="coerce")
        + pd.to_numeric(df["max_players"], errors="coerce")
    ) / 2

    # 3. Playtime Mid
    df["playtime_mid"] = (
        pd.to_numeric(df["min_playtime"], errors="coerce")
        + pd.to_numeric(df["max_playtime"], errors="coerce")
    ) / 2

    # 4. Remove games with 0 playtime
    pre_filter = len(df)
    df = df[df['playtime_mid'] > 0]
    print(f"Removed {pre_filter - len(df)} games with 0 playtime.")

    # 5. Log Transforms
    df["log_num_ratings"] = np.log1p(pd.to_numeric(df["num_ratings"], errors="coerce"))
    df['log_playtime_mid'] = np.log1p(df['playtime_mid'])

    new_cols = ["game_age","players_mid", "playtime_mid", "log_playtime_mid", "log_num_ratings"]
    print(f"Derived features created: {new_cols}")

    return df


def normalize_features(df):
    """
    Normalize numerical features to a common scale (Z-score).

    NON-DESTRUCTIVE: Creates new columns with '_std' suffix.
    """
    print("\n" + "=" * 60)
    print("NORMALIZING FEATURES (CREATING NEW COLUMNS)")
    print("=" * 60)

    target_cols = [
        "avg_rating",
        "complexity",
        "players_mid",
        "playtime_mid",
        "log_num_ratings",
        "log_playtime_mid",
        "game_age"
    ]

    cols_to_norm = [c for c in target_cols if c in df.columns]

    print("Applying Z-Score Standardization (mean=0, std=1)...")
    for col in cols_to_norm:
        mean = df[col].mean()
        std= df[col].std()

        new_col_name = f"{col}_std"
        df[new_col_name] = (df[col] - mean) / std
        print(f" - Created {new_col_name}")

    return df


def save_cleaned_data(df, filepath):
    """Save the cleaned dataset to CSV."""
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATASET")
    print("=" * 60)

    df.to_csv(filepath, index=False)
    print(f"Saved to: {filepath}")
    print(f"Final shape: {df.shape}")


def validate_cleaning(df):
    """
    Verify cleaning results.
    """
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    validation_passed = True

    # 1. Duplicates
    if df.duplicated(subset = ["boardgame"]).sum() > 0:
        print("   WARNING: Duplicates found!")
        validation_passed = False
    else:
        print("   PASS: No duplicates")

    # 2. Year Range
    year_min = df["release_year"].min()
    if year_min < MIN_YEAR:
        print(f"   WARNING: Years found below {MIN_YEAR}!")
        validation_passed = False
    else:
        print(f"   PASS: Years start at {year_min} ( >= {MIN_YEAR})")

    # 3. Missing Values
    if df.isnull().sum().sum() > 0:
        print("   WARNING: Missing values detected")
    else:
        print("   PASS: No missing values")

    # 4. Normalization Quality (Check _std columns)
    print("\n4. Normalization quality (Target: Mean~0, Std~1):")
    norm_passed = True

    # Identify all columns ending in _std
    std_cols = [c for c in df.columns if c.endswith("_std")]

    if not std_cols:
        print("   WARNING: No normalized (_std) columns found!")
        validation_passed = False
    else:
        for col in std_cols:
            mu=df[col].mean()
            sigma=df[col].std()

            if abs(mu) > 0.01 or abs(sigma - 1.0) > 0.01:
                print(f"   WARNING: {col} invalid (Mean={mu:.2f}, Std={sigma:.2f})")
                norm_passed = False
                validation_passed = False
            else:
                print(f"   PASS: {col:<20} normalized.")

    if norm_passed and std_cols:
        print("   PASS: All checked columns appear normalized.")

    print("\n" + "=" * 60)
    if validation_passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 60)

def main():
    df = load_raw_data(RAW_DATA_PATH)

    # Pipeline
    df = remove_duplicates_and_missing(df)
    df= filter_valid_records(df)
    df = create_derived_features(df)
    df = normalize_features(df)
    
    save_cleaned_data(df, CLEANED_DATA_PATH)
    validate_cleaning(df)

if __name__ =='__main__':
    main()
