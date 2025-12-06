#!/usr/bin/env python3
"""
Board Game Geek Dataset Cleaning Script

This script performs initial data cleaning on the BoardGameGeek dataset:
- Removes duplicate game titles
- Filters out invalid/missing data
- Creates derived features for analysis
- Exports cleaned dataset for downstream use

Final dataset spans 1925-2025 with ~1,975 games and 60 columns.
"""

import pandas as pd
import numpy as np

RAW_DATA_PATH = "/content/boardgame-geek-dataset_organized.csv"
CLEANED_DATA_PATH = "/content/bgg_clean.csv"

MIN_YEAR = 1900
MAX_YEAR = 2025
MIN_RATINGS = 0  # Exclusive - games must have > 0 ratings


def load_raw_data(filepath):
    """Load the raw BoardGameGeek dataset and display initial info."""
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("RAW DATA LOADED")
    print("=" * 60)
    print(f"Initial shape: {df.shape}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    return df


def remove_duplicates_and_missing(df):
    """
    Remove duplicate games and drop rows with missing critical fields.

    Cleaning steps:
    1. Drop duplicate boardgame titles (keeps first occurrence)
    2. Drop rows missing avg_rating, num_ratings, or release_year
       - These are essential for any rating-based or temporal analysis
    """
    print("\n" + "=" * 60)
    print("REMOVING DUPLICATES AND MISSING VALUES")
    print("=" * 60)

    # Remove duplicate game titles
    initial_count = len(df)
    df = df.drop_duplicates(subset=["boardgame"])
    duplicates_removed = initial_count - len(df)
    print(f"Duplicates removed: {duplicates_removed}")

    # Drop rows with missing critical fields
    df = df.dropna(subset=["avg_rating", "num_ratings", "release_year"])
    print(f"Shape after removing duplicates/NAs: {df.shape}")

    return df


def filter_valid_records(df):
    """
    Filter dataset to valid year ranges and positive rating counts.

    Filtering logic:
    1. Keep only games released between MIN_YEAR and MAX_YEAR
       - Removes outliers/data entry errors (e.g., year 1, year 3000)
    2. Keep only games with positive rating counts
       - Ensures we have actual user feedback data
       - Converts num_ratings to numeric, coercing errors to NaN
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
    print(f"Final shape: {df.shape}")

    return df


def create_derived_features(df):
    """
    Engineer helper variables for exploratory analysis.

    New features:
    1. players_mid: Average of min_players and max_players
       - Gives single value for typical player count
    2. playtime_mid: Average of min_playtime and max_playtime
       - Represents typical game duration
    3. log_num_ratings: Log transform of num_ratings
       - Reduces right skew in rating counts for better visualization/modeling
       - Uses log1p to handle zeros gracefully
    """
    print("\n" + "=" * 60)
    print("CREATING DERIVED FEATURES")
    print("=" * 60)

    # Calculate midpoint player count
    df["players_mid"] = (
        pd.to_numeric(df["min_players"], errors="coerce")
        + pd.to_numeric(df["max_players"], errors="coerce")
    ) / 2

    # Calculate midpoint playtime
    df["playtime_mid"] = (
        pd.to_numeric(df["min_playtime"], errors="coerce")
        + pd.to_numeric(df["max_playtime"], errors="coerce")
    ) / 2

    # Log transform rating counts to reduce skew
    df["log_num_ratings"] = np.log1p(pd.to_numeric(df["num_ratings"], errors="coerce"))

    new_cols = ["players_mid", "playtime_mid", "log_num_ratings"]
    print(f"New columns added: {[c for c in new_cols if c in df.columns]}")
    print("\nFirst 3 rows with new features:")
    print(df.head(3))

    return df


def save_cleaned_data(df, filepath):
    """Save the cleaned dataset to CSV."""
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATASET")
    print("=" * 60)

    df.to_csv(filepath, index=False)
    print(f"Saved to: {filepath}")
    print(f"Final shape: {df.shape}")
    print(f"Total games: {len(df)}")
    print(f"Total columns: {len(df.columns)}")


def validate_cleaning(df):
    """
    Verify cleaning results and display data quality metrics.

    Validation checks:
    1. Confirm no duplicate boardgame titles remain
    2. Show actual year range in cleaned data
    3. Confirm no non-positive num_ratings values remain
    4. Check for missing values in critical columns
    """
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    validation_passed = True

    duplicate_count = df.duplicated(subset=["boardgame"]).sum()
    print(f"\n1. Duplicate boardgame titles: {duplicate_count}")
    if duplicate_count > 0:
        print("   WARNING: Duplicates found!")
        validation_passed = False
    else:
        print("   PASS: No duplicates")

    year_min = df["release_year"].min()
    year_max = df["release_year"].max()
    print(f"\n2. Year range: {year_min} - {year_max}")

    current_year = 2025
    if year_min < 1800 or year_max > current_year:
        print(f"   WARNING: Unusual year range detected!")
        validation_passed = False
    else:
        print("   PASS: Year range looks reasonable")

    nonpositive_ratings = (pd.to_numeric(df["num_ratings"], errors="coerce") <= 0).sum()
    print(f"\n3. Non-positive num_ratings: {nonpositive_ratings}")

    if nonpositive_ratings > 0:
        print("   WARNING: Non-positive ratings found!")
        validation_passed = False
    else:
        print("   PASS: All ratings are positive")

    print("\n4. Missing values check:")
    missing_values = df.isnull().sum()
    critical_columns = ["boardgame", "release_year", "num_ratings"]

    for col in critical_columns:
        if col in df.columns:
            missing = missing_values[col]
            print(f"   - {col}: {missing} missing")
            if missing > 0:
                print(f"     WARNING: Missing values in {col}!")
                validation_passed = False

    if missing_values.sum() == 0:
        print("   PASS: No missing values in critical columns")

    print("\n5. Summary statistics:")
    print(f"   - Total rows: {len(df)}")
    if "num_ratings" in df.columns:
        ratings_numeric = pd.to_numeric(df["num_ratings"], errors="coerce")
        print(f"   - Mean ratings: {ratings_numeric.mean():.2f}")
        print(f"   - Median ratings: {ratings_numeric.median():.2f}")

    print("\n" + "=" * 60)
    if validation_passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 60)

    return validation_passed


def main():
    """Execute the complete data cleaning pipeline."""
    df = load_raw_data(RAW_DATA_PATH)
    df = create_derived_features(filter_valid_records(remove_duplicates_and_missing(df)))

    save_cleaned_data(df, CLEANED_DATA_PATH)
    validate_cleaning(df)


if __name__ == "__main__":
    main()
