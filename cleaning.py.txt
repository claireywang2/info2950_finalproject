# Data Cleaning

As an *initial* data-clean, we removed duplicate game titles and dropped rows missing key fields (avg_rating, num_ratings, release_year). We kept games released between 1900 and 2025 (resulting dataset spans 1925–2025) and only those with a positive number of ratings. For quick exploration, we added three helper variables: players_mid (mean of min_players and max_players), playtime_mid (mean of min_playtime and max_playtime), and log_num_ratings (log transform to reduce skew). The cleaned dataset contains 1,975 rows and 60 columns and is saved as bgg_clean.csv.

### Load raw dataset
Load the Kaggle dataset to preview its shape and structure before cleaning.
"""

# LOAD RAW DATA
# Import pandas and numpy, then load the dataset from local path.
import pandas as pd
import numpy as np

CSV_PATH = "/content/boardgame-geek-dataset_organized.csv"  # local raw, not in repo
df = pd.read_csv(CSV_PATH)

# Print shape and show first few rows.
print("raw shape:", df.shape)
display(df.head(3))

"""### Remove duplicates and drop missing values
Eliminate duplicate titles and drop rows missing critical variables.
"""

# REMOVE DUPLICATES AND DROP NA
# Drop duplicate rows based on boardgame title.
df = df.drop_duplicates(subset=["boardgame"])

# Drop rows with missing core fields.
df = df.dropna(subset=["avg_rating", "num_ratings", "release_year"])

print("after filters:", df.shape)

"""### Keep valid release years and positive rating counts
Restrict games to the reasonable year range and ensure positive review counts.
"""

# FILTER VALID YEARS AND RATINGS
df = df[df["release_year"].between(1900, 2025)]
df = df[pd.to_numeric(df["num_ratings"], errors="coerce") > 0]
print("after year and rating filters:", df.shape)

"""### Create helper variables
Compute midpoint player count, midpoint play time, and log transformed review counts.
"""

# FEATURE ENGINEERING
df["players_mid"] = (pd.to_numeric(df["min_players"], errors="coerce") +
                     pd.to_numeric(df["max_players"], errors="coerce")) / 2

df["playtime_mid"] = (pd.to_numeric(df["min_playtime"], errors="coerce") +
                      pd.to_numeric(df["max_playtime"], errors="coerce")) / 2

df["log_num_ratings"] = np.log1p(pd.to_numeric(df["num_ratings"], errors="coerce"))

print("new columns added:", [c for c in ["players_mid","playtime_mid","log_num_ratings"] if c in df.columns])
display(df.head(3))

"""### Save cleaned dataset
Save the processed data as bgg_clean.csv for later analysis and EDA.
"""

# SAVE CLEANED DATASET
OUT_PATH = "/content/bgg_clean.csv"
df.to_csv(OUT_PATH, index=False)
print("saved cleaned dataset to:", OUT_PATH)
print("cleaned shape:", df.shape)

"""### Verify cleaning results
Confirm that duplicates and invalid records are removed.

"""

# VALIDATION
print("Duplicate boardgame titles:", df.duplicated(subset=["boardgame"]).sum())
print("Year range:", df["release_year"].min(), "-", df["release_year"].max())
print("Nonpositive num_ratings:", (pd.to_numeric(df["num_ratings"], errors="coerce") <= 0).sum())

