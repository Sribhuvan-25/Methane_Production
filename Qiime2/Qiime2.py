import pandas as pd
import numpy as np
from biom import Table
import json

# ----------------------------
# DATA LOADING & INITIAL CLEANING
# ----------------------------
df = pd.read_excel('Data/Data.xlsx', sheet_name=1, header=None, skiprows=1)

feature_vector = df.iloc[0:1086, :]
additional_features = df.iloc[1087:1099, :]

feature_vector.reset_index(drop=True, inplace=True)
additional_features.reset_index(drop=True, inplace=True)

feature_vector_T = feature_vector.transpose()
additional_features_T = additional_features.transpose()

full_data = pd.concat([feature_vector_T, additional_features_T], axis=1)
full_data.reset_index(drop=True, inplace=True)

df = full_data.drop(0, axis=0)

new_header = df.iloc[0]
df = df[1:]
df.columns = new_header
df.reset_index(drop=True, inplace=True)

df = df[~df.isin(['x']).any(axis=1)]

one_hot_encoded_data = pd.get_dummies(df, columns=['Digester', 'Source', 'Type', 'Waste', 'Biomass']).astype(int)
df_use = one_hot_encoded_data

# ----------------------------
# PREPARE METADATA & FEATURES
# ----------------------------
target_columns = ['ACE-km', 'H2-km']
df_cleaned = df_use[~df_use.isin(['x']).any(axis=1)]

digester_columns = [col for col in df_cleaned.columns if col.startswith('Digester')]
df_cleaned['Group'] = df_cleaned[digester_columns].idxmax(axis=1)

df_cleaned['sample-id'] = [f'sample{i+1}' for i in range(len(df_cleaned))]

metadata = df_cleaned[['sample-id'] + target_columns + ['Group']]
metadata.to_csv('sample-metadata.tsv', sep='\t', index=False)
print("Metadata file 'sample-metadata.tsv' created successfully!")

# ----------------------------
# CREATE BIOM FEATURE TABLE
# ----------------------------
excluded_cols = ['sample-id', 'Group'] + target_columns
feature_columns = [col for col in df_cleaned.columns if col not in excluded_cols]

features = df_cleaned[feature_columns].copy()
features.index = df_cleaned['sample-id']

# Strip whitespace just in case
features.columns = features.columns.str.strip()
features.index = features.index.str.strip()

# Check uniqueness of sample IDs (features.index)
if features.index.duplicated().any():
    dup_samples = features.index[features.index.duplicated()]
    print("Duplicate sample IDs found:")
    print(dup_samples)
    raise ValueError("Duplicate sample IDs exist. Make sure each sample-id is unique.")

# Check uniqueness of feature IDs (features.columns)
if features.columns.duplicated().any():
    dup_feats = features.columns[features.columns.duplicated()]
    print("Duplicate feature IDs found:")
    print(dup_feats)
    raise ValueError("Duplicate feature IDs exist. Make sure each feature name is unique.")

print("No duplicate observation IDs found based on direct checks.")

# Now we must align dimensions for BIOM:
# Observations = features (rows in biom), samples = sample IDs (columns in biom)
data_matrix = features.values.T  # Transpose so shape: (n_features, n_samples)
observation_ids = features.columns
sample_ids = features.index

# Double-check dimensions match
if data_matrix.shape[0] != len(observation_ids):
    raise ValueError(f"Mismatch: data has {data_matrix.shape[0]} obs, but {len(observation_ids)} observation_ids")
if data_matrix.shape[1] != len(sample_ids):
    raise ValueError(f"Mismatch: data has {data_matrix.shape[1]} samples, but {len(sample_ids)} sample_ids")

# ----------------------------
# EXPORT FEATURE TABLE TO TSV FILE
# ----------------------------
# Export the feature table to a TSV file for conversion to BIOM format
features.T.to_csv('feature-table.tsv', sep='\t', header=True, index=True)
print("Feature table exported as 'feature-table.tsv'")

# ----------------------------
# CREATE BIOM TABLE (Optional)
# ----------------------------
# This section creates the BIOM table in Python but may not be needed if using biom convert CLI.
feature_table = Table(data_matrix, observation_ids=observation_ids, sample_ids=sample_ids)

# Convert feature_table to a JSON string with format 2.1.0
biom_data = json.loads(feature_table.to_json("Generated by Python"))

# Update the format field explicitly to ensure compatibility with BIOM v2.1.0
biom_data["format"] = "Biological Observation Matrix 2.1.0"
biom_data["format_url"] = "http://biom-format.org"

# Save the updated BIOM file
with open('feature-table-v210.biom', 'w') as biom_file:
    json.dump(biom_data, biom_file, indent=4)

print("Feature table saved with forced BIOM v2.1.0 format.")