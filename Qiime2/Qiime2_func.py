import pandas as pd
from biom import Table
import json

# ----------------------------
# FUNCTIONS
# ----------------------------

def load_cleaned_data(filepath):
    # Load the preprocessed data
    df = pd.read_csv(filepath)
    print(f"Loaded data from {filepath} with shape {df.shape}")
    
    # Create sample-id column if it doesn't exist
    if 'sample-id' not in df.columns:
        df.insert(0, 'sample-id', [f'sample{i+1}' for i in range(len(df))])
        print("'sample-id' column created.")
    
    return df

def assign_group(df, digester_prefix='Digester'):
    # Identify digester columns and assign groups based on maximum value
    df = df[~df.isin(['x']).any(axis=1)]
    digester_columns = [col for col in df.columns if col.startswith(digester_prefix)]
    if not digester_columns:
        raise KeyError("No columns found with the specified digester prefix.")
    df = df.copy()  # Ensure we operate on a copy of the DataFrame to avoid warnings
    df['Group'] = df[digester_columns].idxmax(axis=1)
    print("Group assignment completed based on digester columns.")
    return df

def create_metadata(df, target_columns, group_column):
    # Check if required columns exist
    required_columns = ['sample-id'] + target_columns + [group_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Create and save metadata
    metadata = df[required_columns]
    metadata.to_csv('sample-metadata.tsv', sep='\t', index=False)
    print("Metadata file 'sample-metadata.tsv' created successfully!")

def prepare_features(df, excluded_columns):
    feature_columns = [col for col in df.columns if col not in excluded_columns]
    features = df[feature_columns].copy()
    features.index = df['sample-id']
    features.columns = features.columns.str.strip()
    features.index = features.index.str.strip()

    # Check for duplicates
    if features.index.duplicated().any():
        dup_samples = features.index[features.index.duplicated()]
        print("Duplicate sample IDs found:")
        print(dup_samples)
        raise ValueError("Duplicate sample IDs exist. Make sure each sample-id is unique.")
    if features.columns.duplicated().any():
        dup_feats = features.columns[features.columns.duplicated()]
        print("Duplicate feature IDs found:")
        print(dup_feats)
        raise ValueError("Duplicate feature IDs exist. Make sure each feature name is unique.")
    
    print("No duplicate observation IDs found.")
    return features

def export_feature_table(features):
    # Save the feature table with samples as rows and features as columns
    features.to_csv('feature-table.tsv', sep='\t', header=True, index=True)
    print("Feature table exported as 'feature-table.tsv' with samples as rows.")

def create_biom_table(features):
    data_matrix = features.values  # No transpose here; features as columns, samples as rows
    observation_ids = features.columns
    sample_ids = features.index

    # Check dimensions
    if data_matrix.shape[1] != len(observation_ids):
        raise ValueError(f"Mismatch: data has {data_matrix.shape[1]} obs, but {len(observation_ids)} observation_ids")
    if data_matrix.shape[0] != len(sample_ids):
        raise ValueError(f"Mismatch: data has {data_matrix.shape[0]} samples, but {len(sample_ids)} sample_ids")
    
    feature_table = Table(data_matrix.T, observation_ids=observation_ids, sample_ids=sample_ids)
    biom_data = json.loads(feature_table.to_json("Generated by Python"))
    biom_data["format"] = "Biological Observation Matrix 2.1.0"
    biom_data["format_url"] = "http://biom-format.org"
    
    with open('feature-table-v210.biom', 'w') as biom_file:
        json.dump(biom_data, biom_file, indent=4)
    
    print("Feature table saved with forced BIOM v2.1.0 format.")

# ----------------------------
# MAIN EXECUTION
# ----------------------------

def main():
    # Load preprocessed data
    filepath = 'Data/New_data.csv'
    df = load_cleaned_data(filepath)

    # Assign groups based on digester columns
    df = assign_group(df)

    # Define target columns and grouping column
    target_columns = ['ACE-km', 'H2-km']
    group_column = 'Group'

    # Create metadata file
    create_metadata(df, target_columns, group_column)

    # Prepare features
    excluded_columns = ['sample-id', 'Group'] + target_columns
    features = prepare_features(df, excluded_columns)

    # Export feature table
    export_feature_table(features)

    # Create BIOM table
    create_biom_table(features)

if __name__ == "__main__":
    main()