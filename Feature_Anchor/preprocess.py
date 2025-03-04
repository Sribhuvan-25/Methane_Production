import pandas as pd
import numpy as np
import os

def create_directories():
    """Create directories for organizing outputs"""
    directories = [
        'data',
        'features',
        'results/plots',
        'results/metrics'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_classified_dataset(input_path, output_path):
    """
    Create classified dataset and feature files
    """
    # Load the original dataset
    df = pd.read_csv(input_path)
    
    # Remove 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Define the key features and their classifications
    key_features = {
        'd__Archaea;p__Halobacterota;c__Methanosarcinia;o__Methanosarciniales;f__Methanosaetaceae;g__Methanosaeta': 'acetoclastic',
        'd__Archaea;p__Halobacterota;c__Methanomicrobia;o__Methanomicrobiales;f__Methanoregulaceae;g__Methanolinea': 'hydrogenotrophic',
        'd__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium': 'hydrogenotrophic',
        'd__Archaea;p__Halobacterota;c__Methanomicrobia;o__Methanomicrobiales;f__Methanospirillaceae;g__Methanospirillum': 'hydrogenotrophic',
        'd__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Smithellaceae;g__Smithella': 'syntrophic',
        'd__Bacteria;p__Desulfobacterota;c__Syntrophorhabdia;o__Syntrophorhabdales;f__Syntrophorhabdaceae;g__Syntrophorhabdus': 'syntrophic',
        'd__Bacteria;p__Desulfobacterota;c__Syntrophobacteria;o__Syntrophobacterales;f__Syntrophobacteraceae;g__Syntrophobacter': 'syntrophic',
        'd__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Syner-01': 'syntrophic',
        'd__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__uncultured;g__uncultured': 'syntrophic',
        'd__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__uncultured': 'syntrophic',
        'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Rikenellaceae;g__DMER64': 'syntrophic',
        'd__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Thermovirga': 'syntrophic',
        'd__Bacteria;p__Firmicutes;c__Syntrophomonadia;o__Syntrophomonadales;f__Syntrophomonadaceae;g__Syntrophomonas': 'syntrophic',
        'd__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Syntrophaceae;g__Syntrophus': 'syntrophic',
        'd__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__JGI-0000079-D21': 'syntrophic',
        'd__Bacteria;p__Desulfobacterota;c__Desulfuromonadia;o__Geobacterales;f__Geobacteraceae;__': 'syntrophic',
        'd__Bacteria;p__Firmicutes;c__Desulfotomaculia;o__Desulfotomaculales;f__Desulfotomaculales;g__Pelotomaculum': 'syntrophic'
    }
    
    # Check which features are available
    missing_features = {k: v for k, v in key_features.items() if k not in df.columns}
    available_features = {k: v for k, v in key_features.items() if k in df.columns}
    
    print("\nFeature Analysis:")
    print(f"Total features needed: {len(key_features)}")
    print(f"Features found: {len(available_features)}")
    print(f"Features missing: {len(missing_features)}")
    
    # Create lists for each classification
    acetoclastic_features = [k for k, v in available_features.items() if v == 'acetoclastic']
    hydrogenotrophic_features = [k for k, v in available_features.items() if v == 'hydrogenotrophic']
    syntrophic_features = [k for k, v in available_features.items() if v == 'syntrophic']
    
    print("\nBreakdown by classification:")
    print(f"acetoclastic: {len(acetoclastic_features)}/{sum(1 for v in key_features.values() if v == 'acetoclastic')} features found")
    print(f"hydrogenotrophic: {len(hydrogenotrophic_features)}/{sum(1 for v in key_features.values() if v == 'hydrogenotrophic')} features found")
    print(f"syntrophic: {len(syntrophic_features)}/{sum(1 for v in key_features.values() if v == 'syntrophic')} features found")
    
    # Create classified dataset
    df_classified = pd.DataFrame()
    
    # Add features
    for feature in available_features.keys():
        df_classified[feature] = df[feature]
    
    # Add target columns
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    
    print("\nTarget columns status:")
    for col in target_columns:
        if col in df.columns:
            df_classified[col] = df[col]
            print(f"Found: {col}")
        else:
            print(f"Missing: {col}")
    
    # Save the processed dataset
    df_classified.to_csv(output_path, index=False)
    
    # Save feature classifications
    classifications_df = pd.DataFrame({
        'feature': list(available_features.keys()),
        'classification': list(available_features.values())
    })
    classifications_df.to_csv('features/feature_classifications.csv', index=False)
    
    # Save separate files for each classification
    pd.DataFrame({'feature': acetoclastic_features}).to_csv('features/acetoclastic_features.csv', index=False)
    pd.DataFrame({'feature': hydrogenotrophic_features}).to_csv('features/hydrogenotrophic_features.csv', index=False)
    pd.DataFrame({'feature': syntrophic_features}).to_csv('features/syntrophic_features.csv', index=False)
    
    print(f"\nDataset shapes:")
    print(f"Original: {df.shape}")
    print(f"Classified: {df_classified.shape}")
    print(f"\nFeature files created in 'features' directory:")
    print("- feature_classifications.csv (all features and their classifications)")
    print("- acetoclastic_features.csv")
    print("- hydrogenotrophic_features.csv")
    print("- syntrophic_features.csv")
    
    return df_classified, classifications_df

if __name__ == "__main__":
    # Create directory structure
    create_directories()
    
    # Set paths
    input_path = "../Data/New_data.csv"
    output_path = "data/classified_data.csv"
    
    # Create classified dataset and feature files
    df_classified, feature_classifications = create_classified_dataset(input_path, output_path) 