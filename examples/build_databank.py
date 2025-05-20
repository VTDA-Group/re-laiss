"""
Building a New Dataset Bank for reLAISS

This script demonstrates how to build a new dataset bank for reLAISS, including:
1. Adding extinction corrections (A_V)
2. Joining new lightcurve features
3. Handling missing values
4. Building the final dataset bank
5. Using different feature combinations for nearest neighbor search

The process involves several steps:
1. Add extinction corrections to the large dataset bank
2. Join new lightcurve features to the small dataset bank
3. Handle missing values using KNN imputation
4. Build the final dataset bank with all features
5. Demonstrate custom feature selection for nearest neighbor search
"""

import os
import pandas as pd
import numpy as np
from sfdmap2 import sfdmap
from sklearn.impute import KNNImputer
from relaiss.features import build_dataset_bank
from relaiss import constants
import relaiss as rl

def add_extinction_corrections(df, path_to_sfd_folder):
    """Add extinction corrections (A_V) to the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with 'ra' and 'dec' columns
    path_to_sfd_folder : str
        Path to SFD dust map files
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added A_V column
    """
    print("Adding extinction corrections...")
    m = sfdmap.SFDMap(path_to_sfd_folder)
    RV = 3.1  # Standard value for Milky Way
    ebv = m.ebv(df['ra'].values, df['dec'].values)
    df['A_V'] = RV * ebv
    return df

def join_features(df_small, df_large, key='ztf_object_id'):
    """Join new features from large dataset to small dataset.
    
    Parameters
    ----------
    df_small : pandas.DataFrame
        Small dataset bank
    df_large : pandas.DataFrame
        Large dataset bank with additional features
    key : str
        Column to join on
        
    Returns
    -------
    pandas.DataFrame
        Merged dataset with all features
    """
    print("Joining features from large dataset...")
    extra_features = [col for col in df_large.columns if col not in df_small.columns]
    merged_df = df_small.merge(df_large[[key] + extra_features], on=key, how='left')
    return merged_df

def handle_missing_values(df, feature_names):
    """Handle missing values using KNN imputation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    feature_names : list
        List of feature column names
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed values
    """
    print("Handling missing values...")
    X = df[feature_names]
    feat_imputer = KNNImputer(weights='distance').fit(X)
    imputed_filt_arr = feat_imputer.transform(X)
    
    imputed_df = pd.DataFrame(imputed_filt_arr, columns=feature_names)
    imputed_df.index = df.index
    df[feature_names] = imputed_df
    return df

def main():
    # Create necessary directories
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../data/sfddata-master', exist_ok=True)
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./sfddata-master', exist_ok=True)
    
    # Step 1: Add extinction corrections to large dataset bank
    print("\nStep 1: Adding extinction corrections")
    df_large = pd.read_csv("../data/large_df_bank.csv")
    df_large = add_extinction_corrections(df_large, '../data/sfddata-master')
    df_large.to_csv("../data/large_df_bank_wAV.csv", index=False)
    print("Saved large dataset bank with extinction corrections")
    
    # Step 2: Join new lightcurve features to small dataset bank
    print("\nStep 2: Joining features")
    df_small = pd.read_csv("../data/small_df_bank_re_laiss.csv")
    merged_df = join_features(df_small, df_large)
    
    # Clean the merged dataset
    lc_feature_names = constants.lc_features_const.copy()
    host_feature_names = constants.host_features_const.copy()
    small_final_df = merged_df.replace([np.inf, -np.inf, -999], np.nan).dropna(
        subset=lc_feature_names + host_feature_names
    )
    small_final_df.to_csv("../data/small_hydrated_df_bank_re_laiss.csv", index=False)
    print("Saved hydrated small dataset bank")
    
    # Step 3: Handle missing values in large dataset
    print("\nStep 3: Handling missing values")
    raw_host_feature_names = constants.raw_host_features_const.copy()
    raw_dataset_bank = pd.read_csv('../data/large_df_bank_wAV.csv')
    print("Shape of raw dataset bank:", raw_dataset_bank.shape)
    
    imputed_df_bank = handle_missing_values(
        raw_dataset_bank, 
        lc_feature_names + raw_host_feature_names
    )
    print("Shape of imputed dataset bank:", imputed_df_bank.shape)
    
    # Step 4: Build final dataset bank
    print("\nStep 4: Building final dataset bank")
    dataset_bank = build_dataset_bank(
        raw_df_bank=imputed_df_bank,
        av_in_raw_df_bank=True,
        path_to_sfd_folder="../data/sfddata-master",
        building_entire_df_bank=True
    )
    
    # Clean and save final dataset
    final_dataset_bank = dataset_bank.replace(
        [np.inf, -np.inf, -999], np.nan
    ).dropna(subset=lc_feature_names + host_feature_names)
    
    print("Shape of final dataset bank:", final_dataset_bank.shape)
    final_dataset_bank.to_csv('../data/large_final_df_bank_new_lc_feats.csv', index=False)
    print("Successfully saved final dataset bank!")
    
    # Step 5: Demonstrate different feature combinations for search
    print("\nStep 5: Using different feature combinations")
    
    # Define default feature sets from constants
    default_lc_features = constants.lc_features_const.copy()
    default_host_features = constants.host_features_const.copy()
    
    # Initialize client
    client = rl.ReLAISS()
    client.load_reference(
        path_to_sfd_folder='./sfddata-master'
    )
    
    # Example 1: Using only lightcurve features (no host features)
    print("\nExample 1: Using only lightcurve features")
    lc_only_client = rl.ReLAISS()
    lc_only_client.load_reference(
        path_to_sfd_folder='./sfddata-master',
        lc_features=default_lc_features,  # Use default lightcurve features
        host_features=[],  # Empty list means no host features
    )
    
    # Find neighbors using only lightcurve features
    neighbors_df_lc_only = lc_only_client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        n=5,
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures/lc_only'
    )
    print("\nNearest neighbors using only lightcurve features:")
    print(neighbors_df_lc_only)
    
    # Example 2: Using only host features (no lightcurve features)
    print("\nExample 2: Using only host features")
    host_only_client = rl.ReLAISS()
    host_only_client.load_reference(
        path_to_sfd_folder='./sfddata-master',
        lc_features=[],  # Empty list means no lightcurve features
        host_features=default_host_features,  # Use default host features
    )
    
    # Find neighbors using only host features
    neighbors_df_host_only = host_only_client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        n=5,
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures/host_only'
    )
    print("\nNearest neighbors using only host features:")
    print(neighbors_df_host_only)
    
    # Example 3: Using custom feature subset
    print("\nExample 3: Using custom feature subset")
    # Select specific lightcurve and host features
    custom_lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
    custom_host_features = ['host_ra', 'host_dec', 'gKronMag', 'rKronMag']
    
    custom_client = rl.ReLAISS()
    custom_client.load_reference(
        path_to_sfd_folder='./sfddata-master',
        lc_features=custom_lc_features,  # Custom subset of lightcurve features
        host_features=custom_host_features,  # Custom subset of host features
    )
    
    # Find neighbors with custom feature subset
    neighbors_df_custom = custom_client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        n=5,
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures/custom_features'
    )
    print("\nNearest neighbors using custom feature subset:")
    print(neighbors_df_custom)
    
    # Example 4: Setting feature importance with feature weighting
    print("\nExample 4: Using feature weighting")
    # Regular search prioritizing lightcurve features
    neighbors_df_lc_weighted = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        n=5,
        weight_lc_feats_factor=3.0,  # Strongly prioritize lightcurve features
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures/lc_weighted'
    )
    print("\nNearest neighbors with lightcurve features weighted 3x:")
    print(neighbors_df_lc_weighted)
    
    # Now prioritize host features by using a factor < 1
    neighbors_df_host_weighted = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        n=5,
        weight_lc_feats_factor=0.3,  # Prioritize host features
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures/host_weighted'
    )
    print("\nNearest neighbors with host features given higher weight:")
    print(neighbors_df_host_weighted)

if __name__ == "__main__":
    main() 