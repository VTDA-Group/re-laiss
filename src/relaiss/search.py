import os
import time

import annoy
import antares_client
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.decomposition import PCA

from . import constants
from .fetch import get_timeseries_df, get_TNS_data
from .plotting import plot_hosts, plot_lightcurves

def primer(
    lc_ztf_id,
    theorized_lightcurve_df,
    dataset_bank_path,
    path_to_timeseries_folder,
    path_to_sfd_folder,
    save_timeseries=False,
    host_ztf_id=None,
    lc_features=[],
    host_features=[],
    num_sims=0,
    preprocessed_df=None,
    random_seed=42,
):
    """Assemble input feature vectors (and MC replicas) for a query object.

    This function combines lightcurve and host galaxy features to create a feature vector
    for similarity search. It can optionally swap in a different host galaxy and generate
    Monte Carlo replicas for uncertainty propagation.

    Parameters
    ----------
    lc_ztf_id : str | None
        ZTF ID of the transient to query. Mutually exclusive with theorized_lightcurve_df.
    theorized_lightcurve_df : pandas.DataFrame | None
        Pre-computed ANTARES-style lightcurve for a theoretical model.
        Mutually exclusive with lc_ztf_id.
    dataset_bank_path : str | Path
        Path to the dataset bank CSV file containing feature data.
    path_to_timeseries_folder : str | Path
        Directory for storing/loading timeseries data.
    path_to_sfd_folder : str | Path
        Directory containing SFD dust map files for extinction correction.
    save_timeseries : bool, default False
        Whether to save the timeseries data to disk.
    host_ztf_id : str | None, default None
        If provided, replace the query object's host features with those of this transient.
    lc_features : list[str], default []
        Names of lightcurve feature columns to extract.
    host_features : list[str], default []
        Names of host galaxy feature columns to extract.
    num_sims : int, default 0
        Number of Monte Carlo perturbations for uncertainty propagation.
    preprocessed_df : pandas.DataFrame | None, default None
        Pre-processed dataframe with imputed features. If provided, this is used
        instead of loading and processing the raw dataset bank.

    Returns
    -------
    dict
        Dictionary containing:
        - host_ztf_id: ZTF ID of swapped host (if any)
        - host_tns_name: TNS name of swapped host
        - host_tns_cls: Spectral class of swapped host
        - host_tns_z: Redshift of swapped host
        - host_ztf_id_in_dataset_bank: Whether host is in dataset bank
        - host_galaxy_ra/dec: Host galaxy coordinates
        - lc_ztf_id: ZTF ID of input transient
        - lc_tns_name/cls/z: TNS info for input transient
        - lc_ztf_id_in_dataset_bank: Whether transient is in dataset bank
        - locus_feat_arr: Combined feature array
        - locus_feat_arrs_mc_l: List of MC perturbed feature arrays
        - lc_galaxy_ra/dec: Input transient host coordinates
        - lc_feat_names: List of lightcurve feature names
        - host_feat_names: List of host feature names

    Raises
    ------
    ValueError
        If both lc_ztf_id and theorized_lightcurve_df are provided
        If neither lc_ztf_id nor theorized_lightcurve_df is provided
        If theorized_lightcurve_df is provided without host_ztf_id
        If required features are missing from dataset bank
        If NaN features are found in timeseries data
    """
    feature_names = lc_features + host_features
    host_locus_feat_arr = None
    host_tns_name = host_tns_cls = host_tns_z = None
    host_ztf_id_in_dataset_bank = None

    if lc_ztf_id is not None and theorized_lightcurve_df is not None:
        print(
            "Expected only one of theorized_lightcurve_df and transient_ztf_id. Try again!"
        )
        raise ValueError(
            "Cannot provide both a transient ZTF ID and a theorized lightcurve."
        )
    if lc_ztf_id is None and theorized_lightcurve_df is None:
        print("Requires one of theorized_lightcurve_df or transient_ztf_id. Try again!")
        raise ValueError(
            "Transient ZTF ID and theorized lightcurve cannot both be None."
        )
    if theorized_lightcurve_df is not None and host_ztf_id is None:
        print(
            "Inputing theorized_lightcurve_df requires host_ztf_id_to_swap_in. Try again!"
        )
        raise ValueError(
            "If providing a theorized lightcurve, must also provide a host galaxy ZTF ID."
        )

    host_galaxy_ra = None
    host_galaxy_dec = None
    lc_galaxy_ra = None
    lc_galaxy_dec = None

    # Loop through lightcurve object and host object to create feature array
    for ztf_id, host_loop in [(lc_ztf_id, False), (host_ztf_id, True)]:

        if host_loop and ztf_id is None:
            continue

        if ztf_id is None and not host_loop:
            ts = get_timeseries_df(
                ztf_id=None,
                theorized_lightcurve_df=theorized_lightcurve_df,
                path_to_timeseries_folder=path_to_timeseries_folder,
                path_to_sfd_folder=path_to_sfd_folder,
                path_to_dataset_bank=dataset_bank_path,
                save_timeseries=save_timeseries,
                swapped_host=False,
            )

            # Skip host loop if host galaxy to swap is not provided
            ts = ts.dropna(subset=lc_features)

            locus_feat_series = ts[lc_features].iloc[-1]
            locus_feat_arr    = locus_feat_series.values

            lc_tns_name, lc_tns_cls, lc_tns_z = "No TNS", "---", -99
            lc_ztf_id_in_dataset_bank = False
            lc_galaxy_ra, lc_galaxy_dec = np.nan, np.nan

            ztf_id_in_dataset_bank = False

            lc_locus_feat_arr = locus_feat_series

            continue

        # Check if ztf_id is in dataset bank
        if preprocessed_df is not None:
            df_bank = preprocessed_df.copy()
            # Make sure we have ztf_object_id as expected
            if 'ZTFID' in df_bank.columns:
                df_bank = df_bank.rename(columns={'ZTFID': 'ztf_object_id'})
            if 'ztf_object_id' not in df_bank.columns:
                raise ValueError("preprocessed_df must contain a 'ztf_object_id' column")
            df_bank = df_bank.set_index("ztf_object_id", drop=True)
        else:
            df_bank = pd.read_csv(dataset_bank_path, low_memory=False)
            # Normalize column names - make sure we use ztf_object_id consistently
            if 'ZTFID' in df_bank.columns:
                df_bank = df_bank.rename(columns={'ZTFID': 'ztf_object_id'})
            df_bank = df_bank.set_index("ztf_object_id", drop=True)

        # Check to make sure all features are in the dataset bank
        missing_cols = [col for col in feature_names if col not in df_bank.columns]
        if missing_cols:
            raise KeyError(
                f"KeyError: The following columns are not in the raw data provided: {missing_cols}. Abort!"
            )

        # ------------------------------------------------------------------
        if host_loop and ztf_id not in df_bank.index:          # ← add host_loop check
            print(f"{ztf_id} not in dataset bank – extracting host features directly.")

            ts_host = get_timeseries_df(
                ztf_id                  = ztf_id,
                theorized_lightcurve_df = None,
                path_to_timeseries_folder = path_to_timeseries_folder,
                path_to_sfd_folder      = path_to_sfd_folder,
                path_to_dataset_bank    = dataset_bank_path,
                save_timeseries         = save_timeseries,
                swapped_host            = True,
                preprocessed_df         = preprocessed_df,
            )

            ts_host = ts_host.dropna(subset=host_features)
            if ts_host.empty:
                raise ValueError(f"{ztf_id} has NaNs in host features – abort.")

            host_locus_feat_arr         = ts_host[host_features].iloc[-1].values
            host_ztf_id_in_dataset_bank = False

            # ⇢ make sure TNS variables exist
            host_tns_name, host_tns_cls, host_tns_z = "No TNS", "---", -99

            if {"raMean", "decMean"}.issubset(ts_host.columns):
                host_galaxy_ra  = ts_host["raMean"].iloc[0]
                host_galaxy_dec = ts_host["decMean"].iloc[0]
            elif {"host_ra", "host_dec"}.issubset(ts_host.columns):
                host_galaxy_ra  = ts_host["host_ra"].iloc[0]
                host_galaxy_dec = ts_host["host_dec"].iloc[0]
            else:
                host_galaxy_ra = host_galaxy_dec = np.nan

            continue                                   # skip dataframe-based path
        # ------------------------------------------------------------------

        try:
            locus_feat_arr = df_bank.loc[ztf_id][feature_names].values
        except KeyError:
            raise ValueError(f"ZTF ID '{ztf_id}' not found in dataset bank")

        ztf_id_in_dataset_bank = True

        df_bank_input_only = df_bank.loc[[ztf_id]]
        if host_loop:
            host_galaxy_ra = df_bank_input_only.iloc[0].host_ra
            host_galaxy_dec = df_bank_input_only.iloc[0].host_dec
        else:
            lc_galaxy_ra = df_bank_input_only.iloc[0].host_ra
            lc_galaxy_dec = df_bank_input_only.iloc[0].host_dec

        if save_timeseries:
            timeseries_df = get_timeseries_df(
                ztf_id=ztf_id,
                theorized_lightcurve_df=None,
                path_to_timeseries_folder=path_to_timeseries_folder,
                path_to_sfd_folder=path_to_sfd_folder,
                path_to_dataset_bank=dataset_bank_path,
                save_timeseries=save_timeseries,
                swapped_host=host_loop,
                preprocessed_df=preprocessed_df,
            )

            # Extract timeseries dataframe
            if ztf_id is not None:
                print(f"Attempting to fetch timeseries data for {ztf_id}...")

            timeseries_df = get_timeseries_df(
                ztf_id=ztf_id,
                theorized_lightcurve_df=(
                    theorized_lightcurve_df if not host_loop else None
                ),
                path_to_timeseries_folder=path_to_timeseries_folder,
                path_to_sfd_folder=path_to_sfd_folder,
                path_to_dataset_bank=dataset_bank_path,
                save_timeseries=save_timeseries,
                swapped_host=host_loop,
                preprocessed_df=preprocessed_df,
            )

            if host_loop:
                # More robust column handling for host galaxies
                if "raMean" in timeseries_df.columns and "decMean" in timeseries_df.columns:
                    host_galaxy_ra = timeseries_df["raMean"].iloc[0]
                    host_galaxy_dec = timeseries_df["decMean"].iloc[0]
                elif "host_ra" in timeseries_df.columns and "host_dec" in timeseries_df.columns:
                    host_galaxy_ra = timeseries_df["host_ra"].iloc[0]
                    host_galaxy_dec = timeseries_df["host_dec"].iloc[0]
                else:
                    print(f"Warning: Could not find RA/Dec columns for host {ztf_id}. Using NaN values.")
                    host_galaxy_ra = np.nan
                    host_galaxy_dec = np.nan
            else:
                if theorized_lightcurve_df is None:
                    # More robust column handling for source galaxies
                    if "raMean" in timeseries_df.columns and "decMean" in timeseries_df.columns:
                        lc_galaxy_ra = timeseries_df["raMean"].iloc[0]
                        lc_galaxy_dec = timeseries_df["decMean"].iloc[0]
                    elif "host_ra" in timeseries_df.columns and "host_dec" in timeseries_df.columns:
                        lc_galaxy_ra = timeseries_df["host_ra"].iloc[0]
                        lc_galaxy_dec = timeseries_df["host_dec"].iloc[0]
                    else:
                        print(f"Warning: Could not find RA/Dec columns for source {ztf_id}. Using NaN values.")
                        lc_galaxy_ra = np.nan
                        lc_galaxy_dec = np.nan

            # If timeseries_df is from theorized lightcurve, it only has lightcurve features
            if not host_loop and theorized_lightcurve_df is not None:
                all_feats = lc_features
            else:
                all_feats = lc_features + host_features

            timeseries_df = timeseries_df.dropna(subset=all_feats)
            if timeseries_df.empty:
                raise ValueError(f"{ztf_id} has some NaN features. Abort!")

            # Extract feature array from timeseries dataframe
            if not host_loop and theorized_lightcurve_df is not None:
                # theorized timeseries_df is just lightcurve data, so we must shape it properly
                for host_feature in host_features:
                    timeseries_df[host_feature] = np.nan

            locus_feat_arr_df = pd.DataFrame(timeseries_df[all_feats].iloc[-1]).T
            locus_feat_arr = locus_feat_arr_df[all_feats].iloc[0]

        # Pull TNS data for ztf_id
        if ztf_id is not None:
            tns_name, tns_cls, tns_z = get_TNS_data(ztf_id)
        else:
            tns_name, tns_cls, tns_z = "No TNS", "---", -99

        if host_loop:
            host_tns_name, host_tns_cls, host_tns_z = tns_name, tns_cls, tns_z
            host_ztf_id_in_dataset_bank = ztf_id_in_dataset_bank
            host_locus_feat_arr = locus_feat_arr
        else:
            lc_tns_name, lc_tns_cls, lc_tns_z = tns_name, tns_cls, tns_z
            lc_ztf_id_in_dataset_bank = ztf_id_in_dataset_bank
            lc_locus_feat_arr = locus_feat_arr

    # Make final feature array
    lc_feature_err_names = constants.lc_feature_err.copy()
    host_feature_err_names = constants.host_feature_err.copy()
    feature_err_names = lc_feature_err_names + host_feature_err_names

    if host_ztf_id is None:
        # Not swapping out host, use features from lightcurve ztf_id
        if lc_ztf_id_in_dataset_bank:
            # If the data came from the dataset bank, it's already correctly formatted
            locus_feat_arr = lc_locus_feat_arr
            # Also create locus_feat_series for the DataFrame creation later
            if isinstance(lc_locus_feat_arr, pd.Series):
                locus_feat_series = lc_locus_feat_arr
            else:
                # If it's already a numpy array, wrap it in a series with the right index
                locus_feat_series = pd.Series(lc_locus_feat_arr, index=feature_names)
        else:
            # Otherwise, we need to extract the right subset in the right order
            if isinstance(lc_locus_feat_arr, pd.Series):
                locus_feat_series = lc_locus_feat_arr[feature_names]
            else:
                # If it's already a numpy array, wrap it in a series with the right index
                locus_feat_series = pd.Series(lc_locus_feat_arr, index=feature_names)
    else:
        # Create new feature array with mixed lc and host features
        if lc_ztf_id_in_dataset_bank and host_ztf_id_in_dataset_bank:
            # If both sources are from the dataset bank, we need to combine them carefully
            # Convert to Series if they aren't already
            if not isinstance(lc_locus_feat_arr, pd.Series):
                lc_locus_feat_arr = pd.Series(lc_locus_feat_arr, index=feature_names)
            if not isinstance(host_locus_feat_arr, pd.Series):
                host_locus_feat_arr = pd.Series(host_locus_feat_arr, index=feature_names)

            subset_lc = lc_locus_feat_arr[lc_features]
            subset_host = host_locus_feat_arr[host_features]
            locus_feat_series = pd.concat([subset_lc, subset_host])

            # Make sure features are in the right order
            locus_feat_series = locus_feat_series.reindex(feature_names)
        else:
            # Handle the case where one or both aren't from the dataset bank
            lc_vals = np.asarray(lc_locus_feat_arr)[:len(lc_features)]  # keep only LC part
            subset_lc = pd.Series(lc_vals, index=lc_features)

            if isinstance(host_locus_feat_arr, pd.Series):
                subset_host = host_locus_feat_arr[host_features]
            else:
                # If it's already a numpy array, create a Series with the right index
                # First make sure array length matches index length
                if len(host_locus_feat_arr) != len(host_features):
                    # Extract only the values that correspond to host_features
                    # This handles the case when host_locus_feat_arr might contain values for all features
                    if len(host_locus_feat_arr) == len(feature_names):
                        # If the full array size matches all features, slice it to get only host features
                        start_idx = len(lc_features)
                        host_locus_feat_arr = host_locus_feat_arr[start_idx:start_idx+len(host_features)]
                    else:
                        # If we can't determine the correct slicing, log warning and use placeholder values
                        print(f"Warning: Host feature array length ({len(host_locus_feat_arr)}) doesn't match host_features length ({len(host_features)}). Using placeholder values.")
                        host_locus_feat_arr = np.full(len(host_features), np.nan)

                subset_host = pd.Series(host_locus_feat_arr, index=host_features)

            # If light curve features are empty, fill with NaN
            if subset_lc.empty:
                subset_lc = pd.Series(index=lc_features, data=np.nan)

            locus_feat_series = pd.concat([subset_lc, subset_host])

            # Make sure features are in the right order
            locus_feat_series = locus_feat_series.reindex(feature_names)

    # Ensure clean 1-row DataFrame in correct order
    locus_feat_df = pd.DataFrame([locus_feat_series])
    # Define err_lookup before using it
    err_lookup = constants.err_lookup.copy()
    # Add error columns to the DataFrame
    for feat_name, error_name in err_lookup.items():
        if feat_name in feature_names:
            # Only try to access error column if it exists in the series
            if error_name in locus_feat_series:
                locus_feat_df[error_name] = locus_feat_series[error_name]

    # Create Monte Carlo copies locus_feat_arrays_l
    np.random.seed(random_seed)

    locus_feat_arrs_mc_l = []
    for _ in range(num_sims):
        locus_feat_df_for_mc = locus_feat_df.copy()

        for feat_name, error_name in err_lookup.items():
            if feat_name in feature_names:
                # Skip if error column doesn't exist in the dataframe
                if error_name not in locus_feat_df_for_mc.columns:
                    continue

                std = locus_feat_df_for_mc[error_name]
                # Skip if std is NaN
                if std.isna().any():
                    continue

                noise = np.random.normal(0, std)
                if not np.isnan(noise):
                    locus_feat_df_for_mc[feat_name] = (
                        locus_feat_df_for_mc[feat_name] + noise
                    )
                else:
                    pass

        # Make sure we get a 1D array with the right feature order
        mc_array = locus_feat_df_for_mc[feature_names].values
        # Flatten if necessary - DataFrame.values can return a 2D array
        if mc_array.ndim > 1:
            mc_array = mc_array.flatten()
        locus_feat_arrs_mc_l.append(mc_array)

    locus_feat_df.drop_duplicates(inplace=True)

    # Make sure locus_feat_arr is a 1D array with the features in the right order
    if lc_ztf_id_in_dataset_bank and host_ztf_id is None:
        # When using data directly from the dataset bank without host swapping,
        # just use the array directly to avoid any transformations
        if isinstance(locus_feat_arr, np.ndarray):
            # It's already a numpy array in the right format
            pass
        else:
            # Make sure it's a 1D array with the right feature order
            locus_feat_arr = locus_feat_df[feature_names].values.flatten()
    else:
        # In all other cases, make sure we get a 1D array with the right feature order
        locus_feat_arr = locus_feat_df[feature_names].values.flatten()

    output_dict = {
        # host data is optional, it's only if the user decides to swap in a new host
        "host_ztf_id": host_ztf_id if host_ztf_id is not None else None,
        "host_tns_name": host_tns_name if host_ztf_id is not None else None,
        "host_tns_cls": host_tns_cls if host_ztf_id is not None else None,
        "host_tns_z": host_tns_z if host_ztf_id is not None else None,
        "host_ztf_id_in_dataset_bank": (
            host_ztf_id_in_dataset_bank if host_ztf_id is not None else None
        ),
        "host_galaxy_ra": host_galaxy_ra if host_ztf_id is not None else None,
        "host_galaxy_dec": host_galaxy_dec if host_ztf_id is not None else None,
        "lc_ztf_id": lc_ztf_id,
        "lc_tns_name": lc_tns_name,
        "lc_tns_cls": lc_tns_cls,
        "lc_tns_z": lc_tns_z,
        "lc_ztf_id_in_dataset_bank": lc_ztf_id_in_dataset_bank,
        "locus_feat_arr": locus_feat_arr,
        "locus_feat_arrs_mc_l": locus_feat_arrs_mc_l,
        "lc_galaxy_ra": lc_galaxy_ra,
        "lc_galaxy_dec": lc_galaxy_dec,
        "lc_feat_names": lc_features,
        "host_feat_names": host_features,
    }

    return output_dict
