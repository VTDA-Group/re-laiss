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
    path_to_sfd_data_folder,
    save_timeseries=False,
    host_ztf_id=None,
    lc_features=[],
    host_features=[],
    num_sims=10,
):
    """Assemble input feature vectors (and MC replicas) for a query object.

    Combines LC + host features—optionally swapping in a different host—and
    returns a dict used later by NN and AD stages.

    Parameters
    ----------
    lc_ztf_id : str | None
        ZTF ID of the transient to query.  Mutually exclusive with
        *theorized_lightcurve_df*.
    theorized_lightcurve_df : pandas.DataFrame | None
        Pre-computed ANTARES-style LC for a theoretical model.
    host_ztf_id : str | None
        If given, replace the query object’s host features with those of this
        transient.
    dataset_bank_path, path_to_timeseries_folder, path_to_sfd_data_folder : str | Path
        Locations for cached data.
    lc_features, host_features : list[str]
        Names of columns to extract.
    num_sims : int, default 10
        Number of Monte-Carlo perturbations for uncertainty propagation.

    Returns
    -------
    dict
        Primer dictionary containing feature arrays, metadata, and MC sims.

    Raises
    ------
    ValueError
        On inconsistent inputs or missing data.
    """
    feature_names = lc_features + host_features
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

        # Skip host loop if host galaxy to swap is not provided
        if host_loop and ztf_id is None:
            continue

        ztf_id_in_dataset_bank = False

        # Check if ztf_id is in dataset bank
        try:
            df_bank = pd.read_csv(dataset_bank_path, index_col=0)
            # Check to make sure all features are in the dataset bank
            missing_cols = [col for col in feature_names if col not in df_bank.columns]
            if missing_cols:
                raise KeyError(
                    f"KeyError: The following columns are not in the raw data provided: {missing_cols}. Abort!"
                )

            locus_feat_arr = df_bank.loc[ztf_id]

            print(f"{ztf_id} is in dataset_bank.")
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
                    path_to_sfd_data_folder=path_to_sfd_data_folder,
                    path_to_dataset_bank=dataset_bank_path,
                    save_timeseries=save_timeseries,
                    swapped_host=host_loop,
                )

        # If ztf_id is not in dataset bank...
        except:
            # Extract timeseries dataframe
            if ztf_id is not None:
                print(f"{ztf_id} is not in dataset_bank.")
            timeseries_df = get_timeseries_df(
                ztf_id=ztf_id,
                theorized_lightcurve_df=(
                    theorized_lightcurve_df if not host_loop else None
                ),
                path_to_timeseries_folder=path_to_timeseries_folder,
                path_to_sfd_data_folder=path_to_sfd_data_folder,
                path_to_dataset_bank=dataset_bank_path,
                save_timeseries=save_timeseries,
                swapped_host=host_loop,
            )

            if host_loop:
                host_galaxy_ra = timeseries_df["raMean"].iloc[0]
                host_galaxy_dec = timeseries_df["decMean"].iloc[0]
            else:
                if theorized_lightcurve_df is None:
                    lc_galaxy_ra = timeseries_df["raMean"].iloc[0]
                    lc_galaxy_dec = timeseries_df["decMean"].iloc[0]

            # If timeseries_df is from theorized lightcurve, it only has lightcurve features
            if not host_loop and theorized_lightcurve_df is not None:
                subset_feats_for_checking_na = lc_features
            else:
                subset_feats_for_checking_na = lc_features + host_features

            timeseries_df = timeseries_df.dropna(subset=subset_feats_for_checking_na)
            if timeseries_df.empty:
                raise ValueError(f"{ztf_id} has some NaN features. Abort!")

            # Extract feature array from timeseries dataframe
            if not host_loop and theorized_lightcurve_df is not None:
                # theorized timeseries_df is just lightcurve data, so we must shape it properly
                for host_feature in host_features:
                    timeseries_df[host_feature] = np.nan

            locus_feat_arr_df = pd.DataFrame(timeseries_df.iloc[-1]).T
            locus_feat_arr = locus_feat_arr_df.iloc[0]

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
        locus_feat_df = lc_locus_feat_arr[feature_names + feature_err_names]
    else:
        # Create new feature array with mixed lc and host features
        subset_lc_features = lc_locus_feat_arr[lc_features + lc_feature_err_names]
        subset_host_features = host_locus_feat_arr[
            host_features + host_feature_err_names
        ]

        locus_feat_df = pd.concat([subset_lc_features, subset_host_features], axis=0)

    # Create Monte Carlo copies locus_feat_arrays_l
    np.random.seed(888)
    err_lookup = constants.err_lookup.copy()
    locus_feat_arrs_mc_l = []
    for _ in range(num_sims):
        locus_feat_df_for_mc = locus_feat_df.copy()

        for feat_name, error_name in err_lookup.items():
            if feat_name in feature_names:
                std = locus_feat_df_for_mc[error_name]
                noise = np.random.normal(0, std)
                if not np.isnan(noise):
                    locus_feat_df_for_mc[feat_name] = (
                        locus_feat_df_for_mc[feat_name] + noise
                    )
                else:
                    pass

        locus_feat_arrs_mc_l.append(locus_feat_df_for_mc[feature_names].values)

    # Create true feature array
    locus_feat_arr = locus_feat_df[feature_names].values

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

