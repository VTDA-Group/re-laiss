import math
import os
import tempfile
import time
import warnings

import antares_client
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astro_prost.associate import associate_sample
from astropy.coordinates import SkyCoord
from dust_extinction.parameter_averages import G23
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import gamma, uniform
from sfdmap2 import sfdmap
from sklearn.cluster import DBSCAN
from sklearn.impute import KNNImputer, SimpleImputer

from . import constants
from .utils import suppress_output, compute_dataframe_hash, get_cache_key, load_cached_dataframe, cache_dataframe

warnings.filterwarnings("ignore", category=RuntimeWarning)

central_wv = {"g": 4849.11, "r": 6201.20, "i": 7534.96, "z": 8674.20}

def sanitize_features(df):
    return df.replace([np.inf, -np.inf, -999, 999], np.nan)

def build_dataset_bank(
    raw_df_bank,
    av_in_raw_df_bank=False,
    path_to_sfd_folder=None,
    theorized=False,
    path_to_dataset_bank=None,
    building_entire_df_bank=False,
    building_for_AD=False,
    preprocessed_df=None,
):
    """Clean, impute, dust-correct, and engineer features for reLAISS.

    Handles both archival and *theorized* light-curve inputs, performs KNN or
    mean imputation, builds colour indices, propagates uncertainties, and
    returns a ready-to-index DataFrame.

    Parameters
    ----------
    raw_df_bank : pandas.DataFrame
        Input light-curve + host-galaxy features (one or many rows).
    av_in_raw_df_bank : bool
        Whether A_V is already present in *raw_df_bank*.
    path_to_sfd_folder : str | Path | None, optional
        Directory with SFD dust maps (required if ``av_in_raw_df_bank=False``).
    theorized : bool, default False
        Set *True* when the input is a simulated/theoretical light curve that
        lacks host features.
    path_to_dataset_bank : str | Path | None, optional
        Existing bank used to fit the imputer when not building the entire set.
    building_entire_df_bank : bool, default False
        If *True*, fit the imputer on *raw_df_bank* itself.
    building_for_AD : bool, default False
        Use simpler mean imputation and suppress verbose prints for
        anomaly-detection pipelines.
    preprocessed_df : pandas.DataFrame | None, default None
        Pre-processed dataframe with imputed features. If provided, this is returned
        directly instead of processing raw_df_bank.

    Returns
    -------
    pandas.DataFrame
        Fully hydrated feature table indexed by ``ztf_object_id``.
    """
    # If preprocessed dataframe is provided, return it directly
    if preprocessed_df is not None:
        if not building_for_AD:
            print("Using provided preprocessed dataframe instead of processing raw data")
        return preprocessed_df

    # Generate cache key based on input parameters
    df_hash = compute_dataframe_hash(raw_df_bank)
    cache_key = get_cache_key(
        df_hash,
        av_in_raw_df_bank=av_in_raw_df_bank,
        path_to_sfd_folder=str(path_to_sfd_folder),
        theorized=theorized,
        path_to_dataset_bank=str(path_to_dataset_bank) if path_to_dataset_bank else None,
        building_entire_df_bank=building_entire_df_bank,
        building_for_AD=building_for_AD,
    )

    # Try to load from cache
    cached_df = load_cached_dataframe(cache_key)
    if cached_df is not None:
        if not building_for_AD:
            print("Loading preprocessed features from cache...")
        return cached_df

    if not building_for_AD:
        print("Processing features (this may take a while)...")

    raw_lc_features = constants.lc_features_const.copy()
    raw_host_features = constants.raw_host_features_const.copy()

    if av_in_raw_df_bank:
        if "A_V" not in raw_host_features:
            raw_host_features.append("A_V")
    else:
        for col in ["ra", "dec"]:
            if col not in raw_host_features:
                raw_host_features.insert(0, col)

    # if "ztf_object_id" is the index, move it to the first column
    if raw_df_bank.index.name == "ztf_object_id":
        raw_df_bank = raw_df_bank.reset_index()
    elif 'ZTFID' in raw_df_bank.columns.values:
        raw_df_bank['ztf_object_id'] = raw_df_bank['ZTFID']

    if theorized:
        raw_features = raw_lc_features
        raw_feats_no_ztf = raw_lc_features
    else:
        raw_features = ["ztf_object_id"] + raw_lc_features + raw_host_features
        raw_feats_no_ztf = raw_lc_features + raw_host_features

    # Check to make sure all required features are in the raw data
    missing_cols = [col for col in raw_features if col not in raw_df_bank.columns]
    if missing_cols:
        raise ValueError(
            f"KeyError: The following columns for this transient are not in the raw data: {missing_cols}. Abort!"
        )

    # Impute missing features
    test_dataset_bank = raw_df_bank.replace([np.inf, -np.inf, -999], np.nan).dropna(
        subset=raw_features
    )

    nan_cols = [
        col
        for col in raw_features
        if raw_df_bank[col].replace([np.inf, -np.inf, -999, 999], np.nan).isna().all()
    ]

    if not building_for_AD:
        print(
            f"There are {len(raw_df_bank) - len(test_dataset_bank)} of {len(raw_df_bank)} rows in the dataframe with 1 or more NA features."
        )
        if len(nan_cols) != 0:
            print(
                f"The following {len(nan_cols)} feature(s) are NaN for all measurements: {nan_cols}."
            )
        print("Imputing features...")

    wip_dataset_bank = raw_df_bank.copy()
    if "mjd_cutoff" in raw_df_bank.columns:
        mjd_col = raw_df_bank["mjd_cutoff"]
    else:
        mjd_col = None

    if building_entire_df_bank:
        X_input = sanitize_features(raw_df_bank[raw_feats_no_ztf])
        feat_imputer = KNNImputer(weights="distance").fit(X_input)
        # Only impute if there are actually NaN values
        if X_input.isna().any().any():
            imputed_filt_arr = feat_imputer.transform(X_input)
        else:
            imputed_filt_arr = X_input.values
    else:
        true_raw_df_bank = pd.read_csv(path_to_dataset_bank, low_memory=False)
        X_input = sanitize_features(true_raw_df_bank[raw_feats_no_ztf])

        if building_for_AD:
            feat_imputer = SimpleImputer(strategy="mean").fit(X_input)
        else:
            feat_imputer = KNNImputer(weights="distance").fit(X_input)

        X_transform = sanitize_features(wip_dataset_bank[raw_feats_no_ztf])
        
        # FIXED: Only impute NaN values, preserve valid values
        original_values = X_transform.copy()
        nan_mask = X_transform.isna()
        
        if nan_mask.any().any():
            # Only transform if there are NaN values to impute
            imputed_filt_arr = feat_imputer.transform(X_transform)
            # Restore original non-NaN values
            imputed_filt_arr = np.where(nan_mask.values, imputed_filt_arr, original_values.values)
        else:
            # No NaN values, use original data
            imputed_filt_arr = original_values.values

    imputed_filt_df = pd.DataFrame(imputed_filt_arr, columns=raw_feats_no_ztf)
    imputed_filt_df.index = raw_df_bank.index

    wip_dataset_bank[raw_feats_no_ztf] = imputed_filt_df

    wip_dataset_bank = wip_dataset_bank.replace([np.inf, -np.inf, -999], np.nan).dropna(
        subset=raw_features
    )

    if mjd_col is not None:
        wip_dataset_bank = wip_dataset_bank.assign(
            mjd_cutoff=mjd_col.reindex(wip_dataset_bank.index)
        )

    if not building_for_AD:
        if not wip_dataset_bank.empty:
            print("Successfully imputed features.")
        else:
            print("Failed to impute features.")

    # Engineer the remaining features
    if not theorized:
        if not building_for_AD:
            print("Engineering remaining features...")
        # Correct host magnitude features for dust
        m = sfdmap.SFDMap(path_to_sfd_folder)

        MW_RV = 3.1
        ext = G23(Rv=MW_RV)
        MW_EBV = m.ebv(wip_dataset_bank["ra"].to_numpy(), wip_dataset_bank["dec"].to_numpy())
        AV = MW_RV * MW_EBV

        for band in ["g", "r", "i", "z"]:
            mags = wip_dataset_bank[f"{band}KronMag"].to_numpy()
            A_filter = -2.5 * np.log10(
                ext.extinguish(central_wv[band]*u.AA, Av=AV)
            )
            wip_dataset_bank[f"{band}KronMagCorrected"] = mags - A_filter

        # Create color features
        wip_dataset_bank["gminusrKronMag"] = (
            wip_dataset_bank["gKronMag"] - wip_dataset_bank["rKronMag"]
        )
        wip_dataset_bank["rminusiKronMag"] = (
            wip_dataset_bank["rKronMag"] - wip_dataset_bank["iKronMag"]
        )
        wip_dataset_bank["iminuszKronMag"] = (
            wip_dataset_bank["iKronMag"] - wip_dataset_bank["zKronMag"]
        )

        # Calculate color uncertainties
        wip_dataset_bank["gminusrKronMagErr"] = np.sqrt(
            wip_dataset_bank["gKronMagErr"] ** 2 + wip_dataset_bank["rKronMagErr"] ** 2
        )
        wip_dataset_bank["rminusiKronMagErr"] = np.sqrt(
            wip_dataset_bank["rKronMagErr"] ** 2 + wip_dataset_bank["iKronMagErr"] ** 2
        )
        wip_dataset_bank["iminuszKronMagErr"] = np.sqrt(
            wip_dataset_bank["iKronMagErr"] ** 2 + wip_dataset_bank["zKronMagErr"] ** 2
        )

    final_df_bank = wip_dataset_bank

    # Cache the processed DataFrame
    if not building_for_AD:
        print("Caching preprocessed features...")

    cache_dataframe(final_df_bank, cache_key)

    return final_df_bank

def create_features_dict(
    lc_feature_names, host_feature_names, lc_groups=4, host_groups=4
):
    """Partition feature names into evenly-sized groups for weighting.

    Parameters
    ----------
    lc_feature_names : list[str]
        Names of light-curve features.
    host_feature_names : list[str]
        Names of host-galaxy features.
    lc_groups : int, default 4
        Number of LC groups in the output dict.
    host_groups : int, default 4
        Number of host groups in the output dict.

    Returns
    -------
    dict[str, list[str]]
        ``{'lc_group_1': [...], 'host_group_1': [...], ...}``
    """
    feature_dict = {}

    # Split light curve features into evenly sized chunks
    lc_chunk_size = math.ceil(len(lc_feature_names) / lc_groups)
    for i in range(lc_groups):
        start = i * lc_chunk_size
        end = start + lc_chunk_size
        chunk = lc_feature_names[start:end]
        if chunk:
            feature_dict[f"lc_group_{i+1}"] = chunk

    # Split host features into evenly sized chunks
    host_chunk_size = math.ceil(len(host_feature_names) / host_groups)
    for i in range(host_groups):
        start = i * host_chunk_size
        end = start + host_chunk_size
        chunk = host_feature_names[start:end]
        if chunk:
            feature_dict[f"host_group_{i+1}"] = chunk

    return feature_dict

def extract_lc_and_host_features(
    ztf_id,
    path_to_timeseries_folder,
    path_to_sfd_folder,
    path_to_dataset_bank=None,
    theorized_lightcurve_df=None,
    show_lc=False,
    show_host=True,
    store_csv=False,
    building_for_AD=False,
    swapped_host=False,
    preprocessed_df=None,
):
    """End-to-end extraction of light-curve **and** host-galaxy features.

    1. Pulls ZTF photometry from ANTARES (or uses a supplied theoretical LC).
    2. Computes time-series features with *lightcurve_engineer*.
    3. Associates the most probable PS1 host with PROST and appends raw host
       features.
    4. Dust-corrects, builds colours, imputes gaps, and writes an optional CSV.

    Parameters
    ----------
    ztf_id : str
        ZTF object identifier (ignored when *theorized_lightcurve_df* is given).
    path_to_timeseries_folder : str | Path
        Folder to cache per-object time-series CSVs.
    path_to_sfd_folder : str | Path
        Location of SFD dust maps.
    theorized_lightcurve_df : pandas.DataFrame | None, optional
        Pre-simulated LC in ANTARES column format (``ant_passband``, ``ant_mjd``,
        ``ant_mag``, ``ant_magerr``).
    show_lc : bool, default False
        Plot the g/r light curves.
    show_host : bool, default True
        Print PS1 cut-out URL on successful host association.
    store_csv : bool, default False
        Write a timeseries CSV next to *path_to_timeseries_folder*.
    building_for_AD : bool, default False
        Quieter prints + mean imputation only.
    swapped_host : bool, default False
        Indicator used when re-running with an alternate host galaxy.
    preprocessed_df : pandas.DataFrame | None, default None
        Pre-processed dataframe with imputed features. If provided, this is used
        instead of loading and processing the raw dataset bank.

    Returns
    -------
    pandas.DataFrame
        Hydrated feature rows for every increasing-epoch subset of the LC.
    """
    start_time = time.time()
    df_path = path_to_timeseries_folder

    # Look up transient
    if theorized_lightcurve_df is not None:
        df_ref = theorized_lightcurve_df
        # Ensure correct capitalization of passbands ('g' and 'R')
        df_ref["ant_passband"] = df_ref["ant_passband"].replace({"G": "g", "r": "R"})
    else:
        try:
            ref_info = antares_client.search.get_by_ztf_object_id(ztf_id)
            df_ref = ref_info.timeseries.to_pandas()
        except:
            print("antares_client can't find this object. Abort!")
            raise ValueError(f"antares_client can't find object {ztf_id}.")

    # Check for observations
    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]
    try:
        mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
        mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag
    except:
        raise ValueError(
            f"No observations for {ztf_id if theorized_lightcurve_df is None else 'theorized lightcurve'}. Abort!\n"
        )

    # Plot lightcurve
    if show_lc:
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.gca().invert_yaxis()

        ax.errorbar(
            x=df_ref_r.ant_mjd,
            y=df_ref_r.ant_mag,
            yerr=df_ref_r.ant_magerr,
            fmt="o",
            c="r",
            label=f"REF: {ztf_id}",
        )
        ax.errorbar(
            x=df_ref_g.ant_mjd,
            y=df_ref_g.ant_mag,
            yerr=df_ref_g.ant_magerr,
            fmt="o",
            c="g",
        )
        plt.show()

    # Pull required lightcurve features:
    if theorized_lightcurve_df is None:
        lightcurve = df_ref[["ant_passband", "ant_mjd", "ant_mag", "ant_magerr"]]
    else:
        lightcurve = theorized_lightcurve_df

    lightcurve = lightcurve.sort_values(by="ant_mjd").reset_index(drop=True).dropna()
    min_obs_count = 5
    if len(lightcurve) < min_obs_count:
        raise ValueError(
            f"Not enough observations for {ztf_id if theorized_lightcurve_df is None else 'theorized lightcurve'}. Abort!\n"
        )

    # Engineer features in time
    lc_col_names = constants.lc_features_const.copy()
    lc_timeseries_feat_df = pd.DataFrame(
        columns=["ztf_object_id"] + ["obs_num"] + ["mjd_cutoff"] + lc_col_names
    )

    # Keep track of cumulative peak magnitudes, times, and amplitudes
    g_peak_mag_cumulative = None
    r_peak_mag_cumulative = None
    g_peak_time_cumulative = None
    r_peak_time_cumulative = None
    g_amplitude_cumulative = 0.0
    r_amplitude_cumulative = 0.0
    g_faint_mag_cumulative = None
    r_faint_mag_cumulative = None

    for i in range(min_obs_count, len(lightcurve) + 1):
        lightcurve_subset = lightcurve.iloc[:i]
        time_mjd = lightcurve_subset["ant_mjd"].iloc[-1]

        # Engineer lightcurve features
        df_g = lightcurve_subset[lightcurve_subset["ant_passband"] == "g"]
        time_g = df_g["ant_mjd"].tolist()
        mag_g = df_g["ant_mag"].tolist()
        err_g = df_g["ant_magerr"].tolist()

        df_r = lightcurve_subset[lightcurve_subset["ant_passband"] == "R"]
        time_r = df_r["ant_mjd"].tolist()
        mag_r = df_r["ant_mag"].tolist()
        err_r = df_r["ant_magerr"].tolist()

        try:
            extractor = SupernovaFeatureExtractor(
                time_g=time_g,
                mag_g=mag_g,
                err_g=err_g,
                time_r=time_r,
                mag_r=mag_r,
                err_r=err_r,
                ztf_object_id=ztf_id,
            )

            engineered_lc_properties_df = extractor.extract_features(
                return_uncertainty=True
            )

            # Update cumulative peak magnitudes and amplitudes
            if engineered_lc_properties_df is not None:
                current_g_peak = engineered_lc_properties_df['g_peak_mag'].iloc[0]
                current_r_peak = engineered_lc_properties_df['r_peak_mag'].iloc[0]

                # Update cumulative peak magnitudes (brighter = lower mag number)
                if not pd.isna(current_g_peak):
                    if g_peak_mag_cumulative is None or current_g_peak < g_peak_mag_cumulative:
                        g_peak_mag_cumulative = current_g_peak
                        g_peak_time_cumulative = engineered_lc_properties_df['g_peak_time'].iloc[0]

                if not pd.isna(current_r_peak):
                    if r_peak_mag_cumulative is None or current_r_peak < r_peak_mag_cumulative:
                        r_peak_mag_cumulative = current_r_peak
                        r_peak_time_cumulative = engineered_lc_properties_df['r_peak_time'].iloc[0]

                # Update cumulative faint magnitudes (fainter = higher mag number)
                if len(df_g) > 0:
                    current_g_faint = df_g["ant_mag"].max()
                    if g_faint_mag_cumulative is None or current_g_faint > g_faint_mag_cumulative:
                        g_faint_mag_cumulative = current_g_faint

                if len(df_r) > 0:
                    current_r_faint = df_r["ant_mag"].max()
                    if r_faint_mag_cumulative is None or current_r_faint > r_faint_mag_cumulative:
                        r_faint_mag_cumulative = current_r_faint

                # Calculate cumulative amplitudes
                if g_peak_mag_cumulative is not None and g_faint_mag_cumulative is not None:
                    g_amplitude_cumulative = g_faint_mag_cumulative - g_peak_mag_cumulative

                if r_peak_mag_cumulative is not None and r_faint_mag_cumulative is not None:
                    r_amplitude_cumulative = r_faint_mag_cumulative - r_peak_mag_cumulative

                # Override with cumulative values
                if g_peak_mag_cumulative is not None:
                    engineered_lc_properties_df.loc[0, 'g_peak_mag'] = g_peak_mag_cumulative
                    engineered_lc_properties_df.loc[0, 'g_peak_time'] = g_peak_time_cumulative
                    engineered_lc_properties_df.loc[0, 'g_amplitude'] = g_amplitude_cumulative

                if r_peak_mag_cumulative is not None:
                    engineered_lc_properties_df.loc[0, 'r_peak_mag'] = r_peak_mag_cumulative
                    engineered_lc_properties_df.loc[0, 'r_peak_time'] = r_peak_time_cumulative
                    engineered_lc_properties_df.loc[0, 'r_amplitude'] = r_amplitude_cumulative
                
                # CRITICAL FIX: Override duration calculations with cumulative timespan
                total_timespan = time_mjd - lightcurve_subset["ant_mjd"].min()
                
                # For long-lived transients, duration should reflect the total observed timespan
                # where we have significant flux, not just the half-flux calculation from current subset
                if total_timespan > 0:
                    # Use total timespan as a proxy for duration above half flux for long transients
                    # This prevents the duration from decreasing as more data comes in
                    current_g_duration = engineered_lc_properties_df.loc[0, 'g_duration_above_half_flux']
                    current_r_duration = engineered_lc_properties_df.loc[0, 'r_duration_above_half_flux']
                    
                    # Only update if the new calculated duration is longer OR if current is NaN
                    if pd.isna(current_g_duration) or total_timespan > current_g_duration:
                        engineered_lc_properties_df.loc[0, 'g_duration_above_half_flux'] = total_timespan
                    if pd.isna(current_r_duration) or total_timespan > current_r_duration:
                        engineered_lc_properties_df.loc[0, 'r_duration_above_half_flux'] = total_timespan

        except:
            continue

        if engineered_lc_properties_df is not None and not engineered_lc_properties_df.isna().all(axis=None):
            engineered_lc_properties_df.insert(0, "mjd_cutoff", time_mjd)
            engineered_lc_properties_df.insert(0, "obs_num", int(i))
            engineered_lc_properties_df.insert(
                0,
                "ztf_object_id",
                ztf_id if theorized_lightcurve_df is None else "theorized_lightcurve",
            )

            if lc_timeseries_feat_df.empty:
                lc_timeseries_feat_df = engineered_lc_properties_df
            else:
                # Only concat if not all-NA
                if not engineered_lc_properties_df.isna().all(axis=None):
                    lc_timeseries_feat_df = pd.concat(
                        [lc_timeseries_feat_df, engineered_lc_properties_df],
                        ignore_index=True,
                    )

    end_time = time.time()

    if lc_timeseries_feat_df.empty and not swapped_host:
        raise ValueError(
            f"Failed to extract features for {ztf_id if theorized_lightcurve_df is None else 'theorized lightcurve'}"
        )

    print(
        f"Extracted lightcurve features for {ztf_id if theorized_lightcurve_df is None else 'theorized lightcurve'} in {(end_time - start_time):.2f}s!"
    )

    # Get PROST features
    if theorized_lightcurve_df is None:
        print("Searching for host galaxy...")
        ra, dec = np.mean(df_ref.ant_ra), np.mean(df_ref.ant_dec)
        snName = [ztf_id, ztf_id]
        snCoord = [
            SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
            SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            # define priors for properties
            priorfunc_offset = uniform(loc=0, scale=5)

            likefunc_offset = gamma(a=0.75)

            priors = {"offset": priorfunc_offset}
            likes = {"offset": likefunc_offset}

            transient_catalog = pd.DataFrame(
                {"IAUID": [snName], "RA": [ra], "Dec": [dec]}
            )

            catalogs = ["panstarrs"]
            transient_coord_cols = ("RA", "Dec")
            transient_name_col = "IAUID"
            verbose = 0
            parallel = False
            save = False
            progress_bar = False
            cat_cols = True
            with suppress_output():
                hosts = associate_sample(
                    transient_catalog,
                    coord_cols=transient_coord_cols,
                    priors=priors,
                    likes=likes,
                    catalogs=catalogs,
                    parallel=parallel,
                    save=save,
                    progress_bar=progress_bar,
                    cat_cols=cat_cols,
                    calc_host_props=False,
                )
            hosts.rename(
                columns={"host_ra": "raMean", "host_dec": "decMean"}, inplace=True
            )

            if len(hosts) >= 1:
                # CRITICAL FIX: Take only the FIRST (best) host candidate
                best_host = hosts.iloc[0:1]  # Select first row as DataFrame
                hosts_df = pd.DataFrame(best_host).reset_index(drop=True)
            else:
                print(f"Cannot identify host galaxy for {ztf_id}. Abort!\n")
                return

            # Check if required host features are missing
            try:
                raw_host_feature_check = constants.raw_host_features_const.copy()
                hosts_df = hosts_df[raw_host_feature_check]  # Use hosts_df, not hosts
            except KeyError:
                print(
                    f"KeyError: The following columns are not in the identified host feature set. Try engineering: {[col for col in raw_host_feature_check if col not in hosts_df.columns]}.\nAbort!"
                )
                return
            hosts_df = hosts_df[~hosts_df.isnull().any(axis=1)]
            if len(hosts_df) < 1:
                # if any features are nan, we can't use as input
                print(f"Some features are NaN for {ztf_id}. Abort!\n")
                return

            if show_host:
                if not building_for_AD:
                    print(
                        f"Host galaxy identified for {ztf_id}: http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={hosts.raMean.values[0]}+{hosts.decMean.values[0]}&filter=color"
                    )
                else:
                    print("Host identified.")

        if not lc_timeseries_feat_df.empty:
            hosts_df = pd.concat(
                [hosts_df] * len(lc_timeseries_feat_df), ignore_index=True
            )
            lc_and_hosts_df = pd.concat([lc_timeseries_feat_df, hosts_df], axis=1)
        else:
            if swapped_host:
                # For swapped host, create a single row with the host features
                lc_timeseries_feat_df = pd.DataFrame(columns=lc_timeseries_feat_df.columns)
                lc_timeseries_feat_df.loc[0] = np.nan
                lc_timeseries_feat_df.loc[0, "ztf_object_id"] = ztf_id
                lc_and_hosts_df = pd.concat([lc_timeseries_feat_df, hosts_df], axis=1)
            else:
                lc_timeseries_feat_df.loc[0, "ztf_object_id"] = (
                    ztf_id if theorized_lightcurve_df is None else "theorized_lightcurve"
                )
                lc_and_hosts_df = pd.concat([lc_timeseries_feat_df, hosts_df], axis=1)

        lc_and_hosts_df = lc_and_hosts_df.set_index("ztf_object_id")

        lc_and_hosts_df["raMean"] = hosts.raMean.values[0]
        lc_and_hosts_df["decMean"] = hosts.decMean.values[0]

        if not os.path.exists(df_path):
            print(f"Creating path {df_path}.")
            os.makedirs(df_path)

        lc_and_hosts_df["ra"] = ra
        lc_and_hosts_df["dec"] = dec

    # Engineer additonal features in build_dataset_bank function
    if building_for_AD:
        print("Engineering features...")

    lc_and_hosts_df_hydrated = build_dataset_bank(
        raw_df_bank=(
            lc_and_hosts_df
            if theorized_lightcurve_df is None
            else lc_timeseries_feat_df
        ),
        av_in_raw_df_bank=False,
        path_to_sfd_folder=path_to_sfd_folder,
        theorized=True if theorized_lightcurve_df is not None else False,
        path_to_dataset_bank=path_to_dataset_bank,
        building_for_AD=building_for_AD,
        preprocessed_df=preprocessed_df,
    )

    if store_csv and not lc_and_hosts_df_hydrated.empty:
        os.makedirs(df_path, exist_ok=True)
        if theorized_lightcurve_df is None:
            lc_and_hosts_df_hydrated.to_csv(f"{df_path}/{ztf_id}_timeseries.csv")
            print(f"Saved timeseries features for {ztf_id}!\n")
        else:
            lc_and_hosts_df_hydrated.to_csv(f"{df_path}/theorized_timeseries.csv")
            print("Saved timeseries features for theorized lightcurve!\n")

    return lc_and_hosts_df_hydrated

def getExtinctionCorrectedMag(
    transient_row,
    band,
    av_in_raw_df_bank,
    path_to_sfd_folder='./',
    m=None
):
    """Milky-Way extinction-corrected Kron magnitude for one passband.

    Parameters
    ----------
    transient_row : pandas.Series
        Row from the raw host-feature DataFrame.
    band : {'g', 'r', 'i', 'z'}
        Photometric filter to correct.
    av_in_raw_df_bank : bool
        If *True* use ``transient_row["A_V"]`` directly; otherwise compute
        E(B−V) from the SFD dust map in *path_to_sfd_folder*.
    path_to_sfd_folder : str | pathlib.Path | None, optional
        Folder containing *SFDMap* dust files when A_V is not pre-computed.

    Returns
    -------
    float
        Extinction-corrected Kron magnitude.
    """
    central_wv = {"g": 4849.11, "r": 6201.20, "i": 7534.96, "z": 8674.20}
    MW_RV = 3.1
    ext = G23(Rv=MW_RV)

    if av_in_raw_df_bank:
        MW_AV = transient_row["A_V"]
    elif m is not None:
        MW_EBV = m.ebv(float(transient_row["ra"]), float(transient_row["dec"]))
        MW_AV = MW_RV * MW_EBV

    wv_filter = central_wv[band]
    A_filter = -2.5 * np.log10(ext.extinguish(wv_filter * u.AA, Av=MW_AV))

    return transient_row[band + "KronMag"] - A_filter

def local_curvature(times, mags):
    """Median second derivative (curvature) of a light-curve segment.

    Parameters
    ----------
    times : array-like
        Strictly increasing observation times (days).
    mags : array-like
        Corresponding magnitudes.

    Returns
    -------
    float
        Median curvature in mag day⁻²; ``np.nan`` if fewer than three points.
    """
    if len(times) < 3:
        return np.nan
    curvatures = []
    for i in range(1, len(times) - 1):
        t0, t1, t2 = times[i - 1], times[i], times[i + 1]
        m0, m1, m2 = mags[i - 1], mags[i], mags[i + 1]
        dt = t2 - t0
        if dt == 0:
            continue
        a = (m2 - 2 * m1 + m0) / ((dt / 2) ** 2)
        curvatures.append(a)
    return np.median(curvatures) if curvatures else np.nan


m = sfdmap.SFDMap()





class SupernovaFeatureExtractor:
    @staticmethod
    def describe_features():
        """Dictionary mapping feature names → human-readable descriptions.

        Returns
        -------
        dict[str, str]
            Keys follow the column names produced by
            :pymeth:`SupernovaFeatureExtractor.extract_features`.
        """
        return {
            # Core timing and brightness features
            "t0": "Time zero-point for light curve normalization",
            "g_peak_mag": "Minimum magnitude (brightest point) in g band",
            "g_peak_time": "Time of peak brightness in g band",
            "g_rise_time": "Time from 50% peak flux to g-band peak",
            "g_decline_time": "Time from g-band peak to 50% flux decay",
            "g_duration_above_half_flux": "Duration above 50% of g-band peak flux",
            "r_peak_mag": "Minimum magnitude (brightest point) in r band",
            "r_peak_time": "Time of peak brightness in r band",
            "r_rise_time": "Time from 50% peak flux to r-band peak",
            "r_decline_time": "Time from r-band peak to 50% flux decay",
            "r_duration_above_half_flux": "Duration above 50% of r-band peak flux",
            
            # Amplitude and variability features
            "g_amplitude": "Magnitude difference between min and max in g band",
            "r_amplitude": "Magnitude difference between min and max in r band",
            "g_skewness": "Skewness of magnitude distribution in g band",
            "r_skewness": "Skewness of magnitude distribution in r band",
            "g_beyond_2sigma": "Fraction of g-band points beyond 2σ from mean",
            "r_beyond_2sigma": "Fraction of r-band points beyond 2σ from mean",
            
            # Color features
            "mean_g-r": "Average g-r color over shared time range",
            "g-r_at_g_peak": "g-r color at g-band peak time",
            "mean_color_rate": "Average rate of change of g-r color",
            
            # Peak structure features
            "g_n_peaks": "Number of peaks in g band (prominence > 0.1)",
            "r_n_peaks": "Number of peaks in r band (prominence > 0.1)",
            "g_dt_main_to_secondary_peak": "Time between main and secondary peaks in g band",
            "r_dt_main_to_secondary_peak": "Time between main and secondary peaks in r band",
            "g_dmag_secondary_peak": "Magnitude difference between main and secondary peaks in g band",
            "r_dmag_secondary_peak": "Magnitude difference between main and secondary peaks in r band",
            "g_secondary_peak_prominence": "Prominence of secondary peak in g band",
            "r_secondary_peak_prominence": "Prominence of secondary peak in r band",
            "g_secondary_peak_width": "Width of secondary peak in g band",
            "r_secondary_peak_width": "Width of secondary peak in r band",
            
            # Rolling variance features
            "g_max_rolling_variance": "Maximum rolling variance in g band",
            "r_max_rolling_variance": "Maximum rolling variance in r band",
            "g_mean_rolling_variance": "Mean rolling variance in g band",
            "r_mean_rolling_variance": "Mean rolling variance in r band",
            
            # Local curvature features
            "g_rise_local_curvature": "Local curvature during g-band rise",
            "g_decline_local_curvature": "Local curvature during g-band decline",
            "r_rise_local_curvature": "Local curvature during r-band rise",
            "r_decline_local_curvature": "Local curvature during r-band decline",
            
            # Essential extreme transient detection feature
            "total_duration": "Total observational duration (end-to-end timespan in days)",
            
            # Validation features
            "features_valid": "Whether all key features were successfully computed",
            "ztf_object_id": "ZTF object identifier",
        }



    def __init__(
        self, time_g, mag_g, err_g, time_r, mag_r, err_r, ztf_object_id=None, ra=None, dec=None
    ):
        """Create a feature extractor for g/r light curves.

        Times are zero-pointed to the earliest observation; optional Milky-Way
        extinction is applied when *ra/dec* are supplied.

        Parameters
        ----------
        time_g, mag_g, err_g : array-like
            g-band MJD, magnitude and 1-σ uncertainty.
        time_r, mag_r, err_r : array-like
            r-band MJD, magnitude and 1-σ uncertainty.
        ztf_object_id : str | None, optional
            Identifier used in warnings and output tables.
        ra, dec : float | None, optional
            ICRS coordinates (deg) for dust-extinction correction.

        Raises
        ------
        ValueError
            If input arrays are empty or have different lengths.
        """
        # Input validation
        if len(time_g) == 0 and len(time_r) == 0:
            raise ValueError("Both g and r bands are empty")

        # Check that each band's arrays have the same length
        if len(time_g) != len(mag_g) or len(time_g) != len(err_g):
            raise ValueError(f"G band arrays have different lengths: time={len(time_g)}, mag={len(mag_g)}, err={len(err_g)}")

        if len(time_r) != len(mag_r) or len(time_r) != len(err_r):
            raise ValueError(f"R band arrays have different lengths: time={len(time_r)}, mag={len(mag_r)}, err={len(err_r)}")

        if ztf_object_id:
            self.ztf_object_id = ztf_object_id
        else:
            self.ztf_object_id = "Theorized Lightcurve"
        self.g = {
            "time": np.array(time_g),
            "mag": np.array(mag_g),
            "err": np.array(err_g),
        }
        self.r = {
            "time": np.array(time_r),
            "mag": np.array(mag_r),
            "err": np.array(err_r),
        }

        # Handle t0 calculation for empty bands
        if len(self.g["time"]) > 0 and len(self.r["time"]) > 0:
            t0 = min(self.g["time"].min(), self.r["time"].min())
        elif len(self.g["time"]) > 0:
            t0 = self.g["time"].min()
        elif len(self.r["time"]) > 0:
            t0 = self.r["time"].min()
        else:
            t0 = 0  # Both empty (should have been caught earlier)

        self.time_offset = t0

        # Apply time offset
        if len(self.g["time"]) > 0:
            self.g["time"] -= t0
        if len(self.r["time"]) > 0:
            self.r["time"] -= t0

        # Apply extinction correction if coordinates provided
        if ra is not None and dec is not None:
            ebv = m.ebv(ra, dec)
            ext = G23(Rv=3.1)
            lambda_g = 0.477 * u.um
            lambda_r = 0.623 * u.um
            Ag = ext.extinguish(lambda_g, Ebv=ebv)
            Ar = ext.extinguish(lambda_r, Ebv=ebv)
            self.g["mag"] -= -2.5 * np.log10(Ag)
            self.r["mag"] -= -2.5 * np.log10(Ar)
        self._preprocess()

    def _preprocess(self, min_cluster_size=2):
        """Sort, de-duplicate, and DBSCAN-filter out isolated epochs.

        Removes cluster labels with fewer than *min_cluster_size* points and
        re-normalises times so that ``t=0`` corresponds to the earliest good
        observation in either band.
        """
        for band_name in ["g", "r"]:
            band = getattr(self, band_name)
            idx = np.argsort(band["time"])
            for key in band:
                band[key] = band[key][idx]

            # Only apply DBSCAN filtering if we have many points and large time gaps
            if len(band["time"]) > 50:
                time_reshaped = band["time"].reshape(-1, 1)
                from sklearn.cluster import DBSCAN
                
                # Adaptive eps based on median time spacing
                time_diffs = np.diff(np.sort(band["time"]))
                median_spacing = np.median(time_diffs)
                # Use 3x median spacing as eps to be more permissive for long transients
                eps = max(5, 3 * median_spacing)

                clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(time_reshaped)
                labels = clustering.labels_

                # Keep all points that are part of clusters with ≥ min_cluster_size
                good_clusters = [
                    label
                    for label in set(labels)
                    if label != -1 and np.sum(labels == label) >= min_cluster_size
                ]
                
                # If no good clusters found, keep all points (don't filter)
                if len(good_clusters) > 0:
                    mask = np.isin(labels, good_clusters)
                    for key in band:
                        band[key] = band[key][mask]
            # For smaller datasets, skip DBSCAN filtering entirely

        # Recalculate t0 based on filtered times
        if len(self.g["time"]) == 0 or len(self.r["time"]) == 0:
            pass
        else:
            new_time_offset = min(self.g["time"].min(), self.r["time"].min())

            # Normalize times again
            self.g["time"] -= new_time_offset
            self.r["time"] -= new_time_offset

            self.time_offset += new_time_offset

    def _select_main_cluster(self, time, mag, min_samples=3, eps=20):
        """Return a boolean mask selecting the dominant DBSCAN time cluster.

        The cluster with the brightest peak and tightest span wins the tie-break.
        """
        if len(time) < min_samples:
            return np.ones_like(time, dtype=bool)
        time_reshaped = np.array(time).reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(time_reshaped)
        labels = clustering.labels_
        if all(labels == -1):
            return np.ones_like(time, dtype=bool)
        unique_labels = np.unique(labels[labels != -1])
        best_label = None
        best_brightness = np.inf
        for label in unique_labels:
            mask = labels == label
            cluster_time = np.array(time)[mask]
            cluster_span = cluster_time.max() - cluster_time.min()
            cluster_mag = np.min(np.array(mag)[mask])
            # Prioritize clusters with brighter peak and tighter time span
            score = cluster_mag + 0.05 * cluster_span
            if score < best_brightness:
                best_brightness = score
                best_label = label
        return labels == best_label

    def _flag_isolated_points(time, max_gap_factor=5):
        """Identify photometric points that are isolated by large temporal gaps.

        Returns
        -------
        numpy.ndarray[bool]
            True for epochs flanked by gaps > *max_gap_factor* × median cadence.
        """
        time = np.sort(time)
        dt = np.diff(time)

        # Median cadence (ignoring gaps)
        median_dt = np.median(dt)

        # Find large gaps
        gaps = np.concatenate([[0], dt > max_gap_factor * median_dt])

        # Mark isolated points as True
        isolated = np.zeros_like(time, dtype=bool)
        for i in range(1, len(time) - 1):
            if gaps[i] and gaps[i + 1]:
                isolated[i] = True
        return isolated

    def _core_stats(self, band):
        """Peak, rise/decline and half-flux duration for one band.

        Parameters
        ----------
        band : dict
            ``{'time','mag'}`` arrays for a single filter.

        Returns
        -------
        tuple
            *(peak_mag, peak_time, rise_time, decline_time, duration_above_half)*

        Notes
        -----
        All values are ``np.nan`` if <3 points or total peak-to-peak amplitude <0.2 mag.
        """
        t, m = band["time"], band["mag"]
        mask = np.isfinite(t) & np.isfinite(m) & ~np.isnan(m)
        t, m = t[mask], m[mask]
        t, idx = np.unique(t, return_index=True)
        m = m[idx]
        if len(m) < 3 or np.ptp(m) < 0.2:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        peak_idx = np.argmin(m)
        peak_mag = m[peak_idx]
        peak_time = t[peak_idx]
        last_time = np.nanmax(t)
        flux = 10 ** (-0.4 * m)
        half_flux = 0.5 * 10 ** (-0.4 * peak_mag)
        half_mag = -2.5 * np.log10(half_flux)
        pre, post = t < peak_time, t > peak_time
        try:
            rise_time = peak_time - np.interp(half_mag, m[pre][::-1], t[pre][::-1])
        except:
            rise_time = np.nan
        try:
            decline_time = np.interp(half_mag, m[post], t[post]) - peak_time
        except:
            decline_time = np.nan
        above_half = t[m < half_mag]
        duration = above_half[-1] - above_half[0] if len(above_half) > 1 else np.nan
        return peak_mag, peak_time, rise_time, decline_time, duration, last_time

    def _variability_stats(self, band):
        """Amplitude, skewness, and 2-σ outlier rate of a magnitude series.

        Returns
        -------
        tuple
            *(amplitude, skewness, fraction_beyond_2σ)*
        """
        mag = band["mag"]
        amp = np.max(mag) - np.min(mag)
        std = np.std(mag)
        mean = np.mean(mag)
        skew = (np.mean((mag - mean) ** 3) / std**3) if std > 0 else np.nan
        beyond_2 = np.sum(np.abs(mag - mean) > 2 * std) / len(mag)
        return amp, skew, beyond_2

    def _color_features(self):
        """Compute mean g–r colour, g–r at g-band peak, and average colour slope.

        Returns
        -------
        tuple
            ``(mean_colour, colour_at_g_peak, mean_dcolour_dt)``
            or ``None`` when bands lack overlap.
        """
        if len(self.g["time"]) < 2 or len(self.r["time"]) < 2:
            return None

        def dedup(t, m):
            mask = np.isfinite(t) & np.isfinite(m) & ~np.isnan(m)
            t, m = t[mask], m[mask]
            _, idx = np.unique(t, return_index=True)
            return t[idx], m[idx]

        g_time, g_mag = dedup(self.g["time"], self.g["mag"])
        r_time, r_mag = dedup(self.r["time"], self.r["mag"])

        t_min = max(g_time.min(), r_time.min())
        t_max = min(g_time.max(), r_time.max())

        if t_max <= t_min or np.isnan(t_min) or np.isnan(t_max):
            return None
        
        # MUCH SIMPLER APPROACH: Use actual observation times for color calculation
        # Find times where we have measurements in both bands (within reasonable tolerance)
        color_measurements = []
        color_times = []
        
        # For each g observation, find the closest r observation (within 1 day)
        for gt, gm in zip(g_time, g_mag):
            if gt < t_min or gt > t_max:
                continue
            
            # Find closest r-band observation
            time_diffs = np.abs(r_time - gt)
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= 1.0:  # Within 1 day
                color_measurements.append(gm - r_mag[closest_idx])
                color_times.append(gt)
        
        # Also try the reverse: for each r observation, find closest g
        for rt, rm in zip(r_time, r_mag):
            if rt < t_min or rt > t_max:
                continue
            
            time_diffs = np.abs(g_time - rt)
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= 1.0:  # Within 1 day
                # Avoid duplicates by checking if we already have a measurement close to this time
                if not any(abs(ct - rt) < 0.1 for ct in color_times):
                    color_measurements.append(g_mag[closest_idx] - rm)
                    color_times.append(rt)
        
        if len(color_measurements) < 3:
            # Fall back to simple interpolation if not enough direct measurements
            t_grid = np.linspace(t_min, t_max, max(10, int(t_max - t_min)))
            g_interp = interp1d(g_time, g_mag, kind="linear", bounds_error=False, fill_value=np.nan)
            r_interp = interp1d(r_time, r_mag, kind="linear", bounds_error=False, fill_value=np.nan)
            color = g_interp(t_grid) - r_interp(t_grid)
            valid_mask = ~np.isnan(color)
            
            if valid_mask.sum() < 3:
                mean_color = np.nan
                mean_rate = 0.0
            else:
                mean_color = np.mean(color[valid_mask])
                # Use a very simple linear fit for rate
                valid_times = t_grid[valid_mask]
                valid_colors = color[valid_mask]
                if len(valid_colors) >= 2:
                    # Simple linear regression
                    coeffs = np.polyfit(valid_times, valid_colors, 1)
                    mean_rate = coeffs[0]  # Slope
                    # Sanity check
                    if abs(mean_rate) > 0.1:  # > 0.1 mag/day is very fast
                        mean_rate = 0.0
                else:
                    mean_rate = 0.0
        else:
            # Use direct measurements
            color_measurements = np.array(color_measurements)
            color_times = np.array(color_times)
            
            # Sort by time
            sort_idx = np.argsort(color_times)
            color_times = color_times[sort_idx]
            color_measurements = color_measurements[sort_idx]
            
            mean_color = np.mean(color_measurements)
            
            # Calculate rate using simple linear fit to actual measurements
            if len(color_measurements) >= 2:
                # Simple linear regression
                coeffs = np.polyfit(color_times, color_measurements, 1)
                mean_rate = coeffs[0]  # Slope
                
                # Sanity check: typical SNe Ia color evolution is ~0.01-0.05 mag/day
                if abs(mean_rate) > 0.1:
                    mean_rate = 0.0
            else:
                mean_rate = 0.0

        # Calculate g-r at g-band peak
        tpg = self.g["time"][np.argmin(self.g["mag"])]
        try:
            gr_at_gpeak = self.g["mag"][np.argmin(self.g["mag"])] - np.interp(
                tpg, r_time, r_mag
            )
        except Exception:
            gr_at_gpeak = np.nan
        
        return mean_color, gr_at_gpeak, mean_rate

    def _rolling_variance(self, band, window_size=5):
        """Max & mean variance in sliding windows over an interpolated LC.

        Parameters
        ----------
        window_size : int, default 5
            Number of interpolated samples per window.

        Returns
        -------
        tuple
            *(max_var, mean_var)*
        """
        def dedup(t, m):
            _, idx = np.unique(t, return_index=True)
            return t[idx], m[idx]

        t_dedup, m_dedup = dedup(band["time"], band["mag"])
        t_uniform = np.linspace(t_dedup.min(), t_dedup.max(), 100)
        mag_interp = interp1d(
            t_dedup, m_dedup, kind="linear", fill_value="extrapolate"
        )(t_uniform)
        views = sliding_window_view(mag_interp, window_shape=window_size)
        rolling_vars = np.var(views, axis=1)
        return np.max(rolling_vars), np.mean(rolling_vars)

    def _peak_structure(self, band):
        """Secondary-peak diagnostics using SciPy ``find_peaks``.

        Returns
        -------
        tuple
            *(n_peaks, Δt, Δmag, prominence₂, width₂)* with NaNs when <2 peaks.
        """
        if np.ptp(band["mag"]) < 0.5:
            return 0, np.nan, np.nan, np.nan, np.nan
        t_uniform = np.linspace(band["time"].min(), band["time"].max(), 300)
        mag_interp = interp1d(
            band["time"], band["mag"], kind="linear", fill_value="extrapolate"
        )(t_uniform)
        
        # Try strict parameters first
        peaks, properties = find_peaks(-mag_interp, prominence=0.1, width=5)
        
        # If no peaks found, try more relaxed parameters to ensure we find the main peak
        if len(peaks) == 0:
            peaks, properties = find_peaks(-mag_interp, prominence=0.05, width=2)
        
        # If still no peaks, find the global minimum as the peak
        if len(peaks) == 0:
            main_peak_idx = np.argmin(mag_interp)
            return 1, np.nan, np.nan, np.nan, np.nan
        
        n_peaks = len(peaks)
        if n_peaks < 2:
            return n_peaks, np.nan, np.nan, np.nan, np.nan
        mags = mag_interp[peaks]
        times = t_uniform[peaks]
        prominences = properties["prominences"]
        widths = properties["widths"]
        main_idx = np.argmin(mags)
        other_idx = np.argsort(mags)[1]
        dt = np.abs(times[main_idx] - times[other_idx])
        dmag = mags[other_idx] - mags[main_idx]
        prominence_second = prominences[other_idx]
        width_second = widths[other_idx]
        return n_peaks, dt, dmag, prominence_second, width_second

    def _local_curvature_features(self, band, window_days=20):
        """Median curvature on the rise and decline within ±*window_days* of peak.

        Returns
        -------
        tuple
            ``(rise_curvature, decline_curvature)``
        """
        t, m = band["time"], band["mag"]
        mask = np.isfinite(t) & np.isfinite(m)
        t, m = t[mask], m[mask]
        if len(t) < 3:
            return np.nan, np.nan

        # Sort and deduplicate
        t, idx = np.unique(t, return_index=True)
        m = m[idx]

        # Identify peak time
        peak_idx = np.argmin(m)
        t_peak = t[peak_idx]

        # Define ±window around peak
        tmin = t_peak - window_days
        tmax = t_peak + window_days
        local_mask = (t >= tmin) & (t <= tmax)
        t_local, m_local = t[local_mask], m[local_mask]
        if len(t_local) < 3:
            return np.nan, np.nan

        # Split into rise and decline
        rise_t, rise_m = t_local[t_local <= t_peak], m_local[t_local <= t_peak]
        decline_t, decline_m = t_local[t_local >= t_peak], m_local[t_local >= t_peak]

        rise_curv = local_curvature(rise_t, rise_m)
        decline_curv = local_curvature(decline_t, decline_m)
        return rise_curv, decline_curv

    def _total_duration_feature(self, band_g, band_r):
        """Calculate total observational duration - essential for extreme transient detection.
        
        Returns
        -------
        float
            Total end-to-end duration in days
        """
        # Get combined time arrays
        all_times_g = band_g["time"][np.isfinite(band_g["time"]) & np.isfinite(band_g["mag"])]
        all_times_r = band_r["time"][np.isfinite(band_r["time"]) & np.isfinite(band_r["mag"])]
        
        if len(all_times_g) > 0 and len(all_times_r) > 0:
            all_times = np.concatenate([all_times_g, all_times_r])
        elif len(all_times_g) > 0:
            all_times = all_times_g
        elif len(all_times_r) > 0:
            all_times = all_times_r
        else:
            return np.nan
        
        if len(all_times) > 1:
            return np.max(all_times) - np.min(all_times)
        else:
            return 0.0

    def extract_features(self, return_uncertainty=False, n_trials=20):
        """Generate the full reLAISS feature vector for the supplied LC.

        Parameters
        ----------
        return_uncertainty : bool, default False
            If True, performs *n_trials* MC perturbations and appends 1-σ errors
            (columns with ``_err`` suffix).
        n_trials : int, default 20
            Number of Monte-Carlo resamples when *return_uncertainty* is True.

        Returns
        -------
        pandas.DataFrame | None
            Single-row feature table (with optional error columns) or *None* when
            either band lacks data after pre-processing.
        """
        if len(self.g["time"]) == 0 or len(self.r["time"]) == 0:
            return None

        g_core = self._core_stats(self.g)
        r_core = self._core_stats(self.r)

        g_var = self._variability_stats(self.g)
        r_var = self._variability_stats(self.r)

        color_feats = self._color_features()

        g_rise_curv, g_decline_curv = self._local_curvature_features(self.g)
        r_rise_curv, r_decline_curv = self._local_curvature_features(self.r)

        if color_feats is None:
            color_feats = (np.nan, np.nan, np.nan)
        g_peak_struct = self._peak_structure(self.g)
        r_peak_struct = self._peak_structure(self.r)
        g_rollvar = self._rolling_variance(self.g)
        r_rollvar = self._rolling_variance(self.r)
        
        # Calculate total duration for extreme transient detection
        total_duration = self._total_duration_feature(self.g, self.r)
        
        base_row = {
            "t0": self.time_offset,
            "g_peak_mag": g_core[0],
            "g_peak_time": g_core[1],
            "g_rise_time": g_core[2],
            "g_decline_time": g_core[3],
            "g_duration_above_half_flux": g_core[4],
            "g_amplitude": g_var[0],
            "g_skewness": g_var[1],
            "g_beyond_2sigma": g_var[2],
            "r_peak_mag": r_core[0],
            "r_peak_time": r_core[1],
            "r_rise_time": r_core[2],
            "r_decline_time": r_core[3],
            "r_duration_above_half_flux": r_core[4],
            "r_amplitude": r_var[0],
            "r_skewness": r_var[1],
            "r_beyond_2sigma": r_var[2],
            "mean_g-r": color_feats[0],
            "g-r_at_g_peak": color_feats[1],
            "mean_color_rate": color_feats[2],
            "g_n_peaks": g_peak_struct[0],
            "g_dt_main_to_secondary_peak": g_peak_struct[1],
            "g_dmag_secondary_peak": g_peak_struct[2],
            "g_secondary_peak_prominence": g_peak_struct[3],
            "g_secondary_peak_width": g_peak_struct[4],
            "r_n_peaks": r_peak_struct[0],
            "r_dt_main_to_secondary_peak": r_peak_struct[1],
            "r_dmag_secondary_peak": r_peak_struct[2],
            "r_secondary_peak_prominence": r_peak_struct[3],
            "r_secondary_peak_width": r_peak_struct[4],
            "g_max_rolling_variance": g_rollvar[0],
            "g_mean_rolling_variance": g_rollvar[1],
            "r_max_rolling_variance": r_rollvar[0],
            "r_mean_rolling_variance": r_rollvar[1],
            "g_rise_local_curvature": g_rise_curv,
            "g_decline_local_curvature": g_decline_curv,
            "r_rise_local_curvature": r_rise_curv,
            "r_decline_local_curvature": r_decline_curv,
            # Essential extreme transient detection feature
            "total_duration": total_duration,
        }
        
        base_row["features_valid"] = all(
            not np.isnan(base_row[k])
            for k in [
                "g_peak_time",
                "g_rise_time",
                "g_decline_time",
                "g_duration_above_half_flux",
                "r_peak_time",
                "r_rise_time",
                "r_decline_time",
                "r_duration_above_half_flux",
                "mean_g-r",
                "g-r_at_g_peak",
                "mean_color_rate",
                "g_rise_local_curvature",
                "g_decline_local_curvature",
                "r_rise_local_curvature",
                "r_decline_local_curvature",
            ]
        )
        if not return_uncertainty:
            return pd.DataFrame([base_row])
        results = []
        for _ in range(n_trials):
            perturbed_g = self.g["mag"] + np.random.normal(0, self.g["err"])
            perturbed_r = self.r["mag"] + np.random.normal(0, self.r["err"])
            f = SupernovaFeatureExtractor(
                time_g=self.g["time"],
                mag_g=perturbed_g,
                err_g=self.g["err"],
                time_r=self.r["time"],
                mag_r=perturbed_r,
                err_r=self.r["err"],
                ztf_object_id=self.ztf_object_id,
            )
            results.append(f.extract_features().iloc[0])
        df = pd.DataFrame(results)
        uncertainty = df.std().add_suffix("_err")
        return pd.DataFrame([{**base_row, **uncertainty.to_dict()}])
