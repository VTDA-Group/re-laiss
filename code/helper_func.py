import constants
from lightcurve_engineer import *
import numpy as np
import pandas as pd
import time
import math
from pathlib import Path
import os
import sys
import warnings
from contextlib import contextmanager
import io
import logging
import requests
import antares_client
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sfdmap2 import sfdmap
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from astropy.visualization import AsinhStretch, PercentileInterval
from scipy.stats import gamma, uniform
from dust_extinction.parameter_averages import G23
from astro_prost.associate import associate_sample


@contextmanager
def re_suppress_output():
    """Temporarily silence stdout, stderr, warnings *and* all logging messages < CRITICAL."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull

        logging.disable(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            logging.disable(logging.NOTSET)
            sys.stdout, sys.stderr = old_stdout, old_stderr


def re_getTnsData(ztf_id):
    locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id)
    try:
        tns = locus.catalog_objects["tns_public_objects"][0]
        tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
    except:
        tns_name, tns_cls, tns_z = "No TNS", "---", -99
    if tns_cls == "":
        tns_cls, tns_ann_z = "---", -99
    return tns_name, tns_cls, tns_z


def re_getExtinctionCorrectedMag(
    transient_row,
    band,
    av_in_raw_df_bank,
    path_to_sfd_folder=None,
):
    central_wv_filters = {"g": 4849.11, "r": 6201.20, "i": 7534.96, "z": 8674.20}
    MW_RV = 3.1
    ext = G23(Rv=MW_RV)

    if av_in_raw_df_bank:
        MW_AV = transient_row["A_V"]
    else:
        m = sfdmap.SFDMap(path_to_sfd_folder)
        MW_EBV = m.ebv(float(transient_row["ra"]), float(transient_row["dec"]))
        MW_AV = MW_RV * MW_EBV

    wv_filter = central_wv_filters[band]
    A_filter = -2.5 * np.log10(ext.extinguish(wv_filter * u.AA, Av=MW_AV))

    return transient_row[band + "KronMag"] - A_filter


def re_build_dataset_bank(
    raw_df_bank,
    av_in_raw_df_bank,
    path_to_sfd_folder=None,
    theorized=False,
    path_to_dataset_bank=None,
    building_entire_df_bank=False,
    building_for_AD=False,
):

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

    if theorized:
        raw_features = raw_lc_features
        raw_feats_no_ztf = raw_lc_features
    else:
        raw_features = ["ztf_object_id"] + raw_lc_features + raw_host_features
        raw_feats_no_ztf = raw_lc_features + raw_host_features

    # Check to make sure all required features are in the raw data
    missing_cols = [col for col in raw_features if col not in raw_df_bank.columns]
    if missing_cols:
        print(
            f"KeyError: The following columns for this transient are not in the raw data: {missing_cols}. Abort!"
        )
        return

    # Impute missing features
    test_dataset_bank = raw_df_bank.replace([np.inf, -np.inf, -999], np.nan).dropna(
        subset=raw_features
    )

    nan_cols = [
        col
        for col in raw_features
        if raw_df_bank[col].replace([np.inf, -np.inf, -999], np.nan).isna().all()
    ]

    if not building_for_AD:
        print(
            f"There are {len(raw_df_bank) - len(test_dataset_bank)} of {len(raw_df_bank)} rows in the timeseries dataframe with 1 or more NA features."
        )
        if len(nan_cols) != 0:
            print(
                f"The following {len(nan_cols)} feature(s) are NaN for all measurements: {nan_cols}."
            )
        print("Imputing features (if necessary)...")

    wip_dataset_bank = raw_df_bank

    if building_entire_df_bank:
        X = raw_df_bank[raw_feats_no_ztf]

        feat_imputer = KNNImputer(weights="distance").fit(X)
        imputed_filt_arr = feat_imputer.transform(X)
    else:
        true_raw_df_bank = pd.read_csv(path_to_dataset_bank)
        X = true_raw_df_bank[raw_feats_no_ztf]

        if building_for_AD:
            # Use mean imputation
            feat_imputer = SimpleImputer(strategy="mean").fit(X)
        else:
            # Use KNN imputation
            feat_imputer = KNNImputer(weights="distance").fit(X)

        imputed_filt_arr = feat_imputer.transform(wip_dataset_bank[raw_feats_no_ztf])

    imputed_filt_df = pd.DataFrame(imputed_filt_arr, columns=raw_feats_no_ztf)
    imputed_filt_df.index = raw_df_bank.index

    wip_dataset_bank[raw_feats_no_ztf] = imputed_filt_df

    wip_dataset_bank = wip_dataset_bank.replace([np.inf, -np.inf, -999], np.nan).dropna(
        subset=raw_features
    )

    if not building_for_AD:
        if not wip_dataset_bank.empty:
            print("Successfully imputed features.")
        else:
            print("Failed to impute features.")

    # Engineer the remaining features
    if not theorized:
        if not building_for_AD:
            print(f"Engineering remaining features...")
        # Correct host magnitude features for dust
        for band in ["g", "r", "i", "z"]:
            wip_dataset_bank[band + "KronMagCorrected"] = wip_dataset_bank.apply(
                lambda row: re_getExtinctionCorrectedMag(
                    transient_row=row,
                    band=band,
                    av_in_raw_df_bank=av_in_raw_df_bank,
                    path_to_sfd_folder=path_to_sfd_folder,
                ),
                axis=1,
            )

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

    return final_df_bank


def re_extract_lc_and_host_features(
    ztf_id,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
    path_to_dataset_bank=None,
    theorized_lightcurve_df=None,
    show_lc=False,
    show_host=True,
    store_csv=False,
    building_for_AD=False,
    swapped_host=False,
):
    start_time = time.time()
    df_path = path_to_timeseries_folder

    # Look up transient
    if theorized_lightcurve_df is not None:
        df_ref = theorized_lightcurve_df
        # Ensure correct capitalization of passbands ('g' and 'R')
        df_ref["ant_passband"] = df_ref["ant_passband"].replace({"G": "g", "r": "R"})
    else:
        try:
            ref_info = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id)
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
                ZTFID=ztf_id,
            )

            engineered_lc_properties_df = extractor.extract_features(
                return_uncertainty=True
            )
        except:
            continue

        if engineered_lc_properties_df is not None:

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
            with re_suppress_output():
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
                hosts_df = pd.DataFrame(hosts.loc[0]).T
            else:
                print(f"Cannot identify host galaxy for {ztf_id}. Abort!\n")
                return

            # Check if required host features are missing
            try:
                raw_host_feature_check = constants.raw_host_features_const.copy()
                hosts_df = hosts[raw_host_feature_check]
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

        # Lightcurve ra and dec may be needed in feature engineering
        lc_and_hosts_df["ra"] = ra
        lc_and_hosts_df["dec"] = dec

    # Engineer additonal features in build_dataset_bank function
    if building_for_AD:
        print("Engineering features...")
    lc_and_hosts_df_hydrated = re_build_dataset_bank(
        raw_df_bank=(
            lc_and_hosts_df
            if theorized_lightcurve_df is None
            else lc_timeseries_feat_df
        ),
        av_in_raw_df_bank=False,
        path_to_sfd_folder=path_to_sfd_data_folder,
        theorized=True if theorized_lightcurve_df is not None else False,
        path_to_dataset_bank=path_to_dataset_bank,
        building_for_AD=building_for_AD,
    )
    if building_for_AD:
        print("Finished engineering features.\n")

    if store_csv and not lc_and_hosts_df_hydrated.empty:
        os.makedirs(df_path, exist_ok=True)
        if theorized_lightcurve_df is None:
            lc_and_hosts_df_hydrated.to_csv(f"{df_path}/{ztf_id}_timeseries.csv")
            print(f"Saved timeseries features for {ztf_id}!\n")
        else:
            lc_and_hosts_df_hydrated.to_csv(f"{df_path}/theorized_timeseries.csv")
            print(f"Saved timeseries features for theorized lightcurve!\n")

    return lc_and_hosts_df_hydrated


def _ps1_list_filenames(ra_deg, dec_deg, flt):
    """
    Return the first stack FITS filename for (ra,dec) and *flt* or None.
    """
    url = (
        "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        f"?ra={ra_deg}&dec={dec_deg}&filters={flt}"
    )
    for line in requests.get(url, timeout=20).text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for tok in line.split():
            if tok.endswith(".fits"):
                return tok
    return None


def fetch_ps1_cutout(ra_deg, dec_deg, *, size_pix=100, flt="r"):
    """
    Grayscale cut-out (2-D float) in a single PS1 filter.
    """
    fits_name = _ps1_list_filenames(ra_deg, dec_deg, flt)
    if fits_name is None:
        raise RuntimeError(f"No {flt}-band stack at this position")

    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
        f"?ra={ra_deg}&dec={dec_deg}&size={size_pix}"
        f"&format=fits&filters={flt}&red={fits_name}"
    )
    r = requests.get(url, timeout=40)
    if r.status_code == 400:
        raise RuntimeError("Outside PS1 footprint or no data in this filter")
    r.raise_for_status()

    with fits.open(io.BytesIO(r.content)) as hdul:
        data = hdul[0].data.astype(float)

    if data is None or data.size == 0 or (data != data).all():
        raise RuntimeError("Empty FITS array returned")

    data[data != data] = 0.0
    return data


def fetch_ps1_rgb_jpeg(ra_deg, dec_deg, *, size_pix=100):
    """
    Colour JPEG (H,W,3  uint8) using PS1 g/r/i stacks.
    Falls back by *raising* RuntimeError when the server lacks colour data.
    """
    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
        f"?ra={ra_deg}&dec={dec_deg}&size={size_pix}"
        f"&format=jpeg&filters=grizy&red=i&green=r&blue=g&autoscale=99.5"
    )
    r = requests.get(url, timeout=40)
    if r.status_code == 400:
        raise RuntimeError("Outside PS1 footprint or no colour data here")
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return np.array(img)


def re_plot_lightcurves(
    primer_dict,
    plot_label,
    theorized_lightcurve_df,
    neighbor_ztfids,
    ann_locus_l,
    ann_dists,
    tns_ann_names,
    tns_ann_classes,
    tns_ann_zs,
    figure_path,
    save_figures=True,
):
    print("Making a plot of stacked lightcurves...")

    if primer_dict["lc_tns_z"] is None:
        primer_dict["lc_tns_z"] = "None"
    elif isinstance(primer_dict["lc_tns_z"], float):
        primer_dict["lc_tns_z"] = round(primer_dict["lc_tns_z"], 3)
    else:
        primer_dict["lc_tns_z"] = primer_dict["lc_tns_z"]

    if primer_dict["lc_ztf_id"] is not None:
        ref_info = antares_client.search.get_by_ztf_object_id(
            ztf_object_id=primer_dict["lc_ztf_id"]
        )
        try:
            df_ref = ref_info.timeseries.to_pandas()
        except:
            raise ValueError(f"{ztf_id} has no timeseries data.")
    else:
        df_ref = theorized_lightcurve_df

    fig, ax = plt.subplots(figsize=(9.5, 6))

    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

    ax.errorbar(
        x=df_ref_r.ant_mjd - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref],
        y=df_ref_r.ant_mag.min() - df_ref_r.ant_mag,
        yerr=df_ref_r.ant_magerr,
        fmt="o",
        c="r",
        label=plot_label
        + f",\nd=0, {primer_dict['lc_tns_name']}, {primer_dict['lc_tns_cls']}, z={primer_dict['lc_tns_z']}",
    )
    ax.errorbar(
        x=df_ref_g.ant_mjd - df_ref_g.ant_mjd.iloc[mjd_idx_at_min_mag_g_ref],
        y=df_ref_g.ant_mag.min() - df_ref_g.ant_mag,
        yerr=df_ref_g.ant_magerr,
        fmt="o",
        c="g",
    )

    markers = ["s", "*", "x", "P", "^", "v", "D", "<", ">", "8", "p", "x"]
    consts = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]

    for num, (l_info, ztfname, dist, iau_name, spec_cls, z) in enumerate(
        zip(
            ann_locus_l,
            neighbor_ztfids,
            ann_dists,
            tns_ann_names,
            tns_ann_classes,
            tns_ann_zs,
        )
    ):
        # Plots up to 8 neighbors
        if num + 1 > 8:
            print(
                "Lightcurve plotter only plots up to 8 neighbors. Stopping at neighbor 8."
            )
            break
        try:
            alpha = 0.25
            c1 = "darkred"
            c2 = "darkgreen"

            df_knn = l_info.timeseries.to_pandas()

            df_g = df_knn[(df_knn.ant_passband == "g") & (~df_knn.ant_mag.isna())]
            df_r = df_knn[(df_knn.ant_passband == "R") & (~df_knn.ant_mag.isna())]

            mjd_idx_at_min_mag_r = df_r[["ant_mag"]].reset_index().idxmin().ant_mag
            mjd_idx_at_min_mag_g = df_g[["ant_mag"]].reset_index().idxmin().ant_mag

            ax.errorbar(
                x=df_r.ant_mjd - df_r.ant_mjd.iloc[mjd_idx_at_min_mag_r],
                y=df_r.ant_mag.min() - df_r.ant_mag,
                yerr=df_r.ant_magerr,
                fmt=markers[num],
                c=c1,
                alpha=alpha,
                label=f"ANN={num+1}:{ztfname}, d={round(dist, 2)},\n{iau_name}, {spec_cls}, z={round(z, 3)}",
            )
            ax.errorbar(
                x=df_g.ant_mjd - df_g.ant_mjd.iloc[mjd_idx_at_min_mag_g],
                y=df_g.ant_mag.min() - df_g.ant_mag,
                yerr=df_g.ant_magerr,
                fmt=markers[num],
                c=c2,
                alpha=alpha,
            )

            plt.ylabel("Apparent Mag. + Constant")
            plt.xlabel("Days since peak ($r$, $g$ indep.)")  # (need r, g to be same)

            if (
                df_ref_r.ant_mjd.iloc[0]
                - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                <= 10
            ):
                plt.xlim(
                    (
                        df_ref_r.ant_mjd.iloc[0]
                        - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                    )
                    - 20,
                    df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                )
            else:
                plt.xlim(
                    2
                    * (
                        df_ref_r.ant_mjd.iloc[0]
                        - df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]
                    ),
                    df_ref_r.ant_mjd.iloc[-1] - df_ref_r.ant_mjd.iloc[0] + 15,
                )

            shift, scale = 1.4, 0.975
            if len(neighbor_ztfids) <= 2:
                shift = 1.175
                scale = 0.9
            elif len(neighbor_ztfids) <= 5:
                shift = 1.3
                scale = 0.925

            plt.legend(
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, shift),
                ncol=3,
                prop={"size": 10},
            )
            plt.grid(True)

            # Shrink axes to leave space above for the legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * scale])

        except Exception as e:
            print(f"Something went wrong with plotting {ztfname}! Excluding from plot.")

    if save_figures:
        os.makedirs(figure_path, exist_ok=True)
        os.makedirs(figure_path + "/lightcurves", exist_ok=True)
        plt.savefig(
            figure_path + f"/lightcurves/{plot_label}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Saved lightcurve plot to:" + figure_path + f"/lightcurves/{plot_label}.png"
        )
    plt.show()


def re_plot_hosts(
    ztfid_ref,
    plot_label,
    df,
    figure_path,
    ann_num,
    save_pdf=True,
    imsizepix=100,
    change_contrast=False,
    prefer_color=True,
):
    """
    Build 3×3 grids of PS1 thumbnails for each row in *df* and write a PDF.

    Set *prefer_color=False* for r-band grayscale only.  With *prefer_color=True*
    (default) the code *tries* colour first and quietly falls back to grayscale
    when colour isn’t available.
    """

    host_grid_path = figure_path + "/host_grids"
    pdf_path = Path(host_grid_path) / f"{plot_label}_host_thumbnails_ann={ann_num}.pdf"
    if save_pdf:
        os.makedirs(figure_path, exist_ok=True)
        Path(host_grid_path).mkdir(parents=True, exist_ok=True)
    pdf_pages = PdfPages(pdf_path) if save_pdf else None

    logging.basicConfig(level=logging.INFO, format="%(levelname)7s : %(message)s")
    rows = cols = 3
    per_page = rows * cols
    pages = math.ceil(len(df) / per_page)

    for pg in range(pages):
        fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
        axs = axs.ravel()

        for k in range(per_page):
            idx = pg * per_page + k
            ax = axs[k]
            ax.set_xticks([])
            ax.set_yticks([])

            if idx >= len(df):
                ax.axis("off")
                continue

            row = df.iloc[idx]
            ztfid, ra, dec = (
                str(row["ZTFID"]),
                float(row["HOST_RA"]),
                float(row["HOST_DEC"]),
            )

            try:
                # validate coordinates
                if np.isnan(ra) or np.isnan(dec):
                    raise ValueError("NaN coordinate")
                SkyCoord(ra * u.deg, dec * u.deg)

                # Attempt colour first (if requested), then grayscale fallback
                if prefer_color:
                    try:
                        im = fetch_ps1_rgb_jpeg(ra, dec, size_pix=imsizepix)
                        ax.imshow(im, origin="lower")
                    except Exception as col_err:
                        im = fetch_ps1_cutout(ra, dec, size_pix=imsizepix, flt="r")
                        stretch = AsinhStretch() + PercentileInterval(
                            93 if change_contrast else 99.5
                        )
                        ax.imshow(stretch(im), cmap="gray", origin="lower")
                else:
                    im = fetch_ps1_cutout(ra, dec, size_pix=imsizepix, flt="r")
                    stretch = AsinhStretch() + PercentileInterval(
                        93 if change_contrast else 99.5
                    )
                    ax.imshow(stretch(im), cmap="gray", origin="lower")

                ax.set_title(ztfid, fontsize=8, pad=1.5)

            except Exception as e:
                logging.warning(f"{ztfid}: {e}")
                ax.imshow(np.full((imsizepix, imsizepix, 3), [1.0, 0, 0]))
                ax.set_title("", fontsize=8, pad=1.5)

        plt.tight_layout(pad=0.2)
        if pdf_pages:
            pdf_pages.savefig(fig, bbox_inches="tight", pad_inches=0.05)
        plt.show(block=False)
        plt.close(fig)

    if pdf_pages:
        pdf_pages.close()
        print(f"PDF written to {pdf_path}\n")


def re_check_anom_and_plot(
    clf,
    input_ztf_id,
    swapped_host_ztf_id,
    input_spec_cls,
    input_spec_z,
    anom_thresh,
    timeseries_df_full,
    timeseries_df_features_only,
    ref_info,
    savefig,
    figure_path,
):
    anom_obj_df = timeseries_df_features_only

    pred_prob_anom = 100 * clf.predict_proba(anom_obj_df)
    pred_prob_anom[:, 0] = [round(a, 1) for a in pred_prob_anom[:, 0]]
    pred_prob_anom[:, 1] = [round(b, 1) for b in pred_prob_anom[:, 1]]
    num_anom_epochs = len(np.where(pred_prob_anom[:, 1] >= anom_thresh)[0])

    try:
        anom_idx = timeseries_df_full.iloc[
            np.where(pred_prob_anom[:, 1] >= anom_thresh)[0][0]
        ].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(
            f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {input_ztf_id}."
            + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else "")
        )
        anom_idx_is = False

    max_anom_score = max(pred_prob_anom[:, 1])
    print("max_anom_score", round(max_anom_score, 1))
    print("num_anom_epochs", num_anom_epochs, "\n")

    df_ref = ref_info.timeseries.to_pandas()

    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 10))
    ax1.invert_yaxis()
    ax1.errorbar(
        x=df_ref_r.ant_mjd,
        y=df_ref_r.ant_mag,
        yerr=df_ref_r.ant_magerr,
        fmt="o",
        c="r",
        label=r"ZTF-$r$",
    )
    ax1.errorbar(
        x=df_ref_g.ant_mjd,
        y=df_ref_g.ant_mag,
        yerr=df_ref_g.ant_magerr,
        fmt="o",
        c="g",
        label=r"ZTF-$g$",
    )
    if anom_idx_is == True:
        ax1.axvline(
            x=timeseries_df_full[
                timeseries_df_full.obs_num == anom_idx
            ].mjd_cutoff.values[0],
            label="Tag anomalous",
            color="dodgerblue",
            ls="--",
        )
        mjd_cross_thresh = round(
            timeseries_df_full[
                timeseries_df_full.obs_num == anom_idx
            ].mjd_cutoff.values[0],
            3,
        )

        left, right = ax1.get_xlim()
        mjd_anom_per = (mjd_cross_thresh - left) / (right - left)
        plt.text(
            mjd_anom_per + 0.073,
            -0.075,
            f"t$_a$ = {int(mjd_cross_thresh)}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes,
            fontsize=16,
            color="dodgerblue",
        )
        print("MJD crossed thresh:", mjd_cross_thresh)

    ax2.plot(
        timeseries_df_full.mjd_cutoff,
        pred_prob_anom[:, 0],
        drawstyle="steps",
        label=r"$p(Normal)$",
    )
    ax2.plot(
        timeseries_df_full.mjd_cutoff,
        pred_prob_anom[:, 1],
        drawstyle="steps",
        label=r"$p(Anomaly)$",
    )

    if input_spec_z is None:
        input_spec_z = "None"
    elif isinstance(input_spec_z, float):
        input_spec_z = round(input_spec_z, 3)
    else:
        input_spec_z = input_spec_z
    ax1.set_title(
        rf"{input_ztf_id} ({input_spec_cls}, $z$={input_spec_z})"
        + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else ""),
        pad=25,
    )
    plt.xlabel("MJD")
    ax1.set_ylabel("Magnitude")
    ax2.set_ylabel("Probability (%)")

    if anom_idx_is == True:
        ax1.legend(
            loc="upper right",
            ncol=3,
            bbox_to_anchor=(1.0, 1.12),
            frameon=False,
            fontsize=14,
        )
    else:
        ax1.legend(
            loc="upper right",
            ncol=2,
            bbox_to_anchor=(0.75, 1.12),
            frameon=False,
            fontsize=14,
        )
    ax2.legend(
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.87, 1.12),
        frameon=False,
        fontsize=14,
    )

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        os.makedirs(figure_path, exist_ok=True)
        os.makedirs(figure_path + "/AD", exist_ok=True)
        plt.savefig(
            (
                f"{figure_path}/AD/{input_ztf_id}"
                + (f"_w_host_{swapped_host_ztf_id}" if swapped_host_ztf_id else "")
                + "_AD.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Saved anomaly detection chart to:"
            + f"{figure_path}/AD/{input_ztf_id}"
            + (f"_w_host_{swapped_host_ztf_id}" if swapped_host_ztf_id else "")
            + "_AD.pdf"
        )
    plt.show()


def re_get_timeseries_df(
    ztf_id,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
    theorized_lightcurve_df=None,
    save_timeseries=False,
    path_to_dataset_bank=None,
    building_for_AD=False,
    swapped_host=False,
):
    if theorized_lightcurve_df is not None:
        print("Extracting full lightcurve features for theorized lightcurve...")
        timeseries_df = re_extract_lc_and_host_features(
            ztf_id=ztf_id,
            theorized_lightcurve_df=theorized_lightcurve_df,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            show_lc=False,
            show_host=True,
            store_csv=save_timeseries,
            swapped_host=swapped_host,
        )
        return timeseries_df

    # Check if timeseries already made (but must rebuild for AD regardless)
    if (
        os.path.exists(f"{path_to_timeseries_folder}/{ztf_id}_timeseries.csv")
        and not building_for_AD
    ):
        timeseries_df = pd.read_csv(
            f"{path_to_timeseries_folder}/{ztf_id}_timeseries.csv"
        )
        print(f"Timeseries dataframe for {ztf_id} is already made. Continue!\n")
    else:
        # If timeseries is not made or building for AD, create timeseries by extracting features
        if not building_for_AD:
            print(
                f"Timeseries dataframe does not exist. Re-extracting lightcurve and host features for {ztf_id}."
            )
        timeseries_df = re_extract_lc_and_host_features(
            ztf_id=ztf_id,
            theorized_lightcurve_df=theorized_lightcurve_df,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            show_lc=False,
            show_host=True,
            store_csv=save_timeseries,
            building_for_AD=building_for_AD,
            swapped_host=swapped_host,
        )
    return timeseries_df


def create_re_laiss_features_dict(
    lc_feature_names, host_feature_names, lc_groups=4, host_groups=4
):
    re_laiss_features_dict = {}

    # Split light curve features into evenly sized chunks
    lc_chunk_size = math.ceil(len(lc_feature_names) / lc_groups)
    for i in range(lc_groups):
        start = i * lc_chunk_size
        end = start + lc_chunk_size
        chunk = lc_feature_names[start:end]
        if chunk:
            re_laiss_features_dict[f"lc_group_{i+1}"] = chunk

    # Split host features into evenly sized chunks
    host_chunk_size = math.ceil(len(host_feature_names) / host_groups)
    for i in range(host_groups):
        start = i * host_chunk_size
        end = start + host_chunk_size
        chunk = host_feature_names[start:end]
        if chunk:
            re_laiss_features_dict[f"host_group_{i+1}"] = chunk

    return re_laiss_features_dict
