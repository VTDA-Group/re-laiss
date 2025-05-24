import os
import pickle
import time
from pathlib import Path

import antares_client
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from pyod.models.iforest import IForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from scipy.stats import rankdata
from pyod.models.iforest import IForest
import warnings

from .fetch import get_timeseries_df, get_TNS_data
from .features import build_dataset_bank

def train_AD_model(
    client,
    lc_features,
    host_features,
    path_to_dataset_bank=None,
    preprocessed_df=None,
    path_to_sfd_folder=None,
    path_to_models_directory="../models",
    n_estimators=500,
    contamination=0.02,
    max_samples=1024,
    force_retrain=True,
):
    """Train an Isolation Forest model for anomaly detection.

    Parameters
    ----------
    lc_features : list[str]
        Names of lightcurve features to use.
    host_features : list[str]
        Names of host galaxy features to use.
    path_to_dataset_bank : str | Path | None, optional
        Path to raw dataset bank CSV. Not used if preprocessed_df is provided.
    preprocessed_df : pandas.DataFrame | None, optional
        Pre-processed dataframe with imputed features. If provided, this is used
        instead of loading and processing the raw dataset bank.
    path_to_sfd_folder : str | Path | None, optional
        Path to SFD dust maps.
    path_to_models_directory : str | Path, default "../models"
        Directory to save trained models.
    n_estimators : int, default 500
        Number of trees in the Isolation Forest.
    contamination : float, default 0.02
        Expected fraction of outliers in the dataset.
    max_samples : int, default 1024
        Number of samples to draw for each tree.
    force_retrain : bool, default False
        Whether to retrain even if a saved model exists.

    Returns
    -------
    str
        Path to the saved model file.

    Notes
    -----
    Either path_to_dataset_bank or preprocessed_df must be provided.
    If both are provided, preprocessed_df takes precedence.
    """
    if preprocessed_df is None and path_to_dataset_bank is None:
        raise ValueError("Either path_to_dataset_bank or preprocessed_df must be provided")

    # Create models directory if it doesn't exist
    os.makedirs(path_to_models_directory, exist_ok=True)

    # Generate model filename based on parameters including feature counts
    num_lc_features = len(lc_features)
    num_host_features = len(host_features)
    model_name = f"IForest_n={n_estimators}_c={contamination}_m={max_samples}_lc={num_lc_features}_host={num_host_features}.pkl"
    model_path = os.path.join(path_to_models_directory, model_name)

    # Check if model already exists
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading existing model from {model_path}")
        return model_path

    print("Training new Isolation Forest model...")

    # Get features from preprocessed dataframe or load and process raw data
    if (preprocessed_df is not None) and (client.built_for_AD):
        print("Using provided preprocessed dataframe")
        df = preprocessed_df
    else:
        print("Loading and preprocessing dataset bank for AD...")
        raw_df = pd.read_csv(path_to_dataset_bank, low_memory=False)
        df = build_dataset_bank(
            raw_df,
            path_to_sfd_folder=path_to_sfd_folder,
            building_entire_df_bank=True,
            building_for_AD=True
        )

    # Extract features
    feature_names = lc_features + host_features

    X = df[feature_names].values

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                IForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    max_samples=max_samples,
                    behaviour="new",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X)

    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    return model_path


def anomaly_detection(
    client,
    transient_ztf_id,
    lc_features,
    host_features,
    path_to_timeseries_folder,
    path_to_sfd_folder,
    path_to_dataset_bank,
    host_ztf_id_to_swap_in=None,
    path_to_models_directory="../models",
    path_to_figure_directory="../figures",
    save_figures=True,
    n_estimators=500,
    contamination=0.02,
    max_samples=1024,
    force_retrain=True,
    preprocessed_df=None,
    return_scores=False
):
    """Run anomaly detection for a single transient (with optional host swap).

    Generates an AD probability plot and calls
    :func:`check_anom_and_plot`.

    Parameters
    ----------
    transient_ztf_id : str
        Target object ID.
    host_ztf_id_to_swap_in : str | None
        Replace host features before scoring.
    lc_features, host_features : list[str]
    path_* : folders for intermediates, models, and figures.
    save_figures : bool, default True
    n_estimators, contamination, max_samples : Isolation-Forest params.
    force_retrain : bool, default False
        Pass-through to :func:`train_AD_model`.
    preprocessed_df : pandas.DataFrame | None, optional
        Pre-processed dataframe with imputed features. If provided, this is used
        instead of loading and processing the raw dataset bank.

    Returns
    -------
    None
    """

    print("Running Anomaly Detection:\n")

    # Train the model (if necessary)
    path_to_trained_model = train_AD_model(
        client,
        lc_features,
        host_features,
        path_to_dataset_bank,
        preprocessed_df=preprocessed_df,
        path_to_sfd_folder=path_to_sfd_folder,
        path_to_models_directory=path_to_models_directory,
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        force_retrain=force_retrain,
    )

    # Load the model
    pipeline = joblib.load(path_to_trained_model)

    # If no preprocessed_df was provided, try to find a cached one
    if preprocessed_df is None:
        # Try to find the cached preprocessed dataframe used for training
        from .utils import get_cache_dir
        cache_dir = Path(get_cache_dir())

        for cache_file in cache_dir.glob("*.pkl"):
            if "dataset_bank" in str(cache_file) and not cache_file.name.startswith("timeseries"):
                try:
                    cached_df = joblib.load(cache_file)
                    if isinstance(cached_df, pd.DataFrame):
                        preprocessed_df = cached_df
                        print("Using cached preprocessed dataframe for feature extraction")
                        break
                except:
                    continue

    # Load the timeseries dataframe
    print("\nRebuilding timeseries dataframe(s) for AD...")
    timeseries_df = get_timeseries_df(
        ztf_id=transient_ztf_id,
        theorized_lightcurve_df=None,
        path_to_timeseries_folder=path_to_timeseries_folder,
        path_to_sfd_folder=path_to_sfd_folder,
        path_to_dataset_bank=path_to_dataset_bank,
        save_timeseries=True,
        building_for_AD=True#,
    #    preprocessed_df=preprocessed_df,
    )

    if host_ztf_id_to_swap_in is not None:
        # Swap in the host galaxy
        swapped_host_timeseries_df = get_timeseries_df(
            ztf_id=host_ztf_id_to_swap_in,
            theorized_lightcurve_df=None,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_folder=path_to_sfd_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            save_timeseries=False,
            building_for_AD=True,
            swapped_host=True#,
            #preprocessed_df=preprocessed_df,
        )

        host_values = swapped_host_timeseries_df[host_features].iloc[0]
        for col in host_features:
            timeseries_df[col] = host_values[col]

    timeseries_df.sort_values(by=['mjd_cutoff'], inplace=True)
    timeseries_df_filt_feats = timeseries_df[lc_features + host_features]
    input_lightcurve_locus = antares_client.search.get_by_ztf_object_id(
        ztf_object_id=transient_ztf_id
    )

    tns_name, tns_cls, tns_z = get_TNS_data(transient_ztf_id)

    # Suppress the UserWarning about feature names
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X has feature names, but IsolationForest was fitted without feature names")
        # Run the anomaly detection check
        mjd_anom, anom_scores, norm_scores = check_anom_and_plot(
            pipeline=pipeline,
            input_ztf_id=transient_ztf_id,
            swapped_host_ztf_id=host_ztf_id_to_swap_in,
            input_spec_cls=tns_cls,
            input_spec_z=tns_z,
            anom_thresh=70,
            timeseries_df_full=timeseries_df,
            timeseries_df_features_only=timeseries_df_filt_feats,
            ref_info=input_lightcurve_locus,
            savefig=save_figures,
            figure_path=path_to_figure_directory,
        )
    if not return_scores:
        return
    else:
        return mjd_anom, anom_scores, norm_scores


def check_anom_and_plot(
    pipeline,
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
    """Run anomaly-detector probabilities over a time-series and plot results.

    Produces a two-panel figure: light curve with anomaly epoch marked, and
    rolling anomaly/normal probabilities.

    Parameters
    ----------
    clf : sklearn.base.ClassifierMixin
        Trained binary classifier with ``predict_proba``.
    input_ztf_id : str
        ID of the object evaluated.
    swapped_host_ztf_id : str | None
        Alternate host ID (annotated in title).
    input_spec_cls : str | None
        Spectroscopic class label for title.
    input_spec_z : float | str | None
        Redshift for title.
    anom_thresh : float
        Legacy parameter, now overridden to 70%.
    timeseries_df_full : pandas.DataFrame
        Hydrated LC + host features
    timeseries_df_features_only : pandas.DataFrame
        Same rows but feature columns only (classifier input).
    ref_info : antares_client.objects.Locus
        ANTARES locus for retrieving original photometry.
    savefig : bool
        Save the plot as ``AD/*.pdf`` inside *figure_path*.
    figure_path : str | Path
        Output directory.

    Returns
    -------
    None
    """
    timeseries_df_full = timeseries_df_full.copy()
    timeseries_df_full.sort_values(by=['mjd_cutoff'], inplace=True)

    # Get anomaly scores from decision_function (-ve = anomalous, +ve = normal)
    pred_prob_anom = 100 * pipeline.predict_proba(np.array(timeseries_df_features_only))
    anom_mjd = timeseries_df_full.mjd_cutoff
    norm_score = np.array([round(a, 1) for a in pred_prob_anom[:, 0]])
    anom_score = np.array([round(b, 1) for b in pred_prob_anom[:, 1]])
    num_anom_epochs = len(np.where(anom_score >= anom_thresh)[0])

    try:
        anom_idx = timeseries_df_full.iloc[
            np.where(anom_score >= anom_thresh)[0][0]
        ].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(
            f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {input_ztf_id}."
            + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else "")
        )
        anom_idx_is = False

    max_anom_score = np.max(anom_score) if anom_score.size > 0 else 0

    # Get the light curve data
    df_ref = ref_info.timeseries.to_pandas()

    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

    # Calculate the actual range of MJD values in the light curve for setting axis limits
    min_mjd = min(df_ref_g['ant_mjd'].min(), df_ref_r['ant_mjd'].min())
    max_mjd = max(df_ref_g['ant_mjd'].max(), df_ref_r['ant_mjd'].max())

    # Add a small margin
    mjd_margin = (max_mjd - min_mjd) * 0.05
    mjd_range = (min_mjd - mjd_margin, max_mjd + mjd_margin)

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

    # Set the x-axis range to match the light curve data
    ax1.set_xlim(mjd_range)

    # Plot anomaly probabilities using the filtered values within the light curve's time range
    ax2.plot(
        anom_mjd,
        norm_score,
        drawstyle="steps",
        label=r"$p(Normal)$",
    )
    ax2.plot(
        anom_mjd,
        anom_score,
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

    return anom_mjd, anom_score, norm_score
