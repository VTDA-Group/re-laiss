import os 
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest

import antares_client                    
from .fetch import get_timeseries_df, get_TNS_data

def train_AD_model(
    lc_features,
    host_features,
    path_to_dataset_bank,
    path_to_models_directory="../models",
    n_estimators=500,
    contamination=0.02,
    max_samples=1024,
    force_retrain=False,
):
    """Train or load an Isolation-Forest anomaly-detection model.

    Parameters
    ----------
    lc_features, host_features : list[str]
        Feature columns used by the model.
    path_to_dataset_bank : str | Path
    path_to_models_directory : str | Path
    n_estimators, contamination, max_samples : see *pyod.models.IForest*
    force_retrain : bool, default False
        Ignore cached model and retrain.

    Returns
    -------
    str
        Filesystem path to the saved ``.pkl`` pipeline.
    """
    feature_names = lc_features + host_features
    df_bank_path = path_to_dataset_bank
    model_dir = path_to_models_directory
    model_name = f"IForest_n{n_estimators}_c{contamination}_ms{max_samples}_lc{len(lc_features)}_host{len(host_features)}.pkl"

    os.makedirs(model_dir, exist_ok=True)

    print("Checking if AD model exists...")

    # If model already exists, don't retrain
    if os.path.exists(os.path.join(model_dir, model_name)) and not force_retrain:
        print("Model already exists →", os.path.join(model_dir, model_name))
        return os.path.join(model_dir, model_name)

    print("AD model does not exist. Training and saving new model.")

    # Train model
    df = pd.read_csv(df_bank_path, low_memory=False)
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
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, model_name), "wb") as f:
        pickle.dump(pipeline, f)

    print(
        "Isolation Forest model trained and saved →",
        os.path.join(model_dir, model_name),
    )

    return os.path.join(model_dir, model_name)


def anomaly_detection(
    transient_ztf_id,
    lc_features,
    host_features,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
    path_to_dataset_bank,
    host_ztf_id_to_swap_in=None,
    path_to_models_directory="../models",
    path_to_figure_directory="../figures",
    save_figures=True,
    n_estimators=500,
    contamination=0.02,
    max_samples=1024,
    force_retrain=False,
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

    Returns
    -------
    None
    """

    print("Running Anomaly Detection:\n")

    # Train the model (if necessary)
    path_to_trained_model = train_AD_model(
        lc_features,
        host_features,
        path_to_dataset_bank,
        path_to_models_directory=path_to_models_directory,
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        force_retrain=force_retrain,
    )

    # Load the model
    with open(path_to_trained_model, "rb") as f:
        clf = pickle.load(f)

    # Load the timeseries dataframe
    print("\nRebuilding timeseries dataframe(s) for AD...")
    timeseries_df = get_timeseries_df(
        ztf_id=transient_ztf_id,
        theorized_lightcurve_df=None,
        path_to_timeseries_folder=path_to_timeseries_folder,
        path_to_sfd_data_folder=path_to_sfd_data_folder,
        path_to_dataset_bank=path_to_dataset_bank,
        save_timeseries=False,
        building_for_AD=True,
    )

    if host_ztf_id_to_swap_in is not None:
        # Swap in the host galaxy
        swapped_host_timeseries_df = get_timeseries_df(
            ztf_id=host_ztf_id_to_swap_in,
            theorized_lightcurve_df=None,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            save_timeseries=False,
            building_for_AD=True,
            swapped_host=True,
        )

        host_values = swapped_host_timeseries_df[host_features].iloc[0]
        for col in host_features:
            timeseries_df[col] = host_values[col]

    timeseries_df_filt_feats = timeseries_df[lc_features + host_features]
    input_lightcurve_locus = antares_client.search.get_by_ztf_object_id(
        ztf_object_id=transient_ztf_id
    )

    tns_name, tns_cls, tns_z = get_TNS_data(transient_ztf_id)

    check_anom_and_plot(
        clf=clf,
        input_ztf_id=transient_ztf_id,
        swapped_host_ztf_id=host_ztf_id_to_swap_in,
        input_spec_cls=tns_cls,
        input_spec_z=tns_z,
        anom_thresh=50,
        timeseries_df_full=timeseries_df,
        timeseries_df_features_only=timeseries_df_filt_feats,
        ref_info=input_lightcurve_locus,
        savefig=save_figures,
        figure_path=path_to_figure_directory,
    )
    return


def check_anom_and_plot(
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
        Probability (%) above which an epoch is flagged anomalous.
    timeseries_df_full : pandas.DataFrame
        Hydrated LC + host features, including ``obs_num`` and ``mjd_cutoff``.
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
