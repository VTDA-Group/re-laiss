import logging
import math
import os
from pathlib import Path

import antares_client
import astropy.units as u
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, PercentileInterval
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels import robust

from .features import create_features_dict
from .fetch import fetch_ps1_cutout, fetch_ps1_rgb_jpeg


def plot_lightcurves(
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
    """Stack reference + neighbour light curves in a single figure.

    Parameters
    ----------
    primer_dict : dict
        Metadata for the reference transient (e.g., TNS name/class/redshift).
    plot_label : str
        Text used for figure title and filename.
    theorized_lightcurve_df : pandas.DataFrame | None
        Optional simulated LC to plot as the reference.
    neighbor_ztfids : list[str]
        ZTF IDs of retrieved neighbours (<= 8 plotted).
    ann_locus_l : list[antares_client.objects.Locus]
        Corresponding ANTARES loci holding photometry.
    ann_dists : list[float]
        ANN distances for labeling.
    tns_ann_names, tns_ann_classes, tns_ann_zs : list
        TNS metadata for neighbours.
    figure_path : str | Path
        Root folder to save PNGs in ``lightcurves/``.
    save_figures : bool, default True
        Write the PNG to disk.

    Returns
    -------
    None
    """
    print("Making a plot of stacked lightcurves...")

    if primer_dict["lc_tns_z"] is None:
        primer_dict["lc_tns_z"] = "None"
    elif isinstance(primer_dict["lc_tns_z"], float):
        primer_dict["lc_tns_z"] = round(primer_dict["lc_tns_z"], 3)
    else:
        primer_dict["lc_tns_z"] = primer_dict["lc_tns_z"]

    if primer_dict["lc_ztf_id"] is not None:
        ztf_id = primer_dict["lc_ztf_id"]
        ref_info = antares_client.search.get_by_ztf_object_id(
            ztf_id
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
        #try:
        if True:
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
                label=f"ANN={num+1}:{ztfname}, d={round(dist, 2)},\n{iau_name}, {spec_cls}, z={z}",
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

        #except Exception:
        #    print(f"Something went wrong with plotting {ztfname}! Excluding from plot.")

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


def plot_hosts(
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
    """Create 3×3 PS1 thumbnail grids for candidate host galaxies.

    This function creates a multi-page PDF containing PS1 thumbnail images of host
    galaxies for a set of transients. Each page contains a 3x3 grid of images, with
    the reference transient's host in the top-left position.

    Parameters
    ----------
    ztfid_ref : str
        ZTF ID of the reference transient (used in title only).
    plot_label : str
        Base name for the output PDF file.
    df : pandas.DataFrame
        DataFrame containing host galaxy information with columns:
        - ztf_object_id: ZTF ID of the transient
        - HOST_RA: Right ascension of the host galaxy
        - HOST_DEC: Declination of the host galaxy
    figure_path : str | Path
        Root directory for saving the PDF file.
    ann_num : int
        ANN neighbor index (used in filename).
    save_pdf : bool, default True
        Whether to save the PDF file.
    imsizepix : int, default 100
        Size of PS1 cutout images in pixels.
    change_contrast : bool, default False
        Whether to use a shallower stretch (93%) for grayscale images.
    prefer_color : bool, default True
        Whether to prefer RGB images over grayscale.

    Returns
    -------
    None

    Notes
    -----
    The output PDF will be saved as:
    {figure_path}/host_grids/{plot_label}_host_thumbnails_ann={ann_num}.pdf
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
                str(row["ztf_object_id"]),
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
                    except Exception:
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


def corner_plot(
    neighbors_df,  # from reLAISS nearest neighbors
    primer_dict,  # from reLAISS nearest neighbors
    path_to_dataset_bank,
    remove_outliers_bool=True,
    path_to_figure_directory="../figures",
    save_plots=True,
    preprocessed_df=None,  # Added parameter for preprocessed dataframe
):
    """Create corner plots comparing feature distributions between neighbors and the full dataset.

    This function creates corner plots that visualize the distribution of features
    for the nearest neighbors compared to the full dataset. The input transient's
    features are marked in green, neighbors in red, and the full dataset in blue.

    Parameters
    ----------
    neighbors_df : pandas.DataFrame
        DataFrame containing neighbor information from find_neighbors().
    primer_dict : dict
        Dictionary containing feature information for the input transient.
    path_to_dataset_bank : str | Path
        Path to the dataset bank CSV file.
    remove_outliers_bool : bool, default True
        Whether to remove outliers using robust MAD clipping.
    path_to_figure_directory : str | Path, default "../figures"
        Directory to save the corner plots.
    save_plots : bool, default True
        Whether to save the plots to disk.
    preprocessed_df : pandas.DataFrame | None, default None
        Optional preprocessed dataframe with imputed features to use instead of loading
        the raw dataset. This ensures no NaN values which could cause issues.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If primer_dict or neighbors_df is None.

    Notes
    -----
    The corner plots are saved as PNG files in:
    {path_to_figure_directory}/corner_plots/{batch_name}.png
    """
    if primer_dict is None:
        raise ValueError(
            "primer_dict is None. Try running NN search with reLAISS again."
        )
    if neighbors_df is None:
        raise ValueError(
            "neighbors_df is None. Try running reLAISS NN search again using run_NN=True, suggest_neighbor_num=False to get correct object."
        )

    lc_feature_names = primer_dict["lc_feat_names"]
    host_feature_names = primer_dict["host_feat_names"]

    if save_plots:
        os.makedirs(path_to_figure_directory, exist_ok=True)
        os.makedirs(path_to_figure_directory + "/corner_plots", exist_ok=True)

    logging.getLogger().setLevel(logging.ERROR)

    features_dict = create_features_dict(
        lc_feature_names, host_feature_names
    )

    neighbor_ztfids = [link.split("/")[-1] for link in neighbors_df["ztf_link"]]

    # Use preprocessed dataframe if provided, otherwise load from file
    if preprocessed_df is not None:
        print("Using provided preprocessed dataframe for corner plots")
        dataset_bank_df = preprocessed_df
        if 'ZTFID' in dataset_bank_df.columns:
            dataset_bank_df = dataset_bank_df.rename(columns={'ZTFID': 'ztf_object_id'})
    else:
        dataset_bank_df = pd.read_csv(path_to_dataset_bank, low_memory=False)
        if 'ZTFID' in dataset_bank_df.columns:
            dataset_bank_df = dataset_bank_df.rename(columns={'ZTFID': 'ztf_object_id'})

    dataset_bank_df = dataset_bank_df[
        ["ztf_object_id"] + lc_feature_names + host_feature_names
    ]
    print("Total number of transients for corner plots:", dataset_bank_df.shape[0])

    for batch_name, features in features_dict.items():
        print(f"Creating corner plot for {batch_name}...")

        # REMOVING OUTLIERS #
        def remove_outliers(df, threshold=7):
            df_clean = df.copy()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                col_data = df_clean[col]
                median_val = col_data.median()
                mad_val = robust.mad(
                    col_data
                )  # By default uses 0.6745 scale factor internally

                # If MAD is zero, it means the column has too little variation (or all same values).
                # In that case, skip it to avoid removing all rows.
                if mad_val == 0:
                    continue

                # Compute robust z-scores
                robust_z = 0.6745 * (col_data - median_val) / mad_val

                # Keep only points where the robust z-score is within the threshold
                df_clean = df_clean[abs(robust_z) <= threshold]

            return df_clean

        dataset_bank_df_batch_features = dataset_bank_df[["ztf_object_id"] + features]

        if remove_outliers_bool:
            dataset_bank_df_batch_features = remove_outliers(
                dataset_bank_df_batch_features
            )
            print(
                "Total number of transients for corner plot after outlier removal:",
                dataset_bank_df_batch_features.shape[0],
            )
        else:
            dataset_bank_df_batch_features = dataset_bank_df_batch_features.replace(
                [np.inf, -np.inf, -999], np.nan
            ).dropna()
            print(
                "Total number of transients for corner plot after NA, inf, and -999 removal:",
                dataset_bank_df_batch_features.shape[0],
            )
        # REMOVING OUTLIERS #
        neighbor_mask = dataset_bank_df_batch_features["ztf_object_id"].isin(
            neighbor_ztfids
        )
        features_df = dataset_bank_df_batch_features[features]

        # remove 'feature_' from column names
        features_df.columns = [
            col.replace("feature_", "", 1) if col.startswith("feature_") else col
            for col in features_df.columns
        ]

        neighbor_features = features_df[neighbor_mask]
        non_neighbor_features = features_df[~neighbor_mask]

        col_order = lc_feature_names + host_feature_names
        queried_transient_feat_df = pd.DataFrame(
            [primer_dict["locus_feat_arr"]], columns=col_order
        )
        queried_features_arr = queried_transient_feat_df[features].values[0]

        figure = corner.corner(
            non_neighbor_features,
            color="blue",
            labels=features_df.columns,
            plot_datapoints=True,
            alpha=0.3,
            plot_contours=False,
            truths=queried_features_arr,
            truth_color="green",
        )

        # Overlay neighbor features (red) with larger, visible markers
        axes = np.array(figure.axes).reshape(len(features), len(features))
        for i in range(len(features)):
            for j in range(i):  # Only the lower triangle of the plot
                ax = axes[i, j]
                ax.scatter(
                    neighbor_features.iloc[:, j],
                    neighbor_features.iloc[:, i],
                    color="red",
                    s=10,
                    marker="x",
                    linewidth=2,
                )

        if save_plots:
            plt.savefig(
                path_to_figure_directory + f"/corner_plots/{batch_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    if save_plots:
        print("Corner plots saved to" + path_to_figure_directory + "/corner_plots")
    else:
        print("Finished creating all corner plots!")
    return
