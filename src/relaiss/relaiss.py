from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import annoy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------- helper imports (internal low‑level funcs) ------------------
from .search import primer

REFERENCE_DIR = Path(__file__).with_suffix("").parent / "reference"

class ReLAISS:
    """A minimal, user‑facing wrapper around the full reLAISS tool‑chain."""

    def __init__(
        self,
        bank_csv: Path | str,
        index_stem: Path | str,
        scaler: StandardScaler,
        pca: Optional[PCA],
        lc_features: list[str],
        host_features: list[str],
    ) -> None:
        self.bank_csv = Path(bank_csv)
        self.index_stem = Path(index_stem)
        self.scaler = scaler
        self.pca = pca
        self.lc_features = lc_features
        self.host_features = host_features

        self._index = annoy.AnnoyIndex(  # loaded lazily on first query
            self.pca.n_components_ if self.pca else len(lc_features + host_features),
            metric="manhattan",
        )
        self._index.load(str(self.index_stem) + ".ann")
        self._ids = np.load(str(self.index_stem) + "_idx_arr.npy", allow_pickle=True)

    # ---------------------------------------------------------------------
    # Public helper constructors
    # ---------------------------------------------------------------------
    @classmethod
    def load_reference(
        cls,
        *,
        bank_path: Path | str = REFERENCE_DIR / "reference_20k.csv",
        lc_features: Optional[Sequence[str]] = None,
        host_features: Optional[Sequence[str]] = None,
        weight_lc: float = 1.0,
        use_pca: bool = False,
        n_components: Optional[int] = None,
    ) -> ReLAISS:
        """Load the shipped 20‑k reference bank and build (or load) its ANNOY index.

        Parameters
        ----------
        bank_path : str or Path
            CSV containing hydrated features.
        lc_features, host_features : sequence of str or *None*
            Columns to include; defaults to constants in `constants`.
        weight_lc : float, default 1.0
            Up‑weight factor for LC features (ignored when *use_pca* is True).
        use_pca : bool, default False
            Project to PCA space before indexing.
        n_components : int | None
            PCA dimensionality; *None* keeps 99 % variance.
        """
        import constants as _c

        lc_features = list(lc_features) if lc_features else _c.lc_features_const.copy()
        host_features = list(host_features) if host_features else _c.raw_host_features_const.copy()

        # build or reuse index
        index_stem = re_build_indexed_sample(
            dataset_bank_path=bank_path,
            lc_features=lc_features,
            host_features=host_features,
            use_pca=use_pca,
            n_components=n_components,
            num_trees=1000,
            path_to_index_directory=str(REFERENCE_DIR),
            save=True,
            force_recreation_of_index=False,
            weight_lc_feats_factor=weight_lc,
        )

        # for transform during queries
        scaler = StandardScaler().fit(pd.read_csv(bank_path)[lc_features + host_features])
        pca = None
        if use_pca:
            pca = PCA(n_components=n_components or 0.99, svd_solver="full").fit(
                scaler.transform(pd.read_csv(bank_path)[lc_features + host_features])
            )
        return cls(bank_path, index_stem, scaler, pca, lc_features, host_features)

    def find_neighbors(
        self,
        primer_dict,
        theorized_lightcurve_df,
        path_to_dataset_bank,
        annoy_index_file_stem,
        use_pca=False,
        num_pca_components=15,
        n=8,
        suggest_neighbor_num=False,
        max_neighbor_dist=None,
        search_k=1000,
        weight_lc_feats_factor=1,
        save_figures=True,
        path_to_figure_directory="../figures",
    ):
        """Query the ANNOY index and plot nearest-neighbor diagnostics.

        Parameters
        ----------
        primer_dict : dict
            Output from :func:`re_LAISS_primer`.
        annoy_index_file_stem : str
            Stem path returned by :func:`re_build_indexed_sample`.
        use_pca, num_pca_components : see above
        n : int, default 8
            Number of neighbours to return.
        suggest_neighbor_num : bool, default False
            If True, plots the distance elbow and exits early.
        max_neighbor_dist : float | None
            Optional cut on L1 distance.
        search_k : int, default 1000
            ANNOY *search_k* parameter.
        weight_lc_feats_factor : float, default 1
            Same interpretation as in ``re_build_indexed_sample``.
        save_figures : bool, default True
            Write LC + host plots and distance-elbow PNGs.
        path_to_figure_directory : str | Path

        Returns
        -------
        pandas.DataFrame | None
            Table summarising neighbours (or *None* if *suggest_neighbor_num=True*).
        """

        start_time = time.time()
        index_file = annoy_index_file_stem + ".ann"

        if n is None or n <= 0:
            raise ValueError("Neighbor number must be a nonzero integer. Abort!")

        plot_label = (
            f"{primer_dict['lc_ztf_id'] if primer_dict['lc_ztf_id'] is not None else 'theorized_lc'}"
            + (
                f"_host_from_{primer_dict['host_ztf_id']}"
                if primer_dict["host_ztf_id"] is not None
                else ""
            )
        )

        # Find neighbors for every Monte Carlo feature array
        scaler = preprocessing.StandardScaler()
        if use_pca:
            print(
                f"Loading previously saved ANNOY PCA={num_pca_components} index:",
                index_file,
                "\n",
            )
        else:
            print("Loading previously saved ANNOY index without PCA:", index_file, "\n")

        bank_feat_arr = np.load(
            annoy_index_file_stem + "_feat_arr.npy",
            allow_pickle=True,
        )
        trained_PCA_feat_arr_scaled = scaler.fit_transform(bank_feat_arr)

        true_and_mc_feat_arrs_l = [primer_dict["locus_feat_arr"]] + primer_dict[
            "locus_feat_arrs_mc_l"
        ]

        neighbor_dist_dict = {}
        if len(primer_dict["locus_feat_arrs_mc_l"]) != 0:
            print("Running Monte Carlo simulation to find possible neighbors...")
        for locus_feat_arr in true_and_mc_feat_arrs_l:
            # Scale locus_feat_arr using the same scaler (fit on dataset bank feature array)
            locus_feat_arr_scaled = scaler.transform([locus_feat_arr])

            if not use_pca:
                # Upweight lightcurve features
                num_lc_feats = len(constants.lc_features_const.copy())
                locus_feat_arr_scaled[:, :num_lc_feats] *= weight_lc_feats_factor

            if use_pca:
                # Transform the scaled locus_feat_arr using the same PCA model
                random_seed = 88
                pca = PCA(n_components=num_pca_components, random_state=random_seed)

                # pca needs to be fit first to the same data as trained
                _ = pca.fit(
                    trained_PCA_feat_arr_scaled
                )
                locus_feat_arr_pca = pca.transform(locus_feat_arr_scaled)

                index_dim = num_pca_components
                query_vector = locus_feat_arr_pca[0]

            else:
                index_dim = len(locus_feat_arr)
                query_vector = locus_feat_arr_scaled[0]

            # 3. Use the ANNOY index to find nearest neighbors (common to both branches)
            index = annoy.AnnoyIndex(index_dim, metric="manhattan")
            index.load(index_file)
            idx_arr = np.load(f"{annoy_index_file_stem}_idx_arr.npy", allow_pickle=True)

            ann_start_time = time.time()
            ann_indexes, ann_dists = index.get_nns_by_vector(
                query_vector, n=n, search_k=search_k, include_distances=True
            )

            # Store neighbors and distances in dictionary
            for ann_index, ann_dist in zip(ann_indexes, ann_dists):
                if ann_index in neighbor_dist_dict:
                    neighbor_dist_dict[ann_index].append(ann_dist)
                else:
                    neighbor_dist_dict[ann_index] = [ann_dist]

        # Pick n neighbors with lowest median distance
        if len(primer_dict["locus_feat_arrs_mc_l"]) != 0:
            print(
                f"Number of unique neighbors found through Monte Carlo: {len(neighbor_dist_dict)}.\nPicking top {n} neighbors."
            )

        medians = {idx: np.median(dists) for idx, dists in neighbor_dist_dict.items()}
        sorted_neighbors = sorted(medians.items(), key=lambda item: item[1])
        top_n_neighbors = sorted_neighbors[:n]

        ann_indexes = [idx for idx, _ in top_n_neighbors]
        ann_dists = [dist for _, dist in top_n_neighbors]

        for i in ann_indexes:
            if idx_arr[i] == primer_dict["lc_ztf_id"]:
                # drop input transient from ann_indexes and ann_dists
                idx_to_del = ann_indexes.index(i)
                del ann_indexes[idx_to_del]
                del ann_dists[idx_to_del]
                print(
                    "First neighbor is input transient, so it will be excluded. Final neighbor count will be one less than expected."
                )
                break

        ann_alerce_links = [
            f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
        ]
        ann_end_time = time.time()

        # Find optimal number of neighbors
        if suggest_neighbor_num:
            number_of_neighbors_found = len(ann_dists)
            neighbor_numbers_for_plot = list(range(1, number_of_neighbors_found + 1))

            knee = KneeLocator(
                neighbor_numbers_for_plot,
                ann_dists,
                curve="concave",
                direction="increasing",
            )
            optimal_n = knee.knee

            if optimal_n is None:
                print(
                    "Couldn't identify optimal number of neighbors. Try a larger neighbor pool."
                )
            else:
                print(
                    f"Suggested number of neighbors is {optimal_n}, chosen by comparing {n} neighbors."
                )

            plt.figure(figsize=(10, 4))
            plt.plot(
                neighbor_numbers_for_plot,
                ann_dists,
                marker="o",
                label="Distances",
            )
            if optimal_n:
                plt.axvline(
                    optimal_n,
                    color="red",
                    linestyle="--",
                    label=f"Elbow at {optimal_n}",
                )
            plt.xlabel("Neighbor Number")
            plt.ylabel("Distance")
            plt.title("Distance for Closest Neighbors")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if save_figures:
                os.makedirs(path_to_figure_directory, exist_ok=True)
                os.makedirs(
                    path_to_figure_directory + "/neighbor_dist_plots/", exist_ok=True
                )
                plt.savefig(
                    path_to_figure_directory
                    + f"/neighbor_dist_plots/{plot_label}_n={n}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                print(
                    f"Saved neighbor distances plot to {path_to_figure_directory}/neighbor_dist_plots/n={n}"
                )
            plt.show()

            print(
                "Stopping nearest neighbor search after suggesting neighbor number. Set run_NN=True and suggest_neighbor_num=False for full search.\n"
            )
            return

        # Filter neighbors for maximum distance, if provided
        if max_neighbor_dist is not None:
            filtered_neighbors = [
                (idx, dist)
                for idx, dist in zip(ann_indexes, ann_dists)
                if dist <= abs(max_neighbor_dist)
            ]
            ann_indexes, ann_dists = (
                zip(*filtered_neighbors) if filtered_neighbors else ([], [])
            )
            ann_indexes = list(ann_indexes)
            ann_dists = list(ann_dists)

            if len(ann_dists) == 0:
                raise ValueError(
                    f"No neighbors found for distance threshold of {abs(max_neighbor_dist)}. Try a larger maximum distance."
                )
            else:
                print(
                    f"Found {len(ann_dists)} neighbors for distance threshold of {abs(max_neighbor_dist)}."
                )

        # 4. Get TNS, spec. class of neighbors
        tns_ann_names, tns_ann_classes, tns_ann_zs, neighbor_ztfids = [], [], [], []
        ann_locus_l = []
        for i in ann_indexes:
            neighbor_ztfids.append(idx_arr[i])

            ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
            ann_locus_l.append(ann_locus)

            tns_ann_name, tns_ann_cls, tns_ann_z = get_TNS_data(idx_arr[i])

            tns_ann_names.append(tns_ann_name)
            tns_ann_classes.append(tns_ann_cls)
            tns_ann_zs.append(tns_ann_z)

        # Print the nearest neighbors and organize them for storage
        if primer_dict["lc_ztf_id"]:
            print("\t\t\t\t\t\t ZTFID     IAU_NAME SPEC  Z")
        else:
            print("\t\t\t\t\tIAU  SPEC  Z")
        print(
            f"Input transient: {'https://alerce.online/object/'+primer_dict['lc_ztf_id'] if primer_dict['lc_ztf_id'] else 'Theorized Lightcurve,'} {primer_dict['lc_tns_name']} {primer_dict['lc_tns_cls']} {primer_dict['lc_tns_z']}\n"
        )
        if primer_dict["host_ztf_id"] is not None:
            print("\t\t\t\t\t\t\t\t\tZTFID     IAU_NAME SPEC  Z")
            print(
                f"Transient with host swapped into input: https://alerce.online/object/{primer_dict['host_ztf_id']} {primer_dict['host_tns_name']} {primer_dict['host_tns_cls']} {primer_dict['host_tns_z']}\n"
            )

        # Plot lightcurves
        plot_lightcurves(
            primer_dict=primer_dict,
            plot_label=plot_label,
            theorized_lightcurve_df=theorized_lightcurve_df,
            neighbor_ztfids=neighbor_ztfids,
            ann_locus_l=ann_locus_l,
            ann_dists=ann_dists,
            tns_ann_names=tns_ann_names,
            tns_ann_classes=tns_ann_classes,
            tns_ann_zs=tns_ann_zs,
            figure_path=path_to_figure_directory,
            save_figures=save_figures,
        )

        # Plot hosts
        print("\nGenerating hosts grid plot...")

        df_bank = pd.read_csv(path_to_dataset_bank, index_col="ztf_object_id")

        hosts_to_plot = neighbor_ztfids.copy()
        host_ra_l, host_dec_l = [], []

        for ztfid in hosts_to_plot:
            host_ra, host_dec = (
                df_bank.loc[ztfid].host_ra,
                df_bank.loc[ztfid].host_dec,
            )
            host_ra_l.append(host_ra), host_dec_l.append(host_dec)

        # Add input host for plotting
        if primer_dict["host_ztf_id"] is None:
            hosts_to_plot.insert(0, primer_dict["lc_ztf_id"])
            host_ra_l.insert(0, primer_dict["lc_galaxy_ra"])
            host_dec_l.insert(0, primer_dict["lc_galaxy_dec"])
        else:
            hosts_to_plot.insert(0, primer_dict["host_ztf_id"])
            host_ra_l.insert(0, primer_dict["host_galaxy_ra"])
            host_dec_l.insert(0, primer_dict["host_galaxy_dec"])

        host_ann_df = pd.DataFrame(
            zip(hosts_to_plot, host_ra_l, host_dec_l),
            columns=["ZTFID", "HOST_RA", "HOST_DEC"],
        )

        plot_hosts(
            ztfid_ref=(
                primer_dict["lc_ztf_id"]
                if primer_dict["host_ztf_id"] is None
                else primer_dict["host_ztf_id"]
            ),
            plot_label=plot_label,
            df=host_ann_df,
            figure_path=path_to_figure_directory,
            ann_num=n,
            save_pdf=save_figures,
            imsizepix=100,
            change_contrast=False,
            prefer_color=True,
        )

        # Store neighbors and return
        storage = []
        neighbor_num = 1
        for al, iau_name, spec_cls, z, dist in zip(
            ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs, ann_dists
        ):
            print(f"ANN={neighbor_num}: {al} {iau_name} {spec_cls}, {z}")
            neighbor_dict = {
                "input_ztf_id": primer_dict["lc_ztf_id"],
                "input_swapped_host_ztf_id": primer_dict["host_ztf_id"],
                "neighbor_num": neighbor_num,
                "ztf_link": al,
                "dist": dist,
                "iau_name": iau_name,
                "spec_cls": spec_cls,
                "z": z,
            }
            storage.append(neighbor_dict)
            neighbor_num += 1

        end_time = time.time()
        ann_elapsed_time = ann_end_time - ann_start_time
        elapsed_time = end_time - start_time
        print(f"\nANN elapsed_time: {round(ann_elapsed_time, 3)} s")
        print(f"total elapsed_time: {round(elapsed_time, 3)} s\n")

        return pd.DataFrame(storage)
