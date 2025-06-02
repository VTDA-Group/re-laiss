from __future__ import annotations

import gdown
from collections.abc import Sequence
from pathlib import Path
import time
from typing import Optional
from .index import build_indexed_sample
from .features import build_dataset_bank
from .search import primer
from .fetch import get_TNS_data
from .plotting import plot_lightcurves, plot_hosts
import os
from kneed import KneeLocator
import annoy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import antares_client
import matplotlib.pyplot as plt
import joblib
import requests
import shutil
from urllib.parse import urljoin
from .utils import (
    compute_dataframe_hash,
    get_cache_key,
    load_cached_dataframe,
    cache_dataframe,
    get_cache_dir,
    compress,
    _fmt_z, _fmt_txt
)

REFERENCE_DIR = Path(__file__).with_suffix("").parent / "reference"
SFD_URL = "https://github.com/kbarbary/sfddata/raw/refs/heads/master/"
SFD_FILES = ["SFD_dust_4096_ngp.fits", "SFD_dust_4096_sgp.fits"]

def download_sfd_files(path_to_sfd_folder):
    """Download SFD dust map files if they don't exist.

    This function downloads the necessary SFD dust map files from the GitHub repository
    if they are not already present in the specified directory. These files are required
    for extinction corrections in the reLAISS pipeline.

    Parameters
    ----------
    path_to_sfd_folder : str | Path
        Directory where SFD files should be stored. The function will create this
        directory if it doesn't exist.

    Notes
    -----
    Downloads two files:
    - SFD_dust_4096_ngp.fits
    - SFD_dust_4096_sgp.fits
    """
    path_to_sfd_folder = Path(path_to_sfd_folder)
    path_to_sfd_folder.mkdir(parents=True, exist_ok=True)

    for filename in SFD_FILES:
        filepath = path_to_sfd_folder / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            url = urljoin(SFD_URL, filename)
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print(f"Downloaded {filename} to {filepath}")

class ReLAISS:
    """A class for finding similar transients using the reLAISS algorithm.

    This class implements the reLAISS (Reference Lightcurve and Host Galaxy Similarity Search)
    algorithm for finding similar astronomical transients based on their lightcurves
    and host galaxy properties.

    Attributes
    ----------
    bank_csv : Path
        Path to the reference dataset bank CSV file.
    index_stem : Path
        Stem path for the ANNOY index files.
    scaler : StandardScaler
        Scaler used for feature normalization.
    pca : Optional[PCA]
        PCA model if dimensionality reduction is used.
    lc_features : list[str]
        List of lightcurve feature names.
    host_features : list[str]
        List of host galaxy feature names.
    feat_arr_scaled : np.ndarray
        Scaled feature array used for indexing.
    path_to_sfd_folder : Path
        Path to SFD dust map files.
    _index : annoy.AnnoyIndex
        ANNOY index for fast similarity search.
    _ids : np.ndarray
        Array of ZTF IDs corresponding to the index.
    use_pca : bool
        Whether PCA is being used for dimensionality reduction.
    hydrated_bank : pd.DataFrame
        The preprocessed dataframe with all features imputed and engineered.
    """

    def __init__(self) -> None:
        """Initialize a new ReLAISS instance.

        Creates a new ReLAISS instance with uninitialized attributes. These will be
        set when load_reference() is called.
        """
        self.bank_csv: Path
        self.index_stem: Path
        self.scaler: StandardScaler
        self.pca: Optional[PCA]
        self.lc_features: list[str]
        self.host_features: list[str]
        self.feat_arr_scaled: np.ndarray
        self.path_to_sfd_folder: Path
        self._index: annoy.AnnoyIndex
        self._ids: np.ndarray
        self.use_pca: bool
        self.built_for_AD: bool
        self.hydrated_bank: pd.DataFrame
        self.random_seed: int

    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        """Get the preprocessed dataframe with all features imputed and engineered.

        Returns
        -------
        pandas.DataFrame
            The preprocessed dataframe with all features imputed and engineered.

        Raises
        ------
        AttributeError
            If load_reference() hasn't been called yet.
        """
        if not hasattr(self, 'hydrated_bank'):
            raise AttributeError("No preprocessed dataframe available. Call load_reference() first.")
        return self.hydrated_bank.copy()

    def load_reference(
        self,
        *,
        bank_path: Path | str = REFERENCE_DIR / "reference_20k.csv",
        path_to_sfd_folder: Path | str = './' ,
        lc_features: Optional[Sequence[str]] = None,
        host_features: Optional[Sequence[str]] = None,
        weight_lc: float = 1.0,
        use_pca: bool = False,
        num_pca_components: Optional[int] = None,
        force_recreation_of_index: bool = False,
        building_for_AD: bool = False,
        random_seed: int = 42,
        num_trees: int = 1000,
    ) -> None:
        """Load the shipped 20‑k reference bank and build (or load) its ANNOY index.

        Parameters
        ----------
        bank_path : str or Path
            Path to CSV containing raw feature file.
        path_to_sfd_folder : str or Path
            Path to SFD dustmaps for extinction-correction.
        lc_features, host_features : sequence of str or *None*
            Columns to include; defaults to constants in `constants`.
        weight_lc : float, default 1.0
            Up‑weight factor for LC features (ignored when *use_pca* is True).
        use_pca : bool, default False
            Project to PCA space before indexing.
        num_pca_components : int | None
            PCA dimensionality; *None* keeps 99 % variance.
        force_recreation_of_index : bool, default False
            Whether to force recreation of the index.
        """
        from . import constants as _c

        # Download SFD files if they don't exist
        download_sfd_files(path_to_sfd_folder)

        lc_features = list(lc_features) if lc_features is not None else _c.lc_features_const.copy()
        host_features = list(host_features) if host_features is not None else _c.raw_host_features_const.copy()

        if not bank_path.exists():
            print(f"Reference data not found at {bank_path}; downloading now...")
            bank_path.parent.mkdir(parents=True, exist_ok=True)

            url = "https://drive.google.com/uc?export=download&id=1uH_03ju50Enb7ZhiduDrmCVTMvTc7bMC"
            gdown.download(url, str(bank_path), quiet=False)

        raw_df_bank = pd.read_csv(bank_path, low_memory=False)

        # Try to load preprocessed data from cache
        cache_params_bank = {
            'bank_hash'        : compute_dataframe_hash(raw_df_bank),
            'path_to_sfd'      : str(path_to_sfd_folder),
            'lc_features'      : compress(lc_features),
            'host_features'    : compress(host_features),
            'weight_lc'        : weight_lc,
            'use_pca'          : use_pca,
            'num_pca_components': num_pca_components,
            'building_for_AD'  : building_for_AD,
        }
        cache_key_bank = get_cache_key('reference_bank', **cache_params_bank)

        hydrated_bank = load_cached_dataframe(cache_key_bank)
        if hydrated_bank is None:
            print("Preprocessing reference bank...")

            # Rename ZTFID to ztf_object_id if it exists
            if 'ZTFID' in raw_df_bank.columns:
                raw_df_bank = raw_df_bank.rename(columns={'ZTFID': 'ztf_object_id'})

            hydrated_bank = build_dataset_bank(
                raw_df_bank,
                building_entire_df_bank=True,
                path_to_sfd_folder=path_to_sfd_folder,
                building_for_AD=building_for_AD,
            )

            if 'ZTFID' in hydrated_bank.columns:
                hydrated_bank = hydrated_bank.rename(columns={'ZTFID': 'ztf_object_id'})
            elif 'ztf_object_id' not in hydrated_bank.columns:
                print("Warning: Could not find 'ztf_object_id' in columns. Some queries may fail.")

            # Cache the preprocessed data
            print("Caching preprocessed reference bank...")
            cache_dataframe(hydrated_bank, cache_key_bank)
        else:
            print("Loading preprocessed reference bank from cache...")

        # Store the hydrated bank as an attribute
        self.hydrated_bank = hydrated_bank
        self.built_for_AD = building_for_AD

        index_dir = Path(get_cache_dir()) / 'indices'
        index_dir.mkdir(exist_ok=True)

        index_params = {
            **cache_params_bank,
            'hydrated_hash': compute_dataframe_hash(hydrated_bank),
            'num_trees'    : num_trees,          # fixed in your code
        }
        index_cache_key = get_cache_key('reference_index', **index_params)
        index_stem = index_dir / index_cache_key

        # Check if index files exist
        index_files_exist = all(
            path.exists()
            for path in (
                index_stem.with_suffix('.ann'),
                index_stem.parent / (index_stem.name + '_idx_arr.npy'),
                index_stem.parent / (index_stem.name + '_scaler.joblib'),
            )
        )

        if not index_files_exist:
            print("Building search index...")
            index_stem, scaler, feat_arr_scaled = build_indexed_sample(
                data_bank=hydrated_bank,
                lc_features=lc_features,
                host_features=host_features,
                use_pca=use_pca,
                num_pca_components=num_pca_components,
                num_trees=num_trees,
                path_to_index_directory=str(index_dir),
                weight_lc_feats_factor=weight_lc,
                force_recreation_of_index=force_recreation_of_index,
                random_seed=random_seed
            )
        else:
            print("Loading existing search index...")
            scaler = joblib.load(str(index_stem) + "_scaler.joblib")
            feat_arr_scaled = np.load(str(index_stem) + "_feat_arr_scaled.npy")

        pca = None
        if use_pca:
            feat_arr_scaled = np.load(
                str(index_stem) + ("_feat_arr_scaled_pca.npy" )
            )
            pca_path = str(index_stem) + "_pca.joblib"
            if os.path.exists(pca_path):
                print(f"Loading saved PCA model from {pca_path}")
                pca = joblib.load(pca_path)
            else:
                print("Training new PCA model...")
                pca = PCA(n_components=num_pca_components).fit(
                    scaler.fit_transform(hydrated_bank[lc_features + host_features])
                )
                joblib.dump(pca, pca_path)

        self.index_stem = index_stem
        self.scaler = scaler
        self.pca = pca
        self.use_pca = bool(pca)
        self.lc_features = lc_features
        self.path_to_sfd_folder = path_to_sfd_folder
        self.host_features = host_features
        self.bank_csv = bank_path
        self.weight_lc = weight_lc
        self.random_seed = random_seed

        dim = pca.n_components_ if pca else len(lc_features + host_features)
        self._index = annoy.AnnoyIndex(dim, metric="manhattan")
        self._index.load(str(self.index_stem) + ".ann")
        self._ids = np.load(str(self.index_stem) + "_idx_arr.npy", allow_pickle=True)

    def find_neighbors(
        self,
        ztf_object_id=None,
        theorized_lightcurve_df=None,
        host_ztf_id=None,
        path_to_dataset_bank: str | Path | None = None,
        n=8,
        suggest_neighbor_num=False,
        max_neighbor_dist=None,
        search_k=5000,
        num_sims=0,
        plot=False,
        save_figures=False,
        path_to_figure_directory="../figures",
    ):
        """Query the ANNOY index and plot nearest-neighbor diagnostics.

        This method finds the most similar transients to the input transient in the
        reference dataset bank. It can use either a real transient (specified by
        ztf_object_id) or a theorized lightcurve.

        Parameters
        ----------
        ztf_object_id : str | None, default None
            ZTF ID of the transient to find neighbors for. If None, theorized_lightcurve_df must be provided.
        theorized_lightcurve_df : pandas.DataFrame | None, default None
            Optional simulated lightcurve to use instead of a real transient.
            Required if ztf_object_id is None.
        host_ztf_id : str | None, default None
            ZTF ID of the host galaxy to swap into the input transient.
        path_to_dataset_bank : str | Path | None, default None
            Path to the dataset bank CSV. If None, uses the bank loaded in load_reference().
        use_pca : bool, default False
            Whether to use PCA for dimensionality reduction.
        num_pca_components : int, default 20
            Number of PCA components to use if use_pca is True.
        n : int, default 8
            Number of neighbors to return.
        suggest_neighbor_num : bool, default False
            If True, plots the distance elbow and exits early.
        max_neighbor_dist : float | None, default None
            Optional maximum L1 distance for neighbors.
        search_k : int, default 5000
            ANNOY search_k parameter for controlling search accuracy.
        weight_lc_feats_factor : float, default 1.0
            Factor to up-weight lightcurve features relative to host features.
        num_sims : int, default 0
            Number of Monte Carlo simulations to perform.
        plot : bool, default False
            Whether to generate diagnostic plots.
        save_figures : bool, default False
            Whether to save the diagnostic plots to disk.
        path_to_figure_directory : str | Path, default "../figures"
            Directory to save figures in if save_figures is True.

        Returns
        -------
        pandas.DataFrame | None
            DataFrame containing neighbor information with columns:
            - input_ztf_id: ZTF ID of the input transient
            - input_swapped_host_ztf_id: ZTF ID of the host galaxy (if swapped)
            - neighbor_num: Index of the neighbor
            - ztf_link: Link to the neighbor in ALeRCE
            - dist: Distance to the neighbor
            - iau_name: IAU name of the neighbor
            - spec_cls: Spectral classification
            - z: Redshift
            Returns None if suggest_neighbor_num is True.

        Raises
        ------
        ValueError
            If n is None or <= 0, or if no neighbors are found within max_neighbor_dist.
        """
        start_time = time.time()

        annoy_index_file_stem = self.index_stem
        dataset_bank = Path(path_to_dataset_bank or self.bank_csv)

        primer_dict = primer(
            lc_ztf_id=ztf_object_id,
            theorized_lightcurve_df=theorized_lightcurve_df,
            host_ztf_id=host_ztf_id,
            dataset_bank_path=dataset_bank,
            path_to_timeseries_folder='./',
            path_to_sfd_folder=self.path_to_sfd_folder,
            lc_features=self.lc_features,
            host_features=self.host_features,
            num_sims=num_sims,
            save_timeseries=False,
            preprocessed_df=self.hydrated_bank,
        )

        start_time = time.time()
        index_file = str(annoy_index_file_stem) + ".ann"

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
        if self.use_pca:
            print(
                f"Loading previously saved ANNOY PCA={self.pca.n_components_} index:",
                index_file,
                "\n",
            )

            bank_feat_arr_scaled = np.load(str(self.index_stem) + "_feat_arr_scaled_pca.npy", allow_pickle=True)

        else:
            print("Loading previously saved ANNOY index without PCA:", index_file, "\n")

            # Scale the feature array
            bank_feat_arr = np.load(
                str(self.index_stem) + "_feat_arr.npy",
                allow_pickle=True,
            )

            bank_feat_arr_scaled = self.scaler.transform(bank_feat_arr)

        # Process all feature arrays (true + MC)
        true_and_mc_feat_arrs_l = [primer_dict["locus_feat_arr"]] + primer_dict["locus_feat_arrs_mc_l"]

        neighbor_dist_dict = {}

        for i, locus_feat_arr in enumerate(true_and_mc_feat_arrs_l):
            scaled = self.scaler.transform([locus_feat_arr])  # shape (1, N)

            if self.use_pca:
                scaled = self.pca.transform(scaled)  # shape (1, 5)
                scaled_vector = scaled[0]            # shape (5,)
            else:
                scaled_vector = scaled[0]
            if np.abs(self.weight_lc - 1.0) > 0.1:
                if self.use_pca:
                    raise ValueError("Error! Cannot weight light curve features when PCA=True.")
                else:
                    if i==0:
                        print(f"Upweighting light curve features by a factor of {self.weight_lc:.1f}.")
                    scaled *= self.weight_lc

            idxs, dists = self._index.get_nns_by_vector(
                scaled_vector, n=n+100, search_k=search_k, include_distances=True
            )

            # Store neighbors and distances in dictionary
            for idx, dist in zip(idxs, dists):
                if idx in neighbor_dist_dict:
                    neighbor_dist_dict[idx].append(dist)
                else:
                    neighbor_dist_dict[idx] = [dist]

        # Pick n neighbors with lowest median distance
        if len(primer_dict["locus_feat_arrs_mc_l"]) != 0:
            print(f"\nNumber of unique neighbors found through monte carlo: {len(neighbor_dist_dict)}.")
            print(f"Picking top {n} neighbors.")

        medians = {idx: np.median(dists) for idx, dists in neighbor_dist_dict.items()}
        sorted_neighbors = sorted(medians.items(), key=lambda item: item[1])
        top_n_neighbors = sorted_neighbors[:n+1]
        idxs = [idx for idx, _ in top_n_neighbors]
        dists = [dist for _, dist in top_n_neighbors]

        # Remove input transient if it's in the results
        input_idx = None
        for i, idx in enumerate(idxs):
            if self._ids[idx] == primer_dict["lc_ztf_id"]:
                input_idx = i
                break

        if input_idx is not None:
            print(f"\nFound input transient at index {input_idx}, removing it...")
            del idxs[input_idx]
            del dists[input_idx]

        # Always return n neighbors
        idxs = idxs[:n]
        dists = dists[:n]

        ann_end_time = time.time()
        ann_elapsed_time = ann_end_time - start_time
        elapsed_time = time.time() - start_time
        print(f"\nANN elapsed time: {round(ann_elapsed_time, 3)} s")
        print(f"Total elapsed time: {round(elapsed_time, 3)} s\n")

        # Find optimal number of neighbors
        if suggest_neighbor_num:
            number_of_neighbors_found = len(dists)
            neighbor_numbers_for_plot = list(range(1, number_of_neighbors_found + 1))

            knee = KneeLocator(
                neighbor_numbers_for_plot,
                dists,
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

            if plot:
                plt.figure(figsize=(10, 4))
                plt.plot(
                    neighbor_numbers_for_plot,
                    dists,
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
                "Stopping nearest neighbor search after suggesting neighbor number. Set suggest_neighbor_num=False for full search.\n"
            )
            return

        # Filter neighbors for maximum distance, if provided
        if max_neighbor_dist is not None:
            filtered_neighbors = [
                (idx, dist)
                for idx, dist in zip(idxs, dists)
                if dist <= abs(max_neighbor_dist)
            ]
            idxs, dists = (
                zip(*filtered_neighbors) if filtered_neighbors else ([], [])
            )
            idxs = list(idxs)
            dists = list(dists)

            if len(dists) == 0:
                raise ValueError(
                    f"No neighbors found for distance threshold of {abs(max_neighbor_dist)}. Try a larger maximum distance."
                )
            else:
                print(
                    f"Found {len(dists)} neighbors for distance threshold of {abs(max_neighbor_dist)}."
                )

        # 4. Get TNS, spec. class of neighbors
        tns_ann_names, tns_ann_classes, tns_ann_zs, neighbor_ztfids = [], [], [], []
        ann_locus_l = []
        for i in idxs:
            neighbor_ztfids.append(self._ids[i])

            ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=self._ids[i])
            ann_locus_l.append(ann_locus)

            tns_ann_name, tns_ann_cls, tns_ann_z = get_TNS_data(self._ids[i])

            tns_ann_names.append(tns_ann_name)
            tns_ann_classes.append(tns_ann_cls)
            tns_ann_zs.append(tns_ann_z)

        # Print the nearest neighbors and organize them for storage
        hdr_fmt = "{:<8} {:<13} {:<10} {:<8} {:>7}"       # tag, ZTFID, IAU, class, z
        print(hdr_fmt.format("", "ZTF ID", "IAU Name", "Spec", "z"))

        # input object (real or theoretical)
        if primer_dict["lc_ztf_id"]:
            print(
                hdr_fmt.format(
                    "INPUT:",
                    primer_dict["lc_ztf_id"],
                    primer_dict["lc_tns_name"],
                    primer_dict["lc_tns_cls"],
                    f"{primer_dict['lc_tns_z']:.3f}",
                )
            )
        else:   # theorized LC
            print(
                hdr_fmt.format(
                    "INPUT:",
                    "[model]",
                    primer_dict["lc_tns_name"],
                    primer_dict["lc_tns_cls"],
                    f"{primer_dict['lc_tns_z']:.3f}",
                )
            )

        # optional swapped host line
        if primer_dict["host_ztf_id"] is not None:
            print(
                hdr_fmt.format(
                    "HOST:",
                    primer_dict["host_ztf_id"],
                    primer_dict["host_tns_name"],
                    primer_dict["host_tns_cls"],
                    f"{primer_dict['host_tns_z']:.3f}",
                )
            )

        print()  # blank line before the ANN table


        if plot:
            # Plot lightcurves
            plot_lightcurves(
                primer_dict=primer_dict,
                plot_label=plot_label,
                theorized_lightcurve_df=theorized_lightcurve_df,
                neighbor_ztfids=neighbor_ztfids,
                ann_locus_l=ann_locus_l,
                ann_dists=dists,
                tns_ann_names=tns_ann_names,
                tns_ann_classes=tns_ann_classes,
                tns_ann_zs=tns_ann_zs,
                figure_path=path_to_figure_directory,
                save_figures=save_figures,
            )

            # Plot hosts
            print("\nGenerating hosts grid plot...")

            # Get preprocessed dataframe for host data
            if hasattr(self, 'hydrated_bank'):
                df_bank = self.hydrated_bank.copy()
            else:
                # Fallback to reading from CSV if hydrated_bank not available
                df_bank = pd.read_csv(dataset_bank, low_memory=False)

            df_bank.set_index('ztf_object_id', drop=False, inplace=True)

            hosts_to_plot = neighbor_ztfids.copy()
            host_ra_l, host_dec_l = [], []

            for ztfid in hosts_to_plot:
                try:
                    # Try both possible column name variations
                    if 'host_ra' in df_bank.columns and 'host_dec' in df_bank.columns:
                        host_ra, host_dec = (
                            df_bank.loc[ztfid].host_ra,
                            df_bank.loc[ztfid].host_dec,
                        )
                    elif 'raMean' in df_bank.columns and 'decMean' in df_bank.columns:
                        host_ra, host_dec = (
                            df_bank.loc[ztfid].raMean,
                            df_bank.loc[ztfid].decMean,
                        )
                    else:
                        print(f"Warning: Could not find host coordinates for {ztfid}")
                        continue

                    host_ra_l.append(host_ra)
                    host_dec_l.append(host_dec)
                except KeyError:
                    print(f"Warning: Could not find host data for {ztfid}")
                    hosts_to_plot.remove(ztfid)
                    continue

            if not hosts_to_plot:
                print("No valid hosts found for plotting")
                return

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
                columns=["ztf_object_id", "HOST_RA", "HOST_DEC"],
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

        # Define ALeRCE links for each neighbor
        ann_alerce_links = [f"https://alerce.online/object/{ztf_id}" for ztf_id in neighbor_ztfids]

        print(f"{'ANN#':<4} {'ZTF ID':<13} {'IAU Name':<8} {'Spec':<10} {'z':>7} {'Dist':>6}  Link")

        storage = []
        for nn, (ztfid, link, dist, iau, cls, z) in enumerate(
                zip(neighbor_ztfids,
                    ann_alerce_links,
                    dists,
                    tns_ann_names,
                    tns_ann_classes,
                    tns_ann_zs),
                1):

            print(f"{nn:<4} {ztfid:<13} "
                  f"{_fmt_txt(iau, 8)} {_fmt_txt(cls, 10)} "
                  f"{_fmt_z(z)} {dist:6.3f}  {link}")

            storage.append({
                "input_ztf_id"            : primer_dict["lc_ztf_id"],
                "neighbor_ztf_id"         : ztfid,
                "input_swapped_host_ztf_id": primer_dict["host_ztf_id"],
                "neighbor_num"            : nn,
                "ztf_link"                : link,
                "dist"                    : dist,
                "iau_name"                : iau or "---",
                "spec_cls"                : cls or "---",
                "z"                       : z,
            })

        return pd.DataFrame(storage)
