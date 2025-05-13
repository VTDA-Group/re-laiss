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

    # ------------------------------------------------------------------
    # Main query
    # ------------------------------------------------------------------
    def find_neighbors(
        self,
        ztf_id: str,
        *,
        k: int = 8,
        use_lightcurve: bool = True,
        use_host: bool = True,
        host_weight: float = 0.3,
        return_dataframe: bool = True,
        search_k: int = 1000,
    ) -> pd.DataFrame | list[tuple[str, float]]:
        """Return *k* nearest neighbours to *ztf_id*.

        Parameters
        ----------
        ztf_id : str
            ZTF object identifier.
        k : int, default 8
            Number of neighbours.
        use_lightcurve, use_host : bool
            Toggle each feature subset.
        host_weight : float, default 0.3
            Relative weight when both subsets are used.
        return_dataframe : bool, default True
            Return `pd.DataFrame` (with distance column) instead of plain list.
        search_k : int, default 1000
            ANNOY *search_k* parameter (trade speed vs accuracy).
        """
        # ------------------------------------------------------------------
        # 1. Build query vector via existing feature‑extraction pipeline.
        primer_vec = primer(
            lc_ztf_id=ztf_id,
            theorized_lightcurve_df=None,
            host_ztf_id=None,
            dataset_bank_path=self.bank_csv,
            path_to_timeseries_folder="../timeseries",
            path_to_sfd_data_folder="../data/sfddata-master",
            lc_features=self.lc_features if use_lightcurve else [],
            host_features=self.host_features if use_host else [],
            num_sims=0,
            save_timeseries=False,
        )
        vec = primer_vec["locus_feat_arr"]
        vec = self.scaler.transform([vec])[0]
        if self.pca is not None:
            vec = self.pca.transform([vec])[0]
        else:
            # manual LC vs host weighting (simple scalar)
            n_lc = len(self.lc_features)
            if use_lightcurve and use_host:
                vec[:n_lc] *= (1 - host_weight)
                vec[n_lc:] *= host_weight
        # ------------------------------------------------------------------
        # 2. Query ANNOY
        idxs, dists = self._index.get_nns_by_vector(vec, k=k, search_k=search_k, include_distances=True)
        neighbor_ids = [self._ids[i] for i in idxs]

        if return_dataframe:
            return pd.DataFrame({"ztfid": neighbor_ids, "distance": dists})
        return list(zip(neighbor_ids, dists))
