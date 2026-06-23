"""Utilitário compartilhado para amostrar features tabulares na inferência."""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def _sample_patch_stats(
    tif_path: Path,
    cx: np.ndarray,
    cy: np.ndarray,
    window_px: int,
    stats: list,
    nodata: float,
) -> dict:
    """Amostra estatísticas focais em uma janela quadrada centrada no patch."""
    import rasterio
    from scipy.ndimage import uniform_filter, maximum_filter

    result = {s: np.full(len(cx), np.nan, dtype=np.float32) for s in stats}
    if not Path(tif_path).exists():
        return result

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        invalid = (arr == nodata) | ~np.isfinite(arr)
        valid = (~invalid).astype(np.float32)
        clean = np.where(invalid, 0.0, arr).astype(np.float32)

        size = max(1, int(window_px))
        focal = {}
        if "mean" in stats or "std" in stats:
            count = uniform_filter(valid, size=size, mode="constant", cval=0.0) * (size * size)
            summed = uniform_filter(clean, size=size, mode="constant", cval=0.0) * (size * size)
            mean = np.divide(summed, count, out=np.full_like(summed, np.nan), where=count > 0)
            focal["mean"] = mean.astype(np.float32)
            if "std" in stats:
                summed2 = uniform_filter(clean * clean, size=size, mode="constant", cval=0.0) * (size * size)
                mean2 = np.divide(summed2, count, out=np.full_like(summed2, np.nan), where=count > 0)
                focal["std"] = np.sqrt(np.maximum(mean2 - mean * mean, 0.0)).astype(np.float32)
        if "max" in stats:
            max_in = np.where(invalid, -np.inf, arr)
            mx = maximum_filter(max_in, size=size, mode="constant", cval=-np.inf)
            mx[~np.isfinite(mx)] = np.nan
            focal["max"] = mx.astype(np.float32)

        rows, cols = rasterio.transform.rowcol(src.transform, cx, cy)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        for stat_name, stat_arr in focal.items():
            vals = result[stat_name]
            vals[inside] = stat_arr[rows[inside], cols[inside]]

    return result


def build_tabular_for_inference(
    cx: np.ndarray,
    cy: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """Amostra rasters de features nas coordenadas dos patches para inferência."""
    import rasterio

    raster_dir = Path(config["paths"]["rasters"])
    nodata = config["output"]["nodata"]
    feat_dict = {}
    train_cfg = config.get("training", {})
    use_geochem = train_cfg.get("use_geochem_features", False)
    use_segemar = train_cfg.get("use_segemar_features", True)
    use_terrain = train_cfg.get("use_terrain_features", True)
    use_s2_indices = train_cfg.get("use_s2_index_features", True)
    patch_stats = train_cfg.get("tabular_patch_stats", ["mean", "std", "max"])
    window_px = int(config["prithvi"]["patch_size"])

    def _sample(tif_path):
        if not Path(tif_path).exists():
            return np.full(len(cx), np.nan, dtype=np.float32)
        with rasterio.open(tif_path) as src:
            coords = list(zip(cx, cy))
            try:
                vals = np.array(list(src.sample(coords)), dtype=np.float32)
                if vals.ndim == 2:
                    vals = vals.mean(axis=1)
                return np.where(vals == nodata, np.nan, vals)
            except Exception:
                return np.full(len(cx), np.nan, dtype=np.float32)

    # Geoquímica
    geochem_dir = raster_dir / "segemar" / "geochem"
    if use_geochem and geochem_dir.exists():
        for tif in sorted(geochem_dir.glob("*.tif")):
            feat_dict[f"geo_{tif.stem.replace('geochem_', '')}"] = _sample(tif)

    # SEGEMAR vetorial rasterizado
    segemar_dir = raster_dir / "segemar"
    if use_segemar:
        for tif in sorted(segemar_dir.glob("*.tif")):
            feat_dict[f"seg_{tif.stem}"] = _sample(tif)

    # DEM terrain — acumular por lista para evitar NaN com +=
    terrain_dir = raster_dir / "terrain"
    if use_terrain and terrain_dir.exists():
        terrain_accum: dict = {}
        for tif in sorted(terrain_dir.rglob("*.tif")):
            name = tif.stem.split("_")[0]
            stats_dict = _sample_patch_stats(tif, cx, cy, window_px, patch_stats, nodata)
            for stat_name, values in stats_dict.items():
                key = f"dem_{name}_{stat_name}"
                if key not in terrain_accum:
                    terrain_accum[key] = []
                terrain_accum[key].append(values)
        for key, arrs in terrain_accum.items():
            stacked = np.vstack(arrs)
            feat_dict[key] = np.nanmean(stacked, axis=0).astype(np.float32)

    # Índices S2 — acumular por lista
    s2_idx_dir = raster_dir / "indices" / "s2"
    if use_s2_indices and s2_idx_dir.exists():
        idx_accum: dict = {}
        for tif in sorted(s2_idx_dir.rglob("*.tif")):
            name = "_".join(tif.stem.split("_")[:2])
            stats_dict = _sample_patch_stats(tif, cx, cy, window_px, patch_stats, nodata)
            for stat_name, values in stats_dict.items():
                key = f"idx_{name}_{stat_name}"
                if key not in idx_accum:
                    idx_accum[key] = []
                idx_accum[key].append(values)
        for key, arrs in idx_accum.items():
            stacked = np.vstack(arrs)
            feat_dict[key] = np.nanmean(stacked, axis=0).astype(np.float32)

    return pd.DataFrame(feat_dict)


def align_features_to_model(
    embeddings: np.ndarray,
    tabular_df: pd.DataFrame,
    feature_names_path: Path,
    n_emb: int = 768,
) -> np.ndarray:
    """
    Alinha a matrix de features da inferência com as colunas usadas no treino.
    Colunas ausentes → 0.0; colunas extras → descartadas.
    """
    if not feature_names_path.exists():
        # Fallback: concatena na ordem atual
        return np.hstack([embeddings, tabular_df.fillna(0).values])

    with open(feature_names_path) as f:
        expected_names = json.load(f)

    emb_names = [n for n in expected_names if n.startswith("emb_")]
    tab_names = [n for n in expected_names if not n.startswith("emb_")]

    # Embeddings sempre presentes com dimensão correta
    emb_part = embeddings[:, :len(emb_names)]

    # Tabular: reindexar para as colunas esperadas (faltantes → 0)
    tab_aligned = tabular_df.reindex(columns=tab_names, fill_value=0.0).fillna(0.0)

    return np.hstack([emb_part, tab_aligned.values]).astype(np.float32)
