"""
Fase 6a · Inferência — Geração de Mapas de Favorabilidade.

Para cada tile, aplica os modelos treinados sobre todos os patches
e reconstrói o raster de favorabilidade 30m (probabilidade 0-1).

Saída por tipo de depósito: data/08_OUTPUT/geotiff/<type>/<tile>.tif
"""

import sys
import gc
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.raster_utils import load_config, save_raster, get_profile
from utils.gpu_utils import get_device, clear_gpu_cache
from utils.tiling import TileGrid
from utils.raster_utils import load_tile_geometries

logger.remove()
logger.add(sys.stderr, level="INFO", format="<cyan>{time:HH:mm:ss}</cyan> | <level>{message}</level>")
logger.add(Path("D:/argentina/logs/inference_{time:YYYY-MM-DD}.log"), level="DEBUG", rotation="50 MB")

PATCH_TYPES = ["porfiro", "skarn", "epitermal", "manto"]


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def build_feature_vector_from_npz(
    npz_path: Path,
    tabular_features: pd.DataFrame,
    n_emb: int = 768,
) -> np.ndarray:
    """Reconstrói matrix de features para inferência."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    tabular = tabular_features.fillna(0).values
    if len(embeddings) != len(tabular):
        logger.warning(f"Mismatch embeddings ({len(embeddings)}) vs tabular ({len(tabular)})")
        n = min(len(embeddings), len(tabular))
        embeddings = embeddings[:n]
        tabular = tabular[:n]
    return np.hstack([embeddings, tabular])


def predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Retorna probabilidades de favorabilidade [0,1]."""
    return model.predict(X).astype(np.float32)


def reconstruct_favorability_raster(
    proba: np.ndarray,
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    tile_bounds: tuple,
    resolution: float,
    patch_size_px: int,
    stride_px: int,
    method: str = "gaussian",
    gaussian_sigma_stride: float = 1.0,
    nodata: float = -9999,
) -> np.ndarray:
    """
    Reconstrói a superfície de favorabilidade a partir dos centros de patch.

    O raster continua em 30 m para compatibilidade GIS, mas a unidade efetiva
    de decisão é o passo dos embeddings (stride). O modo gaussian faz uma
    janela móvel local sobre os centros, costurando limites de chunks sem a
    interpolação regional que criava manchas gigantes.
    """
    minx, miny, maxx, maxy = tile_bounds
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))

    cols = np.floor((centers_x - minx) / resolution).astype(int)
    rows = np.floor((maxy - centers_y) / resolution).astype(int)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    vals = proba.astype(np.float32)[in_bounds]

    if method == "block":
        result = np.full((height, width), nodata, dtype=np.float32)
        cell_px = max(1, int(round(stride_px)))
        half = max(0, cell_px // 2)

        for row, col, val in zip(rows, cols, vals):
            r0 = max(0, row - half)
            r1 = min(height, row + half + 1)
            c0 = max(0, col - half)
            c1 = min(width, col + half + 1)
            block = result[r0:r1, c0:c1]
            empty = block == nodata
            block[empty] = val
            block[~empty] = np.maximum(block[~empty], val)

        result[result == nodata] = 0.0
        return result

    from scipy.ndimage import gaussian_filter

    value_grid = np.zeros((height, width), dtype=np.float32)
    weight_grid = np.zeros((height, width), dtype=np.float32)

    np.add.at(value_grid, (rows, cols), vals)
    np.add.at(weight_grid, (rows, cols), 1.0)

    sigma = max(1.0, float(stride_px) * float(gaussian_sigma_stride))
    smooth_values = gaussian_filter(value_grid, sigma=sigma, mode="constant", cval=0.0, truncate=4.0)
    smooth_weights = gaussian_filter(weight_grid, sigma=sigma, mode="constant", cval=0.0, truncate=4.0)

    result = np.zeros((height, width), dtype=np.float32)
    valid = smooth_weights > 1e-8
    result[valid] = (smooth_values[valid] / smooth_weights[valid]).astype(np.float32)
    return result


def run_inference_tile(
    tile_name: str,
    models: Dict[str, object],
    config: dict,
) -> Dict[str, Path]:
    """Executa inferência para um tile e salva GeoTIFFs por tipo de depósito."""
    embed_dir = Path(config["paths"]["embeddings"])
    raster_dir = Path(config["paths"]["rasters"])
    output_dir = Path(config["paths"]["output"]) / "geotiff"
    nodata = config["output"]["nodata"]
    resolution = config["project"]["resolution"]
    patch_px = config["prithvi"]["patch_size"]
    stride_px = config["prithvi"].get("stride", patch_px)
    surface_method = config.get("output", {}).get("surface_method", "gaussian")
    gaussian_sigma_stride = config.get("output", {}).get("surface_gaussian_sigma_stride", 1.0)

    npz_path = embed_dir / f"{tile_name}.npz"
    if not npz_path.exists():
        logger.warning(f"NPZ não encontrado: {npz_path}")
        return {}

    # Carregar embeddings
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    cx = data["centers_x"]
    cy = data["centers_y"]

    # Construir features tabulares para inferência
    logger.info(f"Tile '{tile_name}': {len(embeddings)} patches")

    # Dummy tabular (zeros) se não disponíveis — features geoquímicas/SEGEMAR
    # Na prática, reusa o pipeline de build_tabular_features do train.py
    from train_utils_infer import build_tabular_for_inference, align_features_to_model
    try:
        tab_df = build_tabular_for_inference(cx, cy, config)
    except Exception as e:
        logger.warning(f"Features tabulares falhou ({e}) — usando zeros")
        tab_df = pd.DataFrame(np.zeros((len(embeddings), 0)))

    # Usar feature_names do primeiro modelo disponível para alinhar colunas
    feat_names_path = next(
        (Path(config["paths"]["models"]) / dt / "feature_names.json"
         for dt in models if (Path(config["paths"]["models"]) / dt / "feature_names.json").exists()),
        Path("nonexistent")
    ) if models else Path("nonexistent")

    X = align_features_to_model(embeddings, tab_df, feat_names_path)

    # Bounds do tile
    tiles_gdf = load_tile_geometries(config)
    tile_row = tiles_gdf[tiles_gdf.get("tile_name", tiles_gdf.index.astype(str)) == tile_name] if len(tiles_gdf) else []
    if len(tile_row) == 0:
        # Fallback: extensão dos patches + margem de meio patch
        half_patch = patch_px * resolution / 2
        tile_bounds = (
            cx.min() - half_patch,
            cy.min() - half_patch,
            cx.max() + half_patch,
            cy.max() + half_patch,
        )
        logger.debug(f"Bounds calculados dos patches: {tile_bounds}")
    else:
        tile_bounds = tile_row.iloc[0].geometry.bounds

    output_paths = {}
    for dtype, model in models.items():
        out_dir = output_dir / dtype
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"favorabilidade_{dtype}_{tile_name}.tif"

        logger.info(f"  Inferência: {dtype}")
        proba = predict_probabilities(model, X)

        grid = reconstruct_favorability_raster(
            proba=proba,
            centers_x=cx,
            centers_y=cy,
            tile_bounds=tile_bounds,
            resolution=resolution,
            patch_size_px=patch_px,
            stride_px=stride_px,
            method=surface_method,
            gaussian_sigma_stride=gaussian_sigma_stride,
            nodata=nodata,
        )

        profile = get_profile(
            bounds=tile_bounds,
            resolution=resolution,
            crs=config["project"]["crs"],
            count=1,
            dtype="float32",
            nodata=nodata,
        )
        save_raster(grid, out_path, profile, descriptions=[f"favorabilidade_{dtype}"])
        output_paths[dtype] = out_path
        logger.success(f"  Salvo: {out_path.name} | proba média={proba.mean():.3f}")

    return output_paths


def run(config_path: str = None, tiles: List[str] = None):
    import time
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FASE 6a · Inferência — Mapas de Favorabilidade")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])

    # Carregar todos os modelos disponíveis
    models = {}
    for dtype in PATCH_TYPES:
        model_path = models_dir / dtype / "lgbm_model.pkl"
        if model_path.exists():
            models[dtype] = load_model(model_path)
            logger.info(f"Modelo carregado: {dtype}")
        else:
            logger.warning(f"Modelo não encontrado: {model_path}")

    if not models:
        logger.error("Nenhum modelo encontrado. Execute train.py primeiro.")
        return

    # Descobrir tiles a partir dos embeddings existentes
    embed_dir = Path(cfg["paths"]["embeddings"])
    if tiles:
        tile_names = tiles
    else:
        cfg_tiles = cfg.get("inference", {}).get("prediction_tiles")
        if cfg_tiles:
            tile_names = cfg_tiles
            logger.info(f"Tiles de predição por config: {tile_names}")
        else:
            tile_names = sorted([p.stem for p in embed_dir.glob("*.npz")])
            logger.info(f"Tiles com embeddings: {tile_names}")
    all_outputs = {}

    for tile_name in tile_names:
        logger.info(f"\nProcessando tile: {tile_name}")
        outputs = run_inference_tile(tile_name, models, cfg)
        for dtype, path in outputs.items():
            all_outputs.setdefault(dtype, []).append(path)

    elapsed = time.time() - t0
    logger.success(f"Inferência concluída em {elapsed/60:.0f} min")
    return all_outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--tiles", nargs="+", default=None)
    args = parser.parse_args()
    run(config_path=args.config, tiles=args.tiles)
