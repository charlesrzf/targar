"""
Fase 3 · Índices espectrais por tipo de depósito.

Calcula índices S2 e ASTER relevantes para cada tipo de depósito:
  - Pórfiro Cu-Mo:   clay index, ferric iron, iron oxide, propylitic, potassic
  - Epitermal HS/LS: alunite, kaolinite, silica (TIR), gossan
  - Skarn:           carbonate (ASTER TIR), calc-silicate
  - IOCG:            Fe-oxide, breccia (SAR ratio)
  - Geral:           NDVI, NDWI, BSI

Saída: D:/argentina/data/05_RASTERS/indices/
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.raster_utils import load_config, save_raster

logger.remove()
logger.add(sys.stderr, level="INFO", format="<yellow>{time:HH:mm:ss}</yellow> | <level>{message}</level>")
logger.add(Path("D:/argentina/logs/spectral_indices_{time:YYYY-MM-DD}.log"), level="DEBUG", rotation="50 MB")


# ─────────────────────────────────────────────────────────────────────────────
# Índices S2 (bandas mapeadas por nome)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_ratio(a: np.ndarray, b: np.ndarray, nodata: float = -9999) -> np.ndarray:
    """Razão segura: retorna nodata onde denominador == 0."""
    mask = (b == 0) | (a == nodata) | (b == nodata)
    result = np.where(mask, nodata, np.divide(a, b, where=b != 0))
    return result.astype(np.float32)


def _norm_diff(a: np.ndarray, b: np.ndarray, nodata: float = -9999) -> np.ndarray:
    """(a - b) / (a + b) — NDVI-like."""
    denom = a + b
    mask = (denom == 0) | (a == nodata) | (b == nodata)
    result = np.where(mask, nodata, np.divide(a - b, denom, where=denom != 0))
    return np.clip(result, -1, 1).astype(np.float32)


def compute_s2_indices(bands: Dict[str, np.ndarray], nodata: float = -9999) -> Dict[str, np.ndarray]:
    """
    Calcula índices espectrais a partir de dict de bandas S2.
    bands: {'B02': arr, 'B03': arr, ..., 'B12': arr}  — reflectância 0-1
    """
    b = bands  # alias

    indices = {}

    # ── Vegetação / Máscara ────────────────────────────────────────────────
    if "B08" in b and "B04" in b:
        indices["NDVI"] = _norm_diff(b["B08"], b["B04"], nodata)

    if "B03" in b and "B08" in b:
        indices["NDWI"] = _norm_diff(b["B03"], b["B08"], nodata)

    # ── Pórfiro / Alteração argílica ──────────────────────────────────────
    # Clay Mineral Index (Drury 2001): SWIR1/SWIR2
    if "B11" in b and "B12" in b:
        indices["Clay_Index"] = _safe_ratio(b["B11"], b["B12"], nodata)

    # Ferrous Iron Index
    if "B08" in b and "B11" in b:
        indices["Ferrous_Iron"] = _safe_ratio(b["B08"], b["B11"], nodata)

    # Ferric Iron (gossan, limonita)
    if "B04" in b and "B02" in b:
        indices["Ferric_Iron"] = _safe_ratio(b["B04"], b["B02"], nodata)

    # Iron Oxide Index
    if "B04" in b and "B03" in b:
        indices["Iron_Oxide"] = _safe_ratio(b["B04"], b["B03"], nodata)

    # ── Pórfiro / Propylítico ──────────────────────────────────────────────
    # Chlorite/Epidote: Red Edge vs SWIR
    if "B05" in b and "B11" in b:
        indices["Propylitic_Proxy"] = _safe_ratio(b["B05"], b["B11"], nodata)

    # ── SWIR alteração hidrotermal (argilominerais) ────────────────────────
    if "B11" in b and "B8A" in b:
        indices["Hydroxyl_Bearing"] = _safe_ratio(b["B11"], b["B8A"], nodata)

    # ── Índice de Óxido de Ferro composto ─────────────────────────────────
    if "B04" in b and "B03" in b and "B02" in b:
        indices["Fe_Oxide_Composite"] = (
            b["B04"] / (b["B03"] + 1e-10) * b["B04"] / (b["B02"] + 1e-10)
        ).astype(np.float32)
        indices["Fe_Oxide_Composite"] = np.where(
            (b["B04"] == nodata), nodata, indices["Fe_Oxide_Composite"]
        )

    # ── Bare Soil Index ────────────────────────────────────────────────────
    if "B11" in b and "B04" in b and "B08" in b and "B02" in b:
        num = b["B11"] + b["B04"] - b["B08"] - b["B02"]
        den = b["B11"] + b["B04"] + b["B08"] + b["B02"]
        indices["BSI"] = np.where(den == 0, nodata, num / (den + 1e-10)).astype(np.float32)

    return indices


def compute_aster_indices(bands: Dict[str, np.ndarray], nodata: float = -9999) -> Dict[str, np.ndarray]:
    """
    Calcula índices espectrais ASTER VNIR+SWIR+TIR.
    Numeração: B01-B03 VNIR, B04-B09 SWIR, B10-B14 TIR
    """
    b = bands

    indices = {}

    # ── Epitermal Alta Sulfidação ──────────────────────────────────────────
    # Alunite (Al-OH): B5 alta absorção alunita vs B7
    if "B05" in b and "B07" in b:
        indices["Alunite_Index"] = _safe_ratio(b["B05"], b["B07"], nodata)

    # ── Argila / Caolinita ────────────────────────────────────────────────
    if "B04" in b and "B06" in b:
        indices["Kaolinite_Index"] = _safe_ratio(b["B04"], b["B06"], nodata)

    # ── Propylítico (Clorita/Epidoto) ─────────────────────────────────────
    if "B06" in b and "B09" in b:
        indices["Propylitic_ASTER"] = _safe_ratio(b["B06"], b["B09"], nodata)

    # ── Carbonato (Calcita/Dolomita) → Skarn ─────────────────────────────
    if "B06" in b and "B08" in b:
        indices["Carbonate_Index"] = _safe_ratio(b["B06"], b["B08"], nodata)

    # ── Potássico (alteração potássica) ───────────────────────────────────
    if "B07" in b and "B06" in b:
        indices["Potassic_Alteration"] = _safe_ratio(b["B07"], b["B06"], nodata)

    # ── Fe-Óxido ASTER VNIR ───────────────────────────────────────────────
    if "B02" in b and "B01" in b:
        indices["Fe_Oxide_ASTER"] = _safe_ratio(b["B02"], b["B01"], nodata)

    # ── Sílica (TIR) ──────────────────────────────────────────────────────
    if "B11" in b and "B10" in b:
        indices["Silica_TIR"] = _safe_ratio(b["B11"], b["B10"], nodata)

    # ── Skarns / Calc-silicatos (TIR multi-banda) ─────────────────────────
    if all(f"B{i:02d}" in b for i in [10, 11, 12, 13, 14]):
        indices["Calc_Silicate"] = (
            (b["B12"] + b["B13"]) / (b["B10"] + b["B14"] + 1e-10)
        ).astype(np.float32)

    # ── Gossã / Limonita (SWIR) ───────────────────────────────────────────
    if "B02" in b and "B03" in b:
        indices["Gossan_ASTER"] = _safe_ratio(b["B02"], b["B03"], nodata)

    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Processamento de arquivos GeoTIFF
# ─────────────────────────────────────────────────────────────────────────────

def load_bands_from_tif(
    tif_path: Path,
    band_names: List[str],
    nodata: float = -9999,
) -> Dict[str, np.ndarray]:
    """Carrega bandas nomeadas de um GeoTIFF multi-banda."""
    bands = {}
    with rasterio.open(tif_path) as src:
        for i in range(1, src.count + 1):
            tag = src.tags(i).get("name", f"B{i:02d}")
            if tag in band_names or not band_names:
                arr = src.read(i).astype(np.float32)
                arr = np.where(arr == src.nodata, nodata, arr)
                bands[tag] = arr
    return bands


def process_tile_s2(
    s2_tif_dir: Path,
    output_dir: Path,
    nodata: float = -9999,
) -> List[Path]:
    """Processa todos os composites S2 de um diretório e salva índices."""
    s2_files = list(s2_tif_dir.rglob("s2_composite_*.tif"))
    if not s2_files:
        logger.warning(f"Nenhum arquivo S2 encontrado em {s2_tif_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for tif in tqdm(s2_files, desc="S2 índices"):
        stem = tif.stem.replace("s2_composite_", "")
        bands = load_bands_from_tif(tif, S2_ALL_BANDS := [
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
        ], nodata)

        if not bands:
            continue

        indices = compute_s2_indices(bands, nodata)

        with rasterio.open(tif) as src:
            profile = src.profile.copy()
            profile.update(count=1, dtype="float32", nodata=nodata)

        for idx_name, arr in indices.items():
            out = output_dir / f"s2_{idx_name}_{stem}.tif"
            if not out.exists():
                save_raster(arr, out, profile, descriptions=[idx_name])
            saved.append(out)

    logger.success(f"S2 índices: {len(saved)} arquivos → {output_dir}")
    return saved


def process_tile_aster(
    aster_tif_dir: Path,
    output_dir: Path,
    nodata: float = -9999,
) -> List[Path]:
    """Processa arquivos ASTER e salva índices minerais."""
    aster_files = list(aster_tif_dir.rglob("*.tif"))
    if not aster_files:
        logger.warning(f"Nenhum arquivo ASTER em {aster_tif_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    ASTER_BANDS = [f"B{i:02d}" for i in range(1, 15)]

    for tif in tqdm(aster_files, desc="ASTER índices"):
        stem = tif.stem
        bands = load_bands_from_tif(tif, ASTER_BANDS, nodata)
        if not bands:
            continue

        indices = compute_aster_indices(bands, nodata)

        with rasterio.open(tif) as src:
            profile = src.profile.copy()
            profile.update(count=1, dtype="float32", nodata=nodata)

        for idx_name, arr in indices.items():
            out = output_dir / f"aster_{idx_name}_{stem}.tif"
            if not out.exists():
                save_raster(arr, out, profile, descriptions=[idx_name])
            saved.append(out)

    logger.success(f"ASTER índices: {len(saved)} arquivos → {output_dir}")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str = None):
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FASE 3 · Índices Espectrais")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    downloads = Path(cfg["paths"]["downloads"])
    indices_dir = Path(cfg["paths"]["rasters"]) / "indices"
    nodata = cfg["output"]["nodata"]

    # Processar S2 — descobrir tiles dinamicamente (qualquer subdir de s2/)
    s2_base = downloads / "s2"
    tile_names_s2 = sorted([d.name for d in s2_base.iterdir() if d.is_dir()]) if s2_base.exists() else []
    logger.info(f"Tiles S2 encontrados: {tile_names_s2}")
    for tile_name in tile_names_s2:
        process_tile_s2(s2_base / tile_name, indices_dir / "s2" / tile_name, nodata)

    # Processar ASTER — idem
    aster_base = downloads / "aster"
    tile_names_aster = sorted([d.name for d in aster_base.iterdir() if d.is_dir()]) if aster_base.exists() else []
    if tile_names_aster:
        logger.info(f"Tiles ASTER encontrados: {tile_names_aster}")
        for tile_name in tile_names_aster:
            process_tile_aster(aster_base / tile_name, indices_dir / "aster" / tile_name, nodata)

    elapsed = time.time() - t0
    logger.success(f"Fase 3 concluída em {elapsed/60:.1f} min | Saída: {indices_dir}")


if __name__ == "__main__":
    run()
