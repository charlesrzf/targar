"""
Fase 6b · Mosaico e Exportação dos Mapas Finais.

- Mosaica tiles de favorabilidade em GeoTIFFs AOI completos
- Gera shapefile de top targets rankeados
- Exporta relatório CSV com métricas dos modelos
- Cria mapa de visualização HTML (Folium)

Saída: data/08_OUTPUT/
  geotiff/  favorabilidade_porfiro_AOI.tif  (e outros tipos)
  shapefiles/  top_targets.shp
  reports/  ranking_targets.csv
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, box
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.raster_utils import load_config, mosaic_rasters, load_aoi_geometry

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

DEPOSIT_TYPES = ["porfiro", "skarn", "epitermal", "manto"]


def mosaic_all_tiles(config: dict) -> Dict[str, Path]:
    """Mosaica tiles de todos os tipos de depósito."""
    out_dir = Path(config["paths"]["output"]) / "geotiff"
    tile_dir = out_dir
    nodata = config["output"]["nodata"]
    mosaics = {}

    for dtype in DEPOSIT_TYPES:
        dtype_dir = tile_dir / dtype
        if not dtype_dir.exists():
            logger.warning(f"Diretório não encontrado: {dtype_dir}")
            continue

        tiles = sorted(dtype_dir.glob("favorabilidade_*.tif"))
        if not tiles:
            continue

        mosaic_path = out_dir / f"favorabilidade_{dtype}_AOI.tif"
        logger.info(f"Mosaicando {dtype}: {len(tiles)} tiles")
        try:
            mosaic_rasters(tiles, mosaic_path, nodata=nodata)
        except Exception as e:
            if "Permission denied" not in str(e):
                raise
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mosaic_path = out_dir / f"favorabilidade_{dtype}_AOI_latest_{stamp}.tif"
            logger.warning(
                f"Mosaico padrão bloqueado por outro aplicativo — salvando como {mosaic_path.name}"
            )
            mosaic_rasters(tiles, mosaic_path, nodata=nodata)

        mosaics[dtype] = mosaic_path
        logger.success(f"Mosaico {dtype}: {mosaic_path.name}")

    return mosaics


def extract_top_targets(
    mosaic_path: Path,
    deposit_type: str,
    n_top: int = 50,
    min_prob: float = 0.75,
    min_area_km2: float = 0.2,
    config: dict = None,
) -> gpd.GeoDataFrame:
    """
    Extrai polígonos de alta favorabilidade como alvos prospectivos.
    Vetoriza regiões com probabilidade > min_prob e rankeia por área x probabilidade.
    """
    nodata = config["output"]["nodata"] if config else -9999
    crs = config["project"]["crs"] if config else "EPSG:32719"

    with rasterio.open(mosaic_path) as src:
        arr = src.read(1)
        transform = src.transform

    # Binarizar com threshold adaptativo
    valid = arr != nodata
    if valid.sum() == 0:
        return gpd.GeoDataFrame()

    out_cfg = config.get("output", {}) if config else {}
    target_pct = float(out_cfg.get("target_percentile", 95))
    max_area_km2 = float(out_cfg.get("target_max_area_km2", 100))
    min_prob = float(out_cfg.get("target_min_prob", min_prob))
    min_area_km2 = float(out_cfg.get("target_min_area_km2", min_area_km2))

    # Para alvos discretos, usar cauda alta da distribuição e não percentil 80.
    pct_threshold = float(np.percentile(arr[valid], target_pct))
    threshold = max(min_prob, pct_threshold)
    binary = ((arr >= threshold) & valid).astype(np.uint8)

    # Vetorizar regiões contíguas
    polys = []
    for geom, val in shapes(binary, transform=transform):
        if val == 1:
            poly = shape(geom)
            area_km2 = poly.area / 1e6
            if min_area_km2 <= area_km2 <= max_area_km2:
                # Média de favorabilidade dentro do polígono
                from rasterio.mask import mask as rio_mask
                with rasterio.open(mosaic_path) as src:
                    try:
                        clipped, _ = rio_mask(src, [geom], crop=True, nodata=nodata)
                        clipped = clipped[0]
                        valid_px = clipped[clipped != nodata]
                        mean_prob = float(valid_px.mean()) if len(valid_px) > 0 else 0.0
                        max_prob = float(valid_px.max()) if len(valid_px) > 0 else 0.0
                    except Exception:
                        mean_prob = threshold
                        max_prob = threshold

                polys.append({
                    "geometry": poly,
                    "deposit_type": deposit_type,
                    "area_km2": round(area_km2, 3),
                    "mean_prob": round(mean_prob, 4),
                    "max_prob": round(max_prob, 4),
                    "score": round((0.7 * mean_prob + 0.3 * max_prob), 4),
                })

    if not polys:
        logger.warning(
            f"Nenhum alvo encontrado para {deposit_type} com prob > {threshold:.2f} "
            f"e área {min_area_km2}-{max_area_km2} km²"
        )
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(polys, crs=crs)
    gdf = gdf.sort_values(["score", "max_prob"], ascending=False).head(n_top).reset_index(drop=True)
    gdf["rank"] = gdf.index + 1
    gdf["target_id"] = [f"{deposit_type[:3].upper()}-{r['rank']:03d}" for _, r in gdf.iterrows()]

    logger.info(f"Alvos {deposit_type}: {len(gdf)} | score máx={gdf['score'].max():.2f}")
    return gdf


def generate_composite_map(mosaics: Dict[str, Path], output_path: Path, config: dict):
    """
    Gera mapa composto de favorabilidade máxima entre todos os tipos.
    Cada pixel recebe a maior probabilidade entre os modelos.
    """
    nodata = config["output"]["nodata"]
    arrays = []
    ref_profile = None

    for dtype, path in mosaics.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            arr = np.where(arr == nodata, np.nan, arr)
            arrays.append(arr)
            if ref_profile is None:
                ref_profile = src.profile.copy()

    if not arrays:
        return

    composite = np.nanmax(np.stack(arrays, axis=0), axis=0)
    composite = np.where(np.isnan(composite), nodata, composite)

    ref_profile.update(dtype="float32", count=1, nodata=nodata)
    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(composite[np.newaxis, :])
    logger.success(f"Mapa composto: {output_path.name}")


def generate_html_map(
    targets_gdf: gpd.GeoDataFrame,
    mosaics: Dict[str, Path],
    output_path: Path,
):
    """Gera mapa interativo HTML com Folium."""
    try:
        import folium
        from folium.plugins import MarkerCluster

        targets_wgs84 = targets_gdf.to_crs("EPSG:4326")
        center = [targets_wgs84.geometry.centroid.y.mean(),
                  targets_wgs84.geometry.centroid.x.mean()]

        m = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

        colors = {"porfiro": "red", "skarn": "blue", "epitermal": "orange", "manto": "green"}
        cluster = MarkerCluster(name="Alvos").add_to(m)

        for _, row in targets_wgs84.iterrows():
            c = row.geometry.centroid
            color = colors.get(row["deposit_type"], "gray")
            folium.CircleMarker(
                location=[c.y, c.x],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['target_id']}</b><br>"
                    f"Tipo: {row['deposit_type']}<br>"
                    f"Área: {row['area_km2']} km²<br>"
                    f"Prob. média: {row['mean_prob']:.3f}<br>"
                    f"Score: {row['score']:.2f}",
                    max_width=200,
                ),
            ).add_to(cluster)

        folium.LayerControl().add_to(m)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        logger.success(f"Mapa HTML: {output_path.name}")

    except Exception as e:
        logger.warning(f"Folium falhou: {e}")


def run(config_path: str = None):
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FASE 6b · Mosaico e Exportação")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    out_dir = Path(cfg["paths"]["output"])
    shp_dir = out_dir / "shapefiles"
    rep_dir = out_dir / "reports"
    shp_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Mosaico
    logger.info("Mosaicando tiles...")
    mosaics = mosaic_all_tiles(cfg)

    if not mosaics:
        logger.error("Nenhum mosaico gerado. Execute predict.py primeiro.")
        return

    # Mapa composto
    composite_path = out_dir / "geotiff" / "favorabilidade_COMPOSTA_AOI.tif"
    generate_composite_map(mosaics, composite_path, cfg)

    # Extrair alvos
    n_top = cfg["output"]["top_targets_n"]
    all_targets = []

    for dtype, mosaic_path in mosaics.items():
        logger.info(f"Extraindo alvos: {dtype}")
        gdf = extract_top_targets(
            mosaic_path=mosaic_path,
            deposit_type=dtype,
            n_top=n_top,
            min_prob=cfg["output"].get("target_min_prob", 0.75),
            min_area_km2=cfg["output"].get("target_min_area_km2", 0.2),
            config=cfg,
        )
        if len(gdf) > 0:
            all_targets.append(gdf)

    if not all_targets:
        logger.warning("Nenhum alvo encontrado")
        return

    targets_all = pd.concat(all_targets, ignore_index=True)
    targets_gdf = gpd.GeoDataFrame(targets_all, crs=cfg["project"]["crs"])

    # Salvar shapefile
    shp_path = shp_dir / "top_targets.gpkg"
    targets_gdf.to_file(shp_path, driver="GPKG")
    logger.success(f"Targets: {shp_path.name} | {len(targets_gdf)} alvos")

    # CSV ranking
    csv_cols = ["target_id", "deposit_type", "rank", "area_km2", "mean_prob", "max_prob", "score"]
    csv_path = rep_dir / "ranking_targets.csv"
    targets_gdf[csv_cols].sort_values(["deposit_type", "rank"]).to_csv(csv_path, index=False)
    logger.success(f"Ranking CSV: {csv_path.name}")

    # Métricas dos modelos
    models_dir = Path(cfg["paths"]["models"])
    metrics_rows = []
    for dtype in DEPOSIT_TYPES:
        meta_path = models_dir / dtype / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            metrics_rows.append({
                "deposit_type": dtype,
                "auc_roc": meta["metrics"].get("auc_roc", "N/A"),
                "auc_pr": meta["metrics"].get("auc_pr", "N/A"),
                "n_positives_test": meta["metrics"].get("n_positive", "N/A"),
            })
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(rep_dir / "model_metrics.csv", index=False)
        logger.success(f"Métricas salvas: {rep_dir}/model_metrics.csv")

    # Mapa HTML
    html_path = rep_dir / "mapa_interativo.html"
    generate_html_map(targets_gdf, mosaics, html_path)

    elapsed = time.time() - t0
    logger.success(f"\nPIPELINE COMPLETO em {elapsed/60:.0f} min")
    logger.success(f"Saída principal: {out_dir}")
    logger.success(f"  Mapas GeoTIFF:  {out_dir}/geotiff/")
    logger.success(f"  Alvos:          {shp_path}")
    logger.success(f"  Ranking:        {csv_path}")
    logger.success(f"  Mapa HTML:      {html_path}")


if __name__ == "__main__":
    run()
