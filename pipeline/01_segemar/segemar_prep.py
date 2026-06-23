"""
Fase 1 · Preparação de dados SEGEMAR locais.

Rasteriza apenas os 3 shapefiles de cobertura regional definidos em config.yaml:
  - e2_5M_Geotectonico   → rasters categóricos (clasif_tec, orogenia)
  - e5M_AMS_Fallas       → raster de densidade de falhas + distância mínima
  - e2_5M_UnidadesGeologicas → rasters categóricos (ambiente, litologia)

Saída: D:/argentina/data/05_RASTERS/segemar/
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import fiona
import pyproj
from shapely.geometry import shape, box
from shapely import wkb as swkb
from shapely.ops import transform as shp_transform
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.raster_utils import load_config, get_profile, save_raster

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add(
    Path("D:/argentina/logs/segemar_prep_{time:YYYY-MM-DD}.log"),
    level="DEBUG", rotation="50 MB",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_shapefile_to_crs(shp_path: str, target_crs: str) -> gpd.GeoDataFrame:
    """Carrega shapefile via fiona + pyproj, evitando o bug shapely create_collection."""
    records = []
    with fiona.open(str(shp_path)) as src:
        src_crs = src.crs or "EPSG:4326"
        transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
        for feat in src:
            try:
                geom = shape(feat["geometry"])
                geom = swkb.loads(geom.wkb)
                geom_t = shp_transform(transformer.transform, geom)
                props = dict(feat["properties"])
                props["geometry"] = geom_t
                records.append(props)
            except Exception:
                pass
    if not records:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]


def build_reference_profile(config: dict) -> dict:
    """Profile de referência cobrindo a AOI completa."""
    bbox = config["aoi"]["bbox_utm19s"]  # [minx, miny, maxx, maxy]
    res = config["project"]["resolution"]
    minx = np.floor(bbox[0] / res) * res
    miny = np.floor(bbox[1] / res) * res
    maxx = np.ceil(bbox[2] / res) * res
    maxy = np.ceil(bbox[3] / res) * res
    return get_profile(
        bounds=(minx, miny, maxx, maxy),
        resolution=res,
        crs=config["project"]["crs"],
        count=1, dtype="float32",
        nodata=config["output"]["nodata"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rasterização
# ─────────────────────────────────────────────────────────────────────────────

def rasterize_categorical(
    gdf: gpd.GeoDataFrame,
    field: str,
    profile: dict,
    output_path: Path,
) -> Tuple[Path, Dict[str, int]]:
    """Rasteriza campo categórico — dados reais (1+), áreas sem dados = -1."""
    gdf = gdf.copy()
    gdf[field] = gdf[field].fillna("DESCONHECIDO").astype(str).str.strip()
    categories = sorted(gdf[field].unique())
    cat_map = {cat: i + 1 for i, cat in enumerate(categories)}
    gdf["_code"] = gdf[field].map(cat_map)

    shapes = [
        (geom, val)
        for geom, val in zip(gdf.geometry, gdf["_code"])
        if geom is not None and not geom.is_empty
    ]

    # fill=-1 → áreas sem cobertura do shapefile marcadas como -1 (sem_dados)
    burned = rasterize(
        shapes=shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=-1, dtype=np.int16, all_touched=True,
    ).astype(np.float32)

    prof = profile.copy()
    prof["dtype"] = "float32"
    save_raster(burned, output_path, prof, descriptions=[field])

    # Salvar legenda
    legend_path = output_path.with_suffix(".csv")
    legend_data = [{"codigo": v, "categoria": k} for k, v in cat_map.items()]
    legend_data.append({"codigo": -1, "categoria": "sem_dados"})
    pd.DataFrame(legend_data).to_csv(legend_path, index=False)

    logger.info(f"  Categórico '{field}': {len(categories)} classes | áreas sem dados = -1 → {output_path.name}")
    return output_path, cat_map


def rasterize_density(
    gdf: gpd.GeoDataFrame,
    profile: dict,
    output_path: Path,
    radius_m: float = 15000.0,
    normalize: bool = True,
) -> Path:
    """Kernel density para linhas/polígonos (KDE simplificado via cKDTree)."""
    if gdf.geom_type.iloc[0] in ["LineString", "MultiLineString"]:
        pts = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            length = geom.length
            n = max(2, int(length / (radius_m / 10)))
            for i in range(n):
                pt = geom.interpolate(i / (n - 1), normalized=True)
                pts.append((pt.x, pt.y))
    else:
        pts = [(g.centroid.x, g.centroid.y) for g in gdf.geometry if g and not g.is_empty]

    if not pts:
        logger.warning(f"Sem geometrias válidas: {output_path.name}")
        arr = np.full((profile["height"], profile["width"]), profile["nodata"], dtype=np.float32)
        save_raster(arr, output_path, profile)
        return output_path

    pts = np.array(pts)
    tree = cKDTree(pts)
    t = profile["transform"]
    xs = np.arange(profile["width"]) * t.a + t.c + t.a / 2
    ys = np.arange(profile["height"]) * t.e + t.f + t.e / 2

    # Processar em chunks de linhas para evitar alocação de 22M+ pontos de uma vez
    CHUNK_ROWS = 500
    counts_rows = []
    for row_start in range(0, profile["height"], CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, profile["height"])
        chunk_ys = ys[row_start:row_end]
        xx_c, yy_c = np.meshgrid(xs, chunk_ys)
        chunk_pts = np.column_stack([xx_c.ravel(), yy_c.ravel()])
        chunk_counts = np.array(
            tree.query_ball_point(chunk_pts, r=radius_m, return_length=True),
            dtype=np.float32,
        ).reshape(row_end - row_start, profile["width"])
        counts_rows.append(chunk_counts)

    counts = np.vstack(counts_rows)

    if normalize and counts.max() > 0:
        counts = counts / counts.max()

    save_raster(counts, output_path, profile, descriptions=["density"])
    logger.info(f"  Densidade: {output_path.name} | max={counts.max():.3f} | r={radius_m/1000:.0f}km")
    return output_path


def rasterize_min_distance(
    gdf: gpd.GeoDataFrame,
    profile: dict,
    output_path: Path,
    sample_step_m: float = 1000.0,
) -> Path:
    """Distância mínima logarítmica a qualquer feição linear do GDF."""
    pts = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        length = getattr(geom, "length", 0)
        n = max(2, int(length / sample_step_m))
        for i in range(n):
            try:
                pt = geom.interpolate(i / (n - 1), normalized=True)
                pts.append((pt.x, pt.y))
            except Exception:
                pass

    if not pts:
        arr = np.full((profile["height"], profile["width"]), profile["nodata"], dtype=np.float32)
        save_raster(arr, output_path, profile)
        return output_path

    tree = cKDTree(np.array(pts))
    t = profile["transform"]
    xs = np.arange(profile["width"]) * t.a + t.c + t.a / 2
    ys = np.arange(profile["height"]) * t.e + t.f + t.e / 2

    # Processar em chunks de linhas para evitar alocação de 22M+ pontos de uma vez
    CHUNK_ROWS = 500
    dist_rows = []
    for row_start in range(0, profile["height"], CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, profile["height"])
        chunk_ys = ys[row_start:row_end]
        xx_c, yy_c = np.meshgrid(xs, chunk_ys)
        chunk_pts = np.column_stack([xx_c.ravel(), yy_c.ravel()])
        dists, _ = tree.query(chunk_pts, k=1)
        dist_rows.append(dists.reshape(row_end - row_start, profile["width"]))

    dist_arr = np.log1p(np.vstack(dist_rows)).astype(np.float32)

    save_raster(dist_arr, output_path, profile, descriptions=["log_dist_m"])
    logger.info(f"  Distância mínima (log): {output_path.name}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str = None):
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FASE 1 · Preparação SEGEMAR (3 shapefiles regionais)")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    crs = cfg["project"]["crs"]
    segemar_dir = Path(cfg["paths"]["segemar_dir"])
    out_dir = Path(cfg["paths"]["rasters"]) / "segemar"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = build_reference_profile(cfg)
    logger.info(
        f"Grid AOI completa: {profile['width']}×{profile['height']} px "
        f"@ {cfg['project']['resolution']}m | CRS: {crs}"
    )

    for layer_cfg in cfg["segemar_layers"]:
        shp = segemar_dir / layer_cfg["file"]
        if not shp.exists():
            logger.warning(f"Arquivo não encontrado: {shp.name} — pulando")
            continue

        logger.info(f"Carregando: {layer_cfg['file']} ({layer_cfg['type']})")
        gdf = load_shapefile_to_crs(str(shp), crs)
        if gdf.empty:
            logger.warning(f"GDF vazio após carregamento: {shp.name}")
            continue

        logger.info(f"  {len(gdf)} feições em {crs}")

        if layer_cfg["type"] == "categorical":
            out_path = out_dir / f"{layer_cfg['name']}.tif"
            rasterize_categorical(gdf, layer_cfg["field"], profile, out_path)

        elif layer_cfg["type"] == "density":
            radius = layer_cfg.get("density_radius_m", 15000)

            out_dens = out_dir / f"{layer_cfg['name']}_density.tif"
            if not out_dens.exists():
                rasterize_density(gdf, profile, out_dens, radius_m=radius)

            out_dist = out_dir / f"{layer_cfg['name']}_dist.tif"
            if not out_dist.exists():
                rasterize_min_distance(gdf, profile, out_dist)

    elapsed = time.time() - t0
    logger.success(f"Fase 1 concluída em {elapsed/60:.1f} min | Saída: {out_dir}")
    logger.info(f"Rasters gerados: {[p.name for p in sorted(out_dir.glob('*.tif'))]}")


if __name__ == "__main__":
    run()
