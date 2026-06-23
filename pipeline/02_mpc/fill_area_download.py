"""
Fase 2.6 · Download dirigido por polígono — fecha buracos S2 residuais.

Você desenha um polígono (em QGIS) cobrindo a área sem dados e salva como
shapefile. Este script:
  1. Lê o polígono (qualquer CRS) e calcula o bbox WGS84 (+ buffer).
  2. Baixa um composite S2 mediano só dessa área (lógica por-MGRS-tile corrigida).
  3. Reprojeta o patch para a grade de cada composite existente que o intersecta
     e preenche SOMENTE os pixels nodata (faz backup .bak antes de gravar).

Uso:
  python fill_area_download.py --polygon D:/argentina/01_AOI/patch_gap.shp
  python fill_area_download.py --polygon ... --buffer 1000 --max-scenes 80
"""
import sys
import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import fiona
import pyproj
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, box
from shapely.ops import transform as shp_transform, unary_union
from loguru import logger

warnings.filterwarnings("ignore")
sys.path.insert(0, "D:/argentina/pipeline")
sys.path.insert(0, "D:/argentina/pipeline/02_mpc")
from utils.raster_utils import load_config
from mpc_download_agent import MPCDownloadAgent, S2_ALL_BANDS

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<cyan>{time:HH:mm:ss}</cyan> | <level>{message}</level>")

CONFIG = "D:/argentina/pipeline/00_config/config.yaml"
S2_DIR = Path("D:/argentina/data/04_MPC_DOWNLOADS/s2")


def load_polygon(path: str):
    """Lê polígono(s) e retorna geometria unificada + CRS."""
    geoms = []
    with fiona.open(path) as src:
        crs = src.crs
        for feat in src:
            g = shape(feat["geometry"])
            if g and not g.is_empty:
                geoms.append(g)
    if not geoms:
        raise SystemExit("Polígono vazio.")
    return unary_union(geoms), crs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polygon", required=True, help="shapefile do polígono da área a preencher")
    ap.add_argument("--buffer", type=float, default=500.0, help="buffer em metros (default 500)")
    ap.add_argument("--max-scenes", type=int, default=None, help="override max_scenes")
    ap.add_argument("--cloud-max", type=int, default=None, help="override cloud_cover_max (%)")
    ap.add_argument("--all-months", action="store_true",
                    help="usar todos os meses (não só estação seca) — maximiza cobertura")
    args = ap.parse_args()

    cfg = load_config(CONFIG)
    utm = cfg["project"]["crs"]
    s2c = cfg["satellite"]["s2"]
    max_scenes = args.max_scenes or s2c["max_scenes"]
    cloud_max = args.cloud_max if args.cloud_max is not None else s2c["cloud_cover_max"]
    dry_only = not args.all_months

    # 1) Polígono → UTM (buffer) → bbox WGS84
    poly, poly_crs = load_polygon(args.polygon)
    to_utm = pyproj.Transformer.from_crs(poly_crs, utm, always_xy=True).transform
    poly_utm = shp_transform(to_utm, poly).buffer(args.buffer)
    to_wgs = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform
    bbox_wgs = tuple(shp_transform(to_wgs, poly_utm).bounds)
    logger.info(f"Área (UTM bounds): {tuple(round(v) for v in poly_utm.bounds)}")
    logger.info(f"bbox WGS84 p/ busca: {tuple(round(v,4) for v in bbox_wgs)}")

    # 2) Download do patch composite
    agent = MPCDownloadAgent(cfg)
    logger.info(f"Filtros: cloud<{cloud_max}% | {'estação seca' if dry_only else 'TODOS os meses'} "
                f"| max_scenes={max_scenes}")
    items = agent.search_s2_scenes(
        bbox=bbox_wgs,
        date_start=s2c["date_start"], date_end=s2c["date_end"],
        cloud_max=cloud_max, dry_months_only=dry_only,
        max_scenes=max_scenes,
    )
    if not items:
        raise SystemExit("Nenhuma cena S2 encontrada para a área.")

    patch_dir = S2_DIR / "_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    bs = "_".join(f"{v:.3f}" for v in bbox_wgs)
    patch_path = patch_dir / f"patch_{bs}.tif"
    if patch_path.exists():
        patch_path.unlink()
    agent.download_s2_composite(bbox_wgs, items, patch_path, bands=S2_ALL_BANDS)
    if not patch_path.exists():
        raise SystemExit("Falha ao gerar o composite do patch.")

    # Medir cobertura do patch (sanidade antes de remendar)
    with rasterio.open(patch_path) as p:
        b0 = p.read(1)
        nd = float(np.mean((b0 == (p.nodata if p.nodata is not None else -9999))
                           | ~np.isfinite(b0)))
    logger.info(f"PATCH gerado: {patch_path.name} | nodata no patch: {100*nd:.1f}%")
    if nd > 0.5:
        logger.warning("Patch ainda tem MUITO nodata — tente --all-months e/ou --cloud-max maior.")

    # 3) Remendar composites existentes que intersectam o polígono
    with rasterio.open(patch_path) as p:
        patch_arr = p.read().astype(np.float32)
        patch_transform, patch_crs, patch_nd = p.transform, p.crs, p.nodata

    n_filled_total = 0
    for tif in sorted(S2_DIR.rglob("s2_composite_*.tif")):
        with rasterio.open(tif) as src:
            if not box(*src.bounds).intersects(poly_utm):
                continue
            target = src.read().astype(np.float32)
            prof = src.profile.copy()
            nodata = src.nodata if src.nodata is not None else -9999

        # nodata atual (qualquer banda inválida no pixel)
        nd_mask = ~np.all(np.isfinite(target) & (target != nodata), axis=0)
        if not nd_mask.any():
            continue

        # reprojetar patch para a grade exata deste composite
        patch_on = np.full(target.shape, np.nan, dtype=np.float32)
        for b in range(target.shape[0]):
            reproject(
                source=patch_arr[b], destination=patch_on[b],
                src_transform=patch_transform, src_crs=patch_crs,
                dst_transform=prof["transform"], dst_crs=prof["crs"],
                src_nodata=patch_nd, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

        patch_valid = np.all(np.isfinite(patch_on) & (patch_on != nodata), axis=0)
        fillable = nd_mask & patch_valid
        n_fill = int(fillable.sum())
        if n_fill == 0:
            logger.info(f"  {tif.name[:42]:42} sem pixels preenchíveis pelo patch")
            continue

        for b in range(target.shape[0]):
            target[b][fillable] = patch_on[b][fillable]

        shutil.copy2(tif, tif.with_suffix(".tif.bak"))
        with rasterio.open(tif, "w", **prof) as dst:
            dst.write(target)

        resid = int((~np.all(np.isfinite(target) & (target != nodata), axis=0)).sum())
        logger.success(f"  {tif.name[:42]:42} +{n_fill} px | residual {resid} px (.bak salvo)")
        n_filled_total += n_fill

    logger.success(f"Concluído. Total preenchido: {n_filled_total} px. Patch: {patch_path.name}")


if __name__ == "__main__":
    main()
