"""
Fase 2.5 · Gap-fill dos composites S2.

Preenche buracos PEQUENOS de nodata por interpolação (GDAL fillnodata),
operando sobre os composites já baixados — não baixa nada novo.

- Buracos pequenos/médios (interior, perto de dados válidos) → preenchidos.
- Buracos grandes (> max_nodata_pct) → PULADOS (precisam de re-download,
  ex.: chunk de alta cordilheira com neve permanente).

Uso:
  python gap_fill_s2.py                 # processa todos os composites
  python gap_fill_s2.py --max-dist 80   # distância máx. de busca (px)
  python gap_fill_s2.py --skip-above 40 # pula chunks com > 40% nodata
"""
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.fill import fillnodata
from loguru import logger

warnings.filterwarnings("ignore")

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<cyan>{time:HH:mm:ss}</cyan> | <level>{message}</level>")

S2_DIR = Path("D:/argentina/data/04_MPC_DOWNLOADS/s2")


def fill_composite(tif_path: Path, max_search_distance: float,
                   smoothing_iters: int, skip_above_pct: float) -> str:
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        data = src.read()                      # (bands, H, W)
        nodata = src.nodata if src.nodata is not None else -9999

    # Máscara de validade combinada (pixel válido em TODAS as bandas)
    valid_all = np.all(np.isfinite(data) & (data != nodata), axis=0)
    nd_pct = 100.0 * (1.0 - valid_all.mean())

    if nd_pct < 0.01:
        return f"ok 0.0% (nada a fazer)"
    if nd_pct > skip_above_pct:
        return f"PULADO {nd_pct:.1f}% (> {skip_above_pct:.0f}% — precisa re-download)"

    # fillnodata por banda; mask: 1=válido (mantém), 0=preenche
    mask = valid_all.astype(np.uint8)
    filled = np.empty_like(data)
    for b in range(data.shape[0]):
        band = data[b].astype(np.float32)
        band = np.where(np.isfinite(band), band, nodata)
        filled[b] = fillnodata(
            band, mask=mask,
            max_search_distance=max_search_distance,
            smoothing_iterations=smoothing_iters,
        )

    # nodata residual (fora do alcance da busca) → mantém nodata
    still_nd = ~np.all(np.isfinite(filled) & (filled != nodata), axis=0)
    resid_pct = 100.0 * still_nd.mean()

    profile.update(nodata=nodata)
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(filled)

    return f"preenchido {nd_pct:.1f}% -> {resid_pct:.1f}% residual"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=float, default=100.0,
                    help="distância máxima de busca em pixels (default 100 = ~9km @90m)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="iterações de suavização pós-fill")
    ap.add_argument("--skip-above", type=float, default=40.0,
                    help="pula chunks com nodata acima deste %% (precisam re-download)")
    args = ap.parse_args()

    tifs = sorted(S2_DIR.rglob("s2_composite_*.tif"))
    if not tifs:
        logger.error(f"Nenhum composite em {S2_DIR}")
        return

    logger.info(f"Gap-fill em {len(tifs)} composites | max_dist={args.max_dist}px "
                f"| skip>{args.skip_above}%")
    for tif in tifs:
        try:
            msg = fill_composite(tif, args.max_dist, args.smooth, args.skip_above)
            logger.info(f"  {tif.parent.name}/{tif.name[:38]:38} {msg}")
        except Exception as e:
            logger.error(f"  {tif.name}: ERRO {e}")

    logger.success("Gap-fill concluído.")


if __name__ == "__main__":
    main()
