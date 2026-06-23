"""Mede os buracos restantes nos composites S2."""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import rasterio

s2_dir = Path("D:/argentina/data/04_MPC_DOWNLOADS/s2")
print(f"{'arquivo':50} {'nodata%':>8} {'WxH':>14} {'res':>6}")
print("-"*82)
tot_nd = tot_px = 0
for seg in sorted(s2_dir.glob("*")):
    if not seg.is_dir():
        continue
    for tif in sorted(seg.glob("*.tif")):
        with rasterio.open(tif) as src:
            b = src.read(1)
            nd = int(np.sum(b == src.nodata) + np.sum(~np.isfinite(b)))
            px = b.size
            tot_nd += nd; tot_px += px
            res = abs(src.transform.a)
            print(f"{seg.name+'/'+tif.name[:35]:50} {100*nd/px:7.1f}% {src.width}x{src.height:<8} {res:5.0f}")
print("-"*82)
print(f"{'TOTAL nodata na AOI':50} {100*tot_nd/tot_px:7.1f}%")
