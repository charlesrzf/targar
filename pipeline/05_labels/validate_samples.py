"""
Validação visual das amostras (patches rotulados).

Gera, em data/08_OUTPUT/validation/:
  - positivos_porfiro.png  : contact sheet RGB dos patches positivos
  - negativos_amostra.png  : contact sheet RGB de uma amostra de negativos
  - mapa_amostras.png      : centros dos patches (pos/neg) sobre ocorrências Cu

Cada chip é o recorte RGB real do composite S2 (B04/B03/B02) na janela do patch.
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "D:/argentina/pipeline")
from utils.raster_utils import load_config

cfg = load_config("D:/argentina/pipeline/00_config/config.yaml")
S2_DIR = Path(cfg["paths"]["downloads"]) / "s2"
LABELS = Path(cfg["paths"]["data"]) / "labels" / "labels.csv"
OUT = Path(cfg["paths"]["output"]) / "validation"
OUT.mkdir(parents=True, exist_ok=True)
PATCH = cfg["prithvi"]["patch_size"]

# índice composite stem -> path
composites = {p.stem: p for p in S2_DIR.rglob("s2_composite_*.tif")}
PID_RE = re.compile(r"^(.*)_r(\d+)_c(\d+)$")


def chip_rgb(patch_id):
    m = PID_RE.match(patch_id)
    if not m:
        return None
    stem, r, c = m.group(1), int(m.group(2)), int(m.group(3))
    tif = composites.get(stem)
    if tif is None:
        return None
    with rasterio.open(tif) as src:
        tags = {src.tags(i).get("name", f"B{i:02d}"): i for i in range(1, src.count + 1)}
        idx = [tags.get(b) for b in ("B04", "B03", "B02")]
        if any(i is None for i in idx):
            return None
        win = Window(c, r, PATCH, PATCH)
        rgb = np.stack([src.read(i, window=win).astype(np.float32) for i in idx], -1)
    rgb[rgb == src.nodata] = np.nan
    # stretch percentil 2-98
    valid = rgb[np.isfinite(rgb)]
    if valid.size == 0:
        return None
    lo, hi = np.nanpercentile(valid, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(rgb)


def contact_sheet(df, title, out_png, ncols=8, max_n=64):
    df = df.head(max_n)
    n = len(df)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.6, nrows*1.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, df.iterrows()):
        chip = chip_rgb(row["patch_id"])
        if chip is not None:
            ax.imshow(chip)
        ax.set_title(f"{row['split']}", fontsize=6)
    fig.suptitle(f"{title}  (n={n})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {out_png}")


def main():
    df = pd.read_csv(LABELS)
    print(f"Labels: {len(df)} | colunas: {list(df.columns)}")
    print(df["label"].value_counts().to_dict() if "label" in df else "")
    pos = df[df["label"] == 1] if "label" in df else df[df["deposit_type"] == "porfiro"]
    neg = df[df["label"] == 0] if "label" in df else df[df["deposit_type"] == "negative"]
    print(f"Positivos: {len(pos)} | Negativos: {len(neg)}")

    contact_sheet(pos, "POSITIVOS · Cu pórfiro", OUT / "positivos_porfiro.png", max_n=64)
    contact_sheet(neg.sample(min(64, len(neg)), random_state=42),
                  "NEGATIVOS (amostra)", OUT / "negativos_amostra.png", max_n=64)

    # Mapa dos centros + ocorrências
    fig, ax = plt.subplots(figsize=(6, 14))
    ax.scatter(neg["center_x"], neg["center_y"], s=8, c="#3b6", label="negativo", alpha=.5)
    ax.scatter(pos["center_x"], pos["center_y"], s=22, c="#e33", label="positivo", edgecolor="k", linewidth=.3)
    try:
        sys.path.insert(0, "D:/argentina/pipeline/05_labels")
        from labels import load_occurrences
        occ = load_occurrences(cfg)
        occ_por = occ[occ["deposit_type"] == "porfiro"]
        ax.scatter(occ_por.geometry.x, occ_por.geometry.y, s=40, marker="*",
                   c="yellow", edgecolor="k", linewidth=.4, label="ocorrência SEGEMAR Cu")
    except Exception as e:
        print(f"(ocorrências não plotadas: {e})")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Amostras × Ocorrências Cu (UTM 19S)")
    fig.savefig(OUT / "mapa_amostras.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {OUT/'mapa_amostras.png'}")


if __name__ == "__main__":
    main()
