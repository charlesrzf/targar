"""
Pipeline PER-PIXEL (benchmark do Prithvi) — Cu pórfiro Argentina.

Classificação clássica de prospectividade por célula de 30 m:
  - Grade mestre 30 m = perfil de um raster SEGEMAR (AOI-wide).
  - Atributos/célula: 10 bandas S2 + índices S2 + terreno (slope/aspect/tpi/curv)
    + SEGEMAR (geologia/geotectônica categóricas, falhas densidade/dist).
  - Amostras = mesma base (zona positiva: ocorrências Cu + buffer ∪ amostras campo;
    negativas afastadas). Células coincidentes → treino/CV.
  - Compara RandomForest, HistGradientBoosting e LightGBM (CV estratificada, AUC),
    escolhe o melhor e aplica em toda a AOI em blocos → raster 30 m REAL.

Independe do Prithvi. Requer Fases 1–3 + derivadas de terreno prontas (30 m).

Uso:
  python pixel_pipeline.py --step all
  python pixel_pipeline.py --step vrt      # só (re)constrói os VRTs alinhados
  python pixel_pipeline.py --step train    # amostra + treina + escolhe modelo
  python pixel_pipeline.py --step predict   # aplica melhor modelo na AOI
"""
import sys, re, argparse, json, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import rasterize
from osgeo import gdal
from loguru import logger

warnings.filterwarnings("ignore")
gdal.UseExceptions()
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "05_labels"))
from utils.raster_utils import load_config

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<cyan>{time:HH:mm:ss}</cyan> | <level>{message}</level>")

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
TERRAIN = ["slope", "aspect", "tpi", "curvature"]
SEG_CONT = ["falhas_densidade_density", "falhas_densidade_dist"]
SEG_CAT = ["geotectonico_clasif", "geotectonico_orogenia",
           "geologia_ambiente", "geologia_litologia"]


# ─────────────────────────────────────────────────────────────────────────────
# Grade mestre + VRTs alinhados
# ─────────────────────────────────────────────────────────────────────────────
def master_grid(cfg):
    seg = Path(cfg["paths"]["rasters"]) / "segemar" / "geologia_ambiente.tif"
    with rasterio.open(seg) as src:
        return dict(crs=src.crs, transform=src.transform, width=src.width,
                    height=src.height, nodata=cfg["output"]["nodata"])


def build_vrts(cfg, vrt_dir: Path):
    """gdal.BuildVRT por camada (mosaica os chunks)."""
    vrt_dir.mkdir(parents=True, exist_ok=True)
    rasters = Path(cfg["paths"]["rasters"])
    downloads = Path(cfg["paths"]["downloads"])
    out = {}

    # S2 composites (multi-banda) → 1 VRT 10-band
    s2_tifs = [str(p) for p in (downloads / "s2").rglob("s2_composite_*.tif")]
    if s2_tifs:
        p = vrt_dir / "s2.vrt"
        gdal.BuildVRT(str(p), s2_tifs)
        out["s2"] = p
        logger.info(f"VRT s2: {len(s2_tifs)} chunks")

    # Índices S2 → 1 VRT por índice
    idx_groups = defaultdict(list)
    for tif in (rasters / "indices" / "s2").rglob("s2_*.tif"):
        key = tif.stem.split("_-")[0]            # s2_NDVI, s2_Clay_Index, ...
        idx_groups[key].append(str(tif))
    for key, tifs in sorted(idx_groups.items()):
        p = vrt_dir / f"idx_{key}.vrt"
        gdal.BuildVRT(str(p), tifs)
        out[f"idx_{key}"] = p
    logger.info(f"VRTs índices: {len(idx_groups)}")

    # Terreno → 1 VRT por derivada
    for name in TERRAIN:
        tifs = [str(p) for p in (rasters / "terrain").rglob(f"{name}_*.tif")]
        if tifs:
            p = vrt_dir / f"terr_{name}.vrt"
            gdal.BuildVRT(str(p), tifs)
            out[f"dem_{name}"] = p
    logger.info(f"VRTs terreno: {sum(1 for k in out if k.startswith('dem_'))}")

    # SEGEMAR (já AOI-wide) → caminho direto
    for name in SEG_CONT + SEG_CAT:
        tif = rasters / "segemar" / f"{name}.tif"
        if tif.exists():
            out[f"seg_{name}"] = tif
    logger.info(f"Camadas SEGEMAR: {sum(1 for k in out if k.startswith('seg_'))}")

    json.dump({k: str(v) for k, v in out.items()}, open(vrt_dir / "layers.json", "w"), indent=2)
    return out


def feature_spec(layers: dict):
    """Lista ordenada (nome_feature, vrt_path, band_idx, categorical)."""
    spec = []
    if "s2" in layers:
        for i, b in enumerate(S2_BANDS, 1):
            spec.append((f"s2_{b}", layers["s2"], i, False))
    for k in sorted(layers):
        if k.startswith("idx_"):
            spec.append((k, layers[k], 1, False))
    for name in TERRAIN:
        if f"dem_{name}" in layers:
            spec.append((f"dem_{name}", layers[f"dem_{name}"], 1, False))
    for name in SEG_CONT:
        if f"seg_{name}" in layers:
            spec.append((f"seg_{name}", layers[f"seg_{name}"], 1, False))
    for name in SEG_CAT:
        if f"seg_{name}" in layers:
            spec.append((f"seg_{name}", layers[f"seg_{name}"], 1, True))
    return spec


def warped(path, grid, categorical):
    src = rasterio.open(path)
    return WarpedVRT(src, crs=grid["crs"], transform=grid["transform"],
                     width=grid["width"], height=grid["height"],
                     resampling=Resampling.nearest if categorical else Resampling.bilinear,
                     src_nodata=None, nodata=np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Zonas explícitas de treino
# ─────────────────────────────────────────────────────────────────────────────
def _safe_union(geoms):
    from shapely.ops import unary_union
    clean = [g for g in geoms if g is not None and not g.is_empty]
    if not clean:
        return None
    return unary_union(clean)


def label_geometries(cfg):
    """Carrega nucleo/interpretado/negativo quando configurados."""
    from labels import load_label_sets

    if cfg["pixel_model"].get("use_label_sets", True) and cfg["deposits"].get("use_label_sets", False):
        sets = load_label_sets(cfg)
        out = {}
        for name, info in sets.items():
            geom = _safe_union(info["gdf"].geometry.values)
            if geom is None:
                continue
            out[name] = {
                "geometry": geom,
                "label": int(info["label"]),
                "weight": float(info["weight"]),
            }
            logger.info(
                f"Pixel label set {name}: label={out[name]['label']} "
                f"peso={out[name]['weight']} área={geom.area / 1e6:.1f} km²"
            )
        if "nucleo" not in out and "interpretado" not in out:
            raise SystemExit("Sem geometrias positivas em nucleo/interpretado.")
        if "negativo" not in out:
            logger.warning("Sem negativo.shp explícito; usarei negativos background.")
        return out

    from labels import load_occurrences, load_amostras
    geoms = []
    if cfg["deposits"].get("use_occurrences", True):
        try:
            occ = load_occurrences(cfg)
            por = occ[occ["deposit_type"] == "porfiro"]
            geoms += list(por["geometry_buffered"].values)
        except Exception as e:
            logger.warning(f"ocorrências: {e}")
    else:
        logger.info("Ocorrências GPKG desativadas; usando apenas shapefiles de amostras")
    try:
        amo = load_amostras(cfg)
        geoms += [g for g in amo.geometry.values if g is not None]
    except Exception as e:
        logger.warning(f"amostras: {e}")
    if not geoms:
        raise SystemExit("Sem geometrias positivas.")
    return {"legacy_positive": {"geometry": _safe_union(geoms), "label": 1, "weight": 1.0}}


def _sample_flat_indices(mask, n, rng):
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0 or n <= 0:
        return np.array([], dtype=np.int64)
    return rng.choice(idx, min(n, idx.size), replace=False)


def _focal_feature_names(base_name, windows, stats):
    return [f"{base_name}_w{w}_{stat}" for w in windows for stat in stats]


def focal_enabled_features(spec, cfg):
    prefixes = tuple(cfg["pixel_model"].get("focal_feature_prefixes", []))
    return [name for name, _path, _band, cat in spec if not cat and name.startswith(prefixes)]


def add_focal_features_to_points(df, cfg, grid, spec):
    """Adiciona estatísticas de vizinhança nos pontos de treino."""
    from scipy.ndimage import uniform_filter

    windows = [int(w) for w in cfg["pixel_model"].get("focal_windows_px", [])]
    stats = cfg["pixel_model"].get("focal_stats", [])
    if not windows or not stats:
        return df

    rows, cols = df["_row"].values.astype(int), df["_col"].values.astype(int)
    for name, path, band, cat in spec:
        if cat or name not in focal_enabled_features(spec, cfg):
            continue
        with warped(path, grid, False) as v:
            arr = v.read(band).astype(np.float32)
        valid = np.isfinite(arr)
        arr0 = np.where(valid, arr, 0.0).astype(np.float32)
        cnt = valid.astype(np.float32)
        for w in windows:
            mean_num = uniform_filter(arr0, size=w, mode="nearest")
            mean_den = uniform_filter(cnt, size=w, mode="nearest")
            mean = np.divide(mean_num, mean_den, out=np.full_like(mean_num, np.nan), where=mean_den > 0)
            if "mean" in stats:
                df[f"{name}_w{w}_mean"] = mean[rows, cols]
            if "std" in stats:
                sq_num = uniform_filter(arr0 * arr0, size=w, mode="nearest")
                sq_mean = np.divide(sq_num, mean_den, out=np.full_like(sq_num, np.nan), where=mean_den > 0)
                var = np.maximum(sq_mean - mean * mean, 0.0)
                df[f"{name}_w{w}_std"] = np.sqrt(var)[rows, cols]
        logger.info(f"  vizinhança: {name}")
    return df


def extend_spec_with_focal(spec, cfg):
    windows = [int(w) for w in cfg["pixel_model"].get("focal_windows_px", [])]
    stats = cfg["pixel_model"].get("focal_stats", [])
    focal_names = set(focal_enabled_features(spec, cfg))
    extended = list(spec)
    for name, path, band, cat in spec:
        if not cat and name in focal_names:
            for fname in _focal_feature_names(name, windows, stats):
                extended.append((fname, path, band, False))
    return extended


# ─────────────────────────────────────────────────────────────────────────────
# Amostragem do dataset de treino
# ─────────────────────────────────────────────────────────────────────────────
def sample_training(cfg, grid, spec, work: Path):
    pm = cfg["pixel_model"]
    rng = np.random.default_rng(pm["random_state"])
    H, W = grid["height"], grid["width"]
    t = grid["transform"]

    label_geoms = label_geometries(cfg)
    nucleo = label_geoms.get("nucleo")
    interpretado = label_geoms.get("interpretado")
    negativo = label_geoms.get("negativo")
    pos_parts = [g["geometry"] for g in [nucleo, interpretado] if g]
    if not pos_parts:
        pos_parts = [g["geometry"] for g in label_geoms.values() if int(g["label"]) == 1]
    pos_geom = _safe_union(pos_parts)
    excl_m = cfg["deposits"]["negative_sampling"]["min_dist_from_deposit_m"]
    excl_geom = pos_geom.buffer(excl_m)

    logger.info("Rasterizando zonas positiva/negativa na grade mestre...")
    pos_mask = rasterize([(pos_geom, 1)], out_shape=(H, W), transform=t,
                         fill=0, dtype="uint8", all_touched=True).astype(bool)
    nucleo_mask = rasterize([(nucleo["geometry"], 1)], out_shape=(H, W), transform=t,
                            fill=0, dtype="uint8", all_touched=True).astype(bool) if nucleo else np.zeros((H, W), bool)
    interp_mask = rasterize([(interpretado["geometry"], 1)], out_shape=(H, W), transform=t,
                            fill=0, dtype="uint8", all_touched=True).astype(bool) if interpretado else np.zeros((H, W), bool)
    explicit_neg_mask = rasterize([(negativo["geometry"], 1)], out_shape=(H, W), transform=t,
                                  fill=0, dtype="uint8", all_touched=True).astype(bool) if negativo else np.zeros((H, W), bool)
    excl_mask = rasterize([(excl_geom, 1)], out_shape=(H, W), transform=t,
                          fill=0, dtype="uint8", all_touched=True).astype(bool)

    # máscara de dado válido (S2 presente)
    s2_spec = next(s for s in spec if s[0] == "s2_B02")
    with warped(s2_spec[1], grid, False) as v:
        b = v.read(1)
    valid = np.isfinite(b) & (b != grid["nodata"]) & (b != 0)
    logger.info(f"Pixels válidos (S2): {valid.sum():,} de {H*W:,}")

    nucleo_candidates = nucleo_mask & valid
    interp_candidates = interp_mask & valid & ~nucleo_mask
    hard_neg_candidates = explicit_neg_mask & valid & ~pos_mask
    bg_neg_candidates = valid & ~excl_mask & ~explicit_neg_mask
    pos_count = int((nucleo_candidates | interp_candidates).sum())
    logger.info(
        f"Candidatos: {int(nucleo_candidates.sum()):,} nucleo | "
        f"{int(interp_candidates.sum()):,} interpretado | "
        f"{int(hard_neg_candidates.sum()):,} neg difíceis | "
        f"{int(bg_neg_candidates.sum()):,} neg background"
    )
    if pos_count == 0:
        raise SystemExit("Nenhum pixel positivo válido.")

    n_pos_target = min(pm["n_pos"], pos_count)
    n_nuc = min(int(round(n_pos_target * 0.60)), int(nucleo_candidates.sum()))
    n_int = min(n_pos_target - n_nuc, int(interp_candidates.sum()))
    if n_nuc + n_int < n_pos_target:
        n_nuc = min(int(nucleo_candidates.sum()), n_nuc + (n_pos_target - n_nuc - n_int))
    nuc_sel = _sample_flat_indices(nucleo_candidates, n_nuc, rng)
    int_sel = _sample_flat_indices(interp_candidates, n_int, rng)

    n_pos = len(nuc_sel) + len(int_sel)
    n_neg_target = pm["neg_ratio"] * n_pos
    hard_sel = _sample_flat_indices(hard_neg_candidates, n_neg_target, rng)
    n_bg = n_neg_target - len(hard_sel)
    if not pm.get("include_background_negatives", True):
        n_bg = 0
    bg_sel = _sample_flat_indices(bg_neg_candidates, n_bg, rng)

    flat = np.concatenate([nuc_sel, int_sel, hard_sel, bg_sel])
    y = np.concatenate([
        np.ones(len(nuc_sel), "int8"),
        np.ones(len(int_sel), "int8"),
        np.zeros(len(hard_sel), "int8"),
        np.zeros(len(bg_sel), "int8"),
    ])
    source = (
        ["nucleo"] * len(nuc_sel)
        + ["interpretado"] * len(int_sel)
        + ["negativo"] * len(hard_sel)
        + ["background"] * len(bg_sel)
    )
    weights = np.array(
        [nucleo["weight"] if nucleo else 1.0] * len(nuc_sel)
        + [interpretado["weight"] if interpretado else 0.45] * len(int_sel)
        + [negativo["weight"] if negativo else 1.0] * len(hard_sel)
        + [float(pm.get("background_negative_weight", 0.35))] * len(bg_sel),
        dtype=np.float32,
    )
    rows, cols = np.divmod(flat, W)
    # coords no centro do pixel
    xs = t.c + (cols + 0.5) * t.a
    ys = t.f + (rows + 0.5) * t.e
    coords = list(zip(xs, ys))
    logger.info(
        f"Amostras: {len(nuc_sel)} nucleo | {len(int_sel)} interpretado | "
        f"{len(hard_sel)} negativo difícil | {len(bg_sel)} background"
    )

    # amostrar cada camada nos pontos
    data = {}
    for name, path, band, _cat in spec:
        with warped(path, grid, _cat) as v:
            vals = np.array([x[band-1] for x in v.sample(coords)], dtype=np.float32)
        data[name] = vals
        logger.info(f"  amostrado: {name}")
    df = pd.DataFrame(data)
    df["label"] = y
    df["label_source"] = source
    df["sample_weight"] = weights
    df["x"] = xs; df["y"] = ys
    df["_row"] = rows
    df["_col"] = cols
    df = add_focal_features_to_points(df, cfg, grid, spec)
    df = df.drop(columns=["_row", "_col"])
    block_m = pm.get("spatial_block_m", 30000)
    df["block_id"] = (
        np.floor(df["x"].values / block_m).astype(int).astype(str)
        + "_"
        + np.floor(df["y"].values / block_m).astype(int).astype(str)
    )
    out_csv = work / "train_dataset.csv"
    df.to_csv(out_csv, index=False)
    logger.success(f"Dataset salvo: {out_csv} | {df.shape}")
    logger.info(f"Fontes: {df['label_source'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Treino + seleção de modelo (CV)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_model_matrix(df: pd.DataFrame, raw_features: list, cat_features: list,
                         model_features: list = None):
    """Transforma features cruas em matriz do modelo com one-hot para categorias."""
    parts = []
    cont_features = [f for f in raw_features if f not in cat_features]
    if cont_features:
        parts.append(df[cont_features].astype(np.float32).fillna(-9999))
    if cat_features:
        cats = df[cat_features].copy()
        for col in cat_features:
            cats[col] = cats[col].where(pd.notna(cats[col]), "missing").astype(str)
        parts.append(pd.get_dummies(cats, columns=cat_features, prefix=cat_features, dtype=np.uint8))

    Xdf = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    if model_features is not None:
        Xdf = Xdf.reindex(columns=model_features, fill_value=0)
    return Xdf.astype(np.float32)


def train_models(cfg, df, spec, work: Path):
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
    from lightgbm import LGBMClassifier

    spec_model = extend_spec_with_focal(spec, cfg)
    raw_features = [s[0] for s in spec_model]
    cat_features = [s[0] for s in spec if s[3]]
    Xdf = prepare_model_matrix(df, raw_features, cat_features)
    model_features = list(Xdf.columns)
    X = Xdf.values
    y = df["label"].values
    sample_weight = df["sample_weight"].values if "sample_weight" in df.columns else None
    pm = cfg["pixel_model"]
    groups = df.get("block_id", pd.Series(np.arange(len(df)))).astype(str).values
    n_splits = min(pm["cv_folds"], len(np.unique(groups)))
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=pm["random_state"])
        cv_groups = groups
        logger.info(f"CV espacial: StratifiedGroupKFold n={n_splits} | blocos={len(np.unique(groups))}")
    else:
        cv = StratifiedKFold(pm["cv_folds"], shuffle=True, random_state=pm["random_state"])
        cv_groups = None
        logger.warning("Poucos blocos espaciais — usando StratifiedKFold")

    nj = pm.get("n_jobs", 4)
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, n_jobs=nj, class_weight="balanced",
            random_state=pm["random_state"]),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05,
            random_state=pm["random_state"]),
        "LightGBM": LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63, n_jobs=nj,
            class_weight="balanced", random_state=pm["random_state"], verbose=-1),
    }

    results = {}
    for name, model in candidates.items():
        try:
            # cv sequencial (n_jobs=1) — o paralelismo fica DENTRO do modelo (nj)
            fit_params = {"sample_weight": sample_weight} if sample_weight is not None else None
            scores = cross_val_score(model, X, y, cv=cv, groups=cv_groups,
                                     scoring="roc_auc", n_jobs=1, fit_params=fit_params)
            mean, std = float(np.nanmean(scores)), float(np.nanstd(scores))
            if not np.isfinite(mean):
                raise ValueError("AUC não finito")
            results[name] = (mean, std)
            logger.info(f"  {name:14} AUC = {mean:.4f} ± {std:.4f}")
        except Exception as e:
            logger.warning(f"  {name}: falhou ({e})")

    best = max(results, key=lambda k: results[k][0])
    if pm.get("prefer_fast_model", False) and "LightGBM" in results:
        tol = float(pm.get("fast_model_auc_tolerance", 0.03))
        if results[best][0] - results["LightGBM"][0] <= tol:
            logger.info(
                f"Preferindo LightGBM para inferência full-AOI "
                f"(delta AUC={results[best][0] - results['LightGBM'][0]:.4f} <= {tol:.4f})"
            )
            best = "LightGBM"
    logger.success(f"Melhor modelo: {best} (AUC {results[best][0]:.4f})")
    if sample_weight is not None:
        best_model = candidates[best].fit(X, y, sample_weight=sample_weight)
    else:
        best_model = candidates[best].fit(X, y)

    mdir = Path(cfg["paths"]["models"]) / "pixel"
    mdir.mkdir(parents=True, exist_ok=True)
    pickle.dump({
        "model": best_model,
        "features": model_features,
        "raw_features": raw_features,
        "cat_features": cat_features,
        "focal_windows_px": cfg["pixel_model"].get("focal_windows_px", []),
        "focal_stats": cfg["pixel_model"].get("focal_stats", []),
        "focal_feature_prefixes": cfg["pixel_model"].get("focal_feature_prefixes", []),
        "name": best,
    }, open(mdir / "best_pixel_model.pkl", "wb"))
    pd.DataFrame([{"modelo": k, "auc_mean": v[0], "auc_std": v[1]}
                  for k, v in results.items()]).to_csv(mdir / "cv_scores.csv", index=False)
    logger.success(f"Modelo salvo: {mdir/'best_pixel_model.pkl'}")
    return best_model, model_features


# ─────────────────────────────────────────────────────────────────────────────
# Inferência full-AOI em blocos → raster 30 m
# ─────────────────────────────────────────────────────────────────────────────
def predict_aoi(cfg, grid, spec, work: Path):
    from scipy.ndimage import uniform_filter
    mdir = Path(cfg["paths"]["models"]) / "pixel"
    saved = pickle.load(open(mdir / "best_pixel_model.pkl", "rb"))
    model = saved["model"]
    model_features = saved["features"]
    raw_features = saved.get("raw_features", [s[0] for s in spec])
    cat_features = saved.get("cat_features", [s[0] for s in spec if s[3]])
    windows = [int(w) for w in saved.get("focal_windows_px", cfg["pixel_model"].get("focal_windows_px", []))]
    stats = saved.get("focal_stats", cfg["pixel_model"].get("focal_stats", []))
    focal_names = set(focal_enabled_features(spec, cfg))
    logger.info(f"Aplicando '{saved['name']}' à AOI ({grid['height']}×{grid['width']})")

    H, W = grid["height"], grid["width"]
    block = cfg["pixel_model"]["block_rows"]
    nodata = grid["nodata"]

    out_dir = Path(cfg["paths"]["output"]) / "geotiff" / "pixel"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "favorabilidade_porfiro_PIXEL_AOI.tif"
    profile = dict(driver="GTiff", dtype="float32", count=1, width=W, height=H,
                   crs=grid["crs"], transform=grid["transform"], nodata=nodata,
                   compress="lzw", tiled=True, blockxsize=512, blockysize=512)

    # abrir todos os WarpedVRT uma vez
    vrts = [(name, warped(path, grid, cat), band) for name, path, band, cat in spec]
    try:
        with rasterio.open(out_path, "w", **profile) as dst:
            for r0 in range(0, H, block):
                h = min(block, H - r0)
                win = Window(0, r0, W, h)
                halo = max([int(w) for w in windows], default=1) // 2
                rr0 = max(0, r0 - halo)
                rr1 = min(H, r0 + h + halo)
                read_win = Window(0, rr0, W, rr1 - rr0)
                crop0 = r0 - rr0
                crop1 = crop0 + h
                raw_names = [name for name, _path, _band, _cat in spec]
                block_data = {}
                for i, (name, v, band) in enumerate(vrts):
                    arr = v.read(band, window=read_win).astype(np.float32)
                    block_data[name] = arr[crop0:crop1, :]
                # válido = S2 B02 presente
                b02 = block_data["s2_B02"].ravel()
                valid = np.isfinite(b02) & (b02 != nodata) & (b02 != 0)
                proba = np.full(h * W, nodata, dtype=np.float32)
                vidx = np.flatnonzero(valid)
                if vidx.size:
                    raw_df = pd.DataFrame(
                        {name: block_data[name].ravel()[vidx] for name in raw_names}
                    )
                    for name in focal_names:
                        full_arr = next(v for n, v, _band in vrts if n == name).read(
                            next(b for n, _v, b in vrts if n == name),
                            window=read_win,
                        ).astype(np.float32)
                        arr = full_arr
                        ok = np.isfinite(arr)
                        arr0 = np.where(ok, arr, 0.0).astype(np.float32)
                        cnt = ok.astype(np.float32)
                        for ww in windows:
                            mean_num = uniform_filter(arr0, size=ww, mode="nearest")
                            mean_den = uniform_filter(cnt, size=ww, mode="nearest")
                            mean = np.divide(
                                mean_num, mean_den,
                                out=np.full_like(mean_num, np.nan),
                                where=mean_den > 0,
                            )
                            mean = mean[crop0:crop1, :]
                            if "mean" in stats:
                                raw_df[f"{name}_w{ww}_mean"] = mean.ravel()[vidx]
                            if "std" in stats:
                                sq_num = uniform_filter(arr0 * arr0, size=ww, mode="nearest")
                                sq_mean = np.divide(
                                    sq_num, mean_den,
                                    out=np.full_like(sq_num, np.nan),
                                    where=mean_den > 0,
                                )
                                sq_mean = sq_mean[crop0:crop1, :]
                                var = np.maximum(sq_mean - mean * mean, 0.0)
                                raw_df[f"{name}_w{ww}_std"] = np.sqrt(var).ravel()[vidx]
                        del full_arr
                    Xv = prepare_model_matrix(
                        raw_df, raw_features, cat_features, model_features
                    ).values
                    # predição em sub-lotes → limita pico de RAM
                    pb = cfg["pixel_model"].get("predict_batch", 300000)
                    out = np.empty(vidx.size, dtype=np.float32)
                    for j in range(0, vidx.size, pb):
                        out[j:j+pb] = model.predict_proba(Xv[j:j+pb])[:, 1]
                    proba[vidx] = out
                    del Xv
                dst.write(proba.reshape(h, W), 1, window=win)
                del block_data, proba
                logger.info(f"  bloco linhas {r0}-{r0+h} | válidos {int(valid.sum()):,}")
    finally:
        for _, v, _ in vrts:
            v.close()
    logger.success(f"Mapa pixel salvo: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
def run(config_path=None, step="all"):
    cfg = load_config(config_path)
    grid = master_grid(cfg)
    work = Path(cfg["paths"]["data"]) / "pixel_model"
    work.mkdir(parents=True, exist_ok=True)
    vrt_dir = work / "vrt"

    if step in ("vrt", "all"):
        logger.info("== Construindo VRTs alinhados ==")
        build_vrts(cfg, vrt_dir)

    layers = {k: Path(v) for k, v in json.load(open(vrt_dir / "layers.json")).items()}
    spec = feature_spec(layers)
    logger.info(f"Features ({len(spec)}): {[s[0] for s in spec]}")

    if step in ("train", "all"):
        logger.info("== Amostragem + treino + CV ==")
        df = sample_training(cfg, grid, spec, work)
        train_models(cfg, df, spec, work)

    if step in ("predict", "all"):
        logger.info("== Inferência full-AOI (30 m) ==")
        predict_aoi(cfg, grid, spec, work)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--step", choices=["vrt", "train", "predict", "all"], default="all")
    args = ap.parse_args()
    run(args.config, args.step)
