"""
Fase 5b · Treino do Ensemble Multi-modal.

Combina:
  - Embeddings Prithvi (768-d) — features espaciais/espectrais
  - Features tabulares (geoquímica, DEM, índices SEGEMAR)

Modelo: LightGBM binário por tipo de depósito
  Priority: pórfiro Cu-Mo
  Validação: split espacial (leave-one-block-out)
  Otimização: Optuna (50 trials)
  Interpretabilidade: SHAP values

Saída: modelos .pkl em data/07_MODELS/
"""

import sys
import time
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.raster_utils import load_config

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add(Path("D:/argentina/logs/train_{time:YYYY-MM-DD}.log"), level="DEBUG", rotation="50 MB")


# ─────────────────────────────────────────────────────────────────────────────
# Construção do dataset de features
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings(labels_df: pd.DataFrame) -> np.ndarray:
    """Carrega embeddings Prithvi para cada patch no DataFrame de labels."""
    embeddings = []
    cache: Dict[str, np.ndarray] = {}

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Carregando embeddings"):
        npz_path = row["npz_path"]
        idx = int(row["npz_idx"])

        if npz_path not in cache:
            data = np.load(npz_path, allow_pickle=True)
            cache[npz_path] = data["embeddings"]

        emb = cache[npz_path][idx].astype(np.float32)
        embeddings.append(emb)

    return np.vstack(embeddings)


def sample_raster_at_points(
    raster_path: Path,
    cx: np.ndarray,
    cy: np.ndarray,
    nodata: float = -9999,
) -> np.ndarray:
    """Amostra valor(es) de raster nas coordenadas centrais dos patches."""
    import rasterio

    if not raster_path.exists():
        return np.full(len(cx), np.nan, dtype=np.float32)

    with rasterio.open(raster_path) as src:
        coords = list(zip(cx, cy))
        try:
            vals = np.array(list(src.sample(coords)), dtype=np.float32)
            if vals.ndim == 2:
                vals = vals.mean(axis=1)
        except Exception:
            vals = np.full(len(cx), np.nan, dtype=np.float32)
        vals = np.where(vals == nodata, np.nan, vals)
    return vals


def sample_raster_patch_stats(
    raster_path: Path,
    cx: np.ndarray,
    cy: np.ndarray,
    window_px: int,
    stats: List[str],
    nodata: float = -9999,
) -> Dict[str, np.ndarray]:
    """Amostra estatísticas focais em uma janela quadrada centrada no patch."""
    import rasterio
    from scipy.ndimage import uniform_filter, maximum_filter

    result = {s: np.full(len(cx), np.nan, dtype=np.float32) for s in stats}
    if not raster_path.exists():
        return result

    with rasterio.open(raster_path) as src:
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
                var = np.maximum(mean2 - mean * mean, 0.0)
                focal["std"] = np.sqrt(var).astype(np.float32)
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


def build_tabular_features(
    labels_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Amostra todos os rasters SEGEMAR e índices espectrais nas coordenadas dos patches.
    Retorna DataFrame com features tabulares.
    """
    cx = labels_df["center_x"].values
    cy = labels_df["center_y"].values
    nodata = config["output"]["nodata"]

    raster_dir = Path(config["paths"]["rasters"])
    feat_dict = {}
    train_cfg = config.get("training", {})
    use_geochem = train_cfg.get("use_geochem_features", False)
    use_segemar = train_cfg.get("use_segemar_features", True)
    use_terrain = train_cfg.get("use_terrain_features", True)
    use_s2_indices = train_cfg.get("use_s2_index_features", True)
    patch_stats = train_cfg.get("tabular_patch_stats", ["mean", "std", "max"])
    window_px = int(config["prithvi"]["patch_size"])

    # Geoquímica
    geochem_dir = raster_dir / "segemar" / "geochem"
    if use_geochem and geochem_dir.exists():
        for tif in sorted(geochem_dir.glob("*.tif")):
            name = tif.stem.replace("geochem_", "geo_")
            feat_dict[name] = sample_raster_at_points(tif, cx, cy, nodata)
            logger.debug(f"Feature: {name}")

    # SEGEMAR vetorial rasterizado
    segemar_dir = raster_dir / "segemar"
    if use_segemar:
        for tif in sorted(segemar_dir.glob("*.tif")):
            name = tif.stem
            feat_dict[f"seg_{name}"] = sample_raster_at_points(tif, cx, cy, nodata)
            logger.debug(f"Feature: seg_{name}")

    # DEM derivadas — múltiplos tiles: estatísticas na janela do patch
    terrain_dir = raster_dir / "terrain"
    if use_terrain and terrain_dir.exists():
        terrain_accum: Dict[str, List[np.ndarray]] = {}
        for tif in sorted(terrain_dir.rglob("*.tif")):
            name = tif.stem.split("_")[0]  # slope, aspect, tpi, curvature
            stats_dict = sample_raster_patch_stats(tif, cx, cy, window_px, patch_stats, nodata)
            for stat_name, values in stats_dict.items():
                key = f"dem_{name}_{stat_name}"
                terrain_accum.setdefault(key, []).append(values)
        for key, arrs in terrain_accum.items():
            stacked = np.vstack(arrs)  # (n_tiles, n_points)
            feat_dict[key] = np.nanmean(stacked, axis=0).astype(np.float32)

    # Índices S2 — estatísticas na janela do patch
    s2_idx_dir = raster_dir / "indices" / "s2"
    if use_s2_indices and s2_idx_dir.exists():
        for tif in sorted(s2_idx_dir.rglob("*.tif")):
            name = "_".join(tif.stem.split("_")[:2])  # s2_ClayIndex
            stats_dict = sample_raster_patch_stats(tif, cx, cy, window_px, patch_stats, nodata)
            for stat_name, values in stats_dict.items():
                key = f"idx_{name}_{stat_name}"
                if key not in feat_dict:
                    feat_dict[key] = []
                feat_dict[key].append(values)

    # Médias de índices duplicados (múltiplas cenas/bboxes)
    for k in list(feat_dict.keys()):
        if isinstance(feat_dict[k], list):
            stacked = np.vstack(feat_dict[k])
            feat_dict[k] = np.nanmean(stacked, axis=0)

    feat_df = pd.DataFrame(feat_dict)
    logger.info(f"Features tabulares: {feat_df.shape[1]} colunas, {feat_df.shape[0]} patches")
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# Treino LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    n_estimators: int = 1000,
    early_stopping: int = 50,
    w_train: np.ndarray = None,
    w_val: np.ndarray = None,
):
    """Treina LightGBM com early stopping."""
    import lightgbm as lgb

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(early_stopping, verbose=False),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return model


def optimize_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    base_params: dict = None,
    w_train: np.ndarray = None,
    w_val: np.ndarray = None,
) -> dict:
    """Otimiza hiperparâmetros LightGBM via Optuna."""
    import optuna
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "device": "cpu",
            "n_jobs": -1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight": "balanced",
        }

        model = lgb.train(
            params,
            lgb.Dataset(X_train, label=y_train, weight=w_train),
            num_boost_round=300,
            valid_sets=[lgb.Dataset(X_val, label=y_val, weight=w_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val)
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    logger.info(f"Melhor AUC Optuna: {study.best_value:.4f}")
    logger.info(f"Melhores params: {best}")

    best["objective"] = "binary"
    best["metric"] = "auc"
    best["verbose"] = -1
    best["device"] = "cpu"
    best["class_weight"] = "balanced"
    return best


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Avalia modelo no conjunto de teste."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        classification_report, confusion_matrix
    )

    preds_proba = model.predict(X_test)
    preds_binary = (preds_proba > 0.5).astype(int)

    n_pos = int(y_test.sum())
    if len(np.unique(y_test)) < 2:
        logger.warning(f"Test set com apenas uma classe ({n_pos} positivos) — AUC não calculado")
        auc_roc, auc_pr = float("nan"), float("nan")
    else:
        auc_roc = round(roc_auc_score(y_test, preds_proba), 4)
        auc_pr = round(average_precision_score(y_test, preds_proba), 4)

    metrics = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "n_test": len(y_test),
        "n_positive": n_pos,
    }
    logger.info(f"AUC-ROC: {metrics['auc_roc']} | AUC-PR: {metrics['auc_pr']}")
    if len(np.unique(y_test)) >= 2:
        logger.info(f"\n{classification_report(y_test, preds_binary, target_names=['Neg','Pos'])}")
    return metrics


def compute_shap(model, X: np.ndarray, feature_names: List[str], output_path: Path):
    """Computa e salva SHAP values para interpretabilidade."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:min(500, len(X))])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        importance = pd.DataFrame({
            "feature": feature_names,
            "shap_mean_abs": np.abs(shap_values).mean(axis=0),
        }).sort_values("shap_mean_abs", ascending=False)
        importance.to_csv(output_path / "shap_importance.csv", index=False)
        logger.info(f"SHAP salvo: {output_path / 'shap_importance.csv'}")
        logger.info(f"Top 10 features:\n{importance.head(10).to_string()}")
    except Exception as e:
        logger.warning(f"SHAP falhou: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de treino
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str = None, deposit_type: str = None, force: bool = False):
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FASE 5b · Treino do Ensemble")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    labels_dir = Path(cfg["paths"]["models"]).parent / "labels"
    models_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = labels_dir / "labels.csv"
    if not labels_csv.exists():
        logger.error(f"Labels não encontrados: {labels_csv}")
        logger.error("Execute primeiro: python 05_labels/labels.py")
        return

    labels_df = pd.read_csv(labels_csv)
    logger.info(f"Labels carregados: {len(labels_df)} patches")
    has_weights = "sample_weight" in labels_df.columns
    if has_weights:
        logger.info(
            "Pesos de amostra: "
            f"{labels_df.groupby('label_source')['sample_weight'].mean().to_dict()}"
        )

    # Tipos de depósito a treinar (por padrão todos, priority_model primeiro)
    priority = cfg["training"]["priority_model"]
    all_types = ["porfiro", "skarn", "epitermal", "manto"]
    if deposit_type:
        all_types = [deposit_type]
    else:
        # Priority first
        all_types = [priority] + [t for t in all_types if t != priority]

    # Carregar embeddings (shared entre todos os modelos)
    logger.info("Carregando embeddings Prithvi...")
    emb_matrix = load_embeddings(labels_df)
    logger.info(f"Embeddings: {emb_matrix.shape}")

    # Construir features tabulares
    logger.info("Amostando features tabulares...")
    tabular_df = build_tabular_features(labels_df, cfg)

    # Combinar features
    X_all = np.hstack([
        emb_matrix,
        tabular_df.fillna(0).values,
    ])
    feature_names = (
        [f"emb_{i}" for i in range(emb_matrix.shape[1])]
        + list(tabular_df.columns)
    )
    logger.info(f"Feature matrix: {X_all.shape}")

    # Treinar modelo para cada tipo de depósito
    results = {}

    for dtype in all_types:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Treinando modelo: {dtype.upper()}")
        logger.info(f"{'─'*50}")

        # Labels binários para este tipo
        if dtype == "porfiro":
            # Inclui pórfiro + amostras (positivos do tipo pórfiro)
            y_all = (labels_df["deposit_type"].isin(["porfiro"])).astype(int).values
        else:
            y_all = (labels_df["deposit_type"] == dtype).astype(int).values

        n_pos = y_all.sum()
        if n_pos < 5:
            logger.warning(f"Poucos positivos para {dtype} ({n_pos}) — pulando")
            continue
        logger.info(f"Positivos: {n_pos} | Negativos: {(y_all==0).sum()}")

        # Split
        train_mask = labels_df["split"] == "train"
        val_mask = labels_df["split"] == "val"
        test_mask = labels_df["split"] == "test"

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_vl, y_vl = X_all[val_mask], y_all[val_mask]
        X_te, y_te = X_all[test_mask], y_all[test_mask]
        w_tr = labels_df.loc[train_mask, "sample_weight"].values.astype(np.float32) if has_weights else None
        w_vl = labels_df.loc[val_mask, "sample_weight"].values.astype(np.float32) if has_weights else None

        # SMOTE para balancear classes no treino (evita viés para negativos)
        if has_weights and cfg["training"].get("use_smote", False):
            logger.info("sample_weight presente — SMOTE ignorado para preservar pesos dos labels")
        elif cfg["training"].get("use_smote", False):
            try:
                from imblearn.over_sampling import SMOTE
                k = min(cfg["training"].get("smote_k_neighbors", 5), int(y_tr.sum()) - 1)
                if k >= 1:
                    logger.info(f"SMOTE k={k}: {y_tr.sum()} positivos → balanceando...")
                    X_tr, y_tr = SMOTE(
                        k_neighbors=k,
                        random_state=cfg["training"]["random_state"],
                    ).fit_resample(X_tr, y_tr)
                    w_tr = None
                    logger.info(f"Após SMOTE: {y_tr.sum()} pos / {(y_tr==0).sum()} neg")
            except ImportError:
                logger.warning("imbalanced-learn não instalado — SMOTE ignorado (pip install imbalanced-learn)")
            except Exception as e:
                logger.warning(f"SMOTE falhou ({e}) — continuando sem")

        # Otimizar hiperparâmetros (só pórfiro para economizar tempo)
        if dtype == priority:
            logger.info("Otimizando hiperparâmetros com Optuna...")
            best_params = optimize_lgbm(
                X_tr, y_tr, X_vl, y_vl,
                n_trials=cfg["training"]["n_trials_optuna"],
                base_params=cfg["training"]["lgbm_params_base"],
                w_train=w_tr,
                w_val=w_vl,
            )
        else:
            best_params = cfg["training"]["lgbm_params_base"].copy()

        # Skip se modelo já treinado e force=False
        model_path_check = models_dir / dtype / "lgbm_model.pkl"
        if model_path_check.exists() and not force:
            logger.info(f"Modelo {dtype} já existe — pulando (use force=True para retreinar)")
            continue

        # Treinar modelo final
        logger.info("Treinando modelo final...")
        model = train_lgbm(X_tr, y_tr, X_vl, y_vl, best_params, w_train=w_tr, w_val=w_vl)

        # Avaliar
        metrics = evaluate_model(model, X_te, y_te)
        metrics["deposit_type"] = dtype
        metrics["best_iteration"] = model.best_iteration
        results[dtype] = metrics

        # Salvar modelo
        model_dir = models_dir / dtype
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "lgbm_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        # Salvar feature names para alinhamento na inferência
        feat_names_path = model_dir / "feature_names.json"
        with open(feat_names_path, "w") as f:
            json.dump(feature_names, f)
        logger.success(f"Modelo salvo: {model_path}")

        # Metadados
        meta = {
            "deposit_type": dtype,
            "metrics": metrics,
            "best_params": {k: v for k, v in best_params.items() if isinstance(v, (int, float, str, bool))},
            "n_features": X_all.shape[1],
            "n_emb_features": emb_matrix.shape[1],
            "n_tabular_features": tabular_df.shape[1],
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # SHAP
        compute_shap(model, X_vl, feature_names, model_dir)

    # Resumo
    logger.info("\n" + "="*50)
    logger.info("RESUMO DOS MODELOS TREINADOS")
    logger.info("="*50)
    for dtype, m in results.items():
        logger.info(f"  {dtype:15s} AUC-ROC={m['auc_roc']} AUC-PR={m['auc_pr']}")

    elapsed = time.time() - t0
    logger.success(f"Fase treino concluída em {elapsed/60:.0f} min")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--type", default=None, help="Tipo específico: porfiro|skarn|epitermal|manto")
    args = parser.parse_args()
    run(config_path=args.config, deposit_type=args.type)
