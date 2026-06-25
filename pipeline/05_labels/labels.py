"""
Fase 5a · Preparação de Labels para Treino.

Combina fontes de ground truth:
  1. Ocorrencias_de_Cobre_Aoi (103 pontos classificados por tipo)
  2. Polígonos "amostra_*" (áreas positivas para pórfiro/alteração)
  3. Negative sampling espacialmente estratificado

Para cada patch de embedding, verifica interseção com positivos/negativos.
Saída: CSV com patch_id | center_x | center_y | label | deposit_type | split
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
from sklearn.model_selection import train_test_split
from loguru import logger

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.raster_utils import load_config

logger.remove()
logger.add(sys.stderr, level="INFO", format="<blue>{time:HH:mm:ss}</blue> | <level>{message}</level>")


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de ground truth
# ─────────────────────────────────────────────────────────────────────────────

# Foco exclusivo em Cu pórfiro — outros tipos mapeados para None (excluídos)
TIPO_MAP = {
    "4b": "porfiro",
    "4c": "porfiro",
}


def load_occurrences(config: dict) -> gpd.GeoDataFrame:
    """Carrega ocorrências de cobre classificadas por tipo."""
    import fiona, pyproj
    from shapely.geometry import shape
    from shapely import wkb as swkb
    from shapely.ops import transform as shp_transform

    gpkg = config["paths"]["amostras_gpkg"]
    layer = config["deposits"]["ocorrencias_layer"]
    target_crs = config["project"]["crs"]

    records = []
    with fiona.open(gpkg, layer=layer) as src:
        transformer = pyproj.Transformer.from_crs(src.crs, target_crs, always_xy=True)
        for feat in src:
            try:
                geom = swkb.loads(shape(feat["geometry"]).wkb)
                geom_t = shp_transform(transformer.transform, geom)
                props = dict(feat["properties"])
                props["geometry"] = geom_t
                records.append(props)
            except Exception:
                pass

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
    gdf = gdf[gdf.geometry.notna()]

    cod_field = config["deposits"]["tipo_field"]
    gdf["deposit_type"] = gdf[cod_field].map(TIPO_MAP).fillna("outro")

    buffer_cfg = config["deposits"]["buffer_m"]
    gdf["buffer_m"] = gdf["deposit_type"].map(buffer_cfg).fillna(500)
    gdf["geometry_buffered"] = gdf.apply(
        lambda r: r.geometry.buffer(r["buffer_m"]), axis=1
    )
    logger.info(f"Ocorrências carregadas: {len(gdf)} | tipos: {gdf['deposit_type'].value_counts().to_dict()}")
    return gdf


def load_amostras(config: dict) -> gpd.GeoDataFrame:
    """Carrega shapefiles de amostras de campo (já em EPSG:32719, sem reprojeção)."""
    import fiona
    from shapely.geometry import shape
    from shapely import wkb as swkb

    amostras_dir = Path(config["deposits"].get("amostras_dir", ""))
    shapefiles = config["deposits"].get("amostras_shapefiles", [])

    if not shapefiles or not amostras_dir:
        logger.warning("amostras_dir ou amostras_shapefiles não configurados")
        return gpd.GeoDataFrame()

    frames = []
    for shp_name in shapefiles:
        shp_path = amostras_dir / shp_name
        if not shp_path.exists():
            logger.warning(f"Shapefile não encontrado: {shp_path}")
            continue
        try:
            records = []
            with fiona.open(str(shp_path)) as src:
                for feat in src:
                    try:
                        geom = shape(feat["geometry"])
                        # Reconstrói via WKB para garantir compatibilidade shapely
                        geom = swkb.loads(geom.wkb)
                        if geom and not geom.is_empty:
                            records.append({"geometry": geom})
                    except Exception:
                        pass
            if records:
                gdf = gpd.GeoDataFrame(records, geometry="geometry",
                                       crs=config["project"]["crs"])
                gdf["deposit_type"] = "porfiro"
                gdf["source"] = shp_name
                frames.append(gdf[["geometry", "deposit_type", "source"]])
                logger.info(f"{shp_name}: {len(gdf)} polígonos")
        except Exception as e:
            logger.warning(f"{shp_name} falhou: {e}")

    if not frames:
        logger.warning("Nenhuma amostra carregada")
        return gpd.GeoDataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info(f"Total amostras: {len(result)} polígonos de {len(frames)} arquivos")
    return result


def load_label_sets(config: dict) -> Dict[str, Dict]:
    """Carrega conjuntos explícitos: nucleo, interpretado e negativo."""
    import fiona
    from shapely.geometry import shape
    from shapely import wkb as swkb

    amostras_dir = Path(config["deposits"].get("amostras_dir", ""))
    sets_cfg = config["deposits"].get("label_sets", {})
    target_crs = config["project"]["crs"]
    result = {}

    for set_name, set_cfg in sets_cfg.items():
        shp_path = amostras_dir / set_cfg["file"]
        if not shp_path.exists():
            logger.warning(f"Label set não encontrado: {shp_path}")
            continue
        records = []
        try:
            with fiona.open(str(shp_path)) as src:
                for feat in src:
                    try:
                        geom = shape(feat["geometry"])
                        geom = swkb.loads(geom.wkb)
                        if geom and not geom.is_empty:
                            records.append({"geometry": geom})
                    except Exception:
                        pass
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
            result[set_name] = {
                "gdf": gdf,
                "label": int(set_cfg["label"]),
                "weight": float(set_cfg.get("weight", 1.0)),
            }
            logger.info(
                f"Label set {set_name}: {len(gdf)} geometrias | "
                f"label={set_cfg['label']} peso={set_cfg.get('weight', 1.0)}"
            )
        except Exception as e:
            logger.warning(f"{set_name} falhou: {e}")

    return result


def build_label_zones(label_sets: Dict[str, Dict]) -> Dict[str, Dict]:
    """Une geometrias de cada conjunto explícito em zonas de label."""
    from shapely.geometry.base import BaseGeometry
    import shapely
    from shapely import wkb as shapely_wkb

    zones = {}
    for set_name, info in label_sets.items():
        geoms = [
            shapely_wkb.loads(g.wkb)
            for g in info["gdf"].geometry.values
            if isinstance(g, BaseGeometry) and not g.is_empty and g.is_valid
        ]
        if not geoms:
            continue
        try:
            zone = shapely.union_all(shapely.from_wkb([g.wkb for g in geoms]))
        except Exception:
            from functools import reduce
            zone = reduce(lambda a, b: a.union(b), geoms)
        zones[set_name] = {**info, "zone": zone}
        logger.info(f"Zona '{set_name}': {zone.area / 1e6:.1f} km²")
    return zones


def build_positive_zone(occ: gpd.GeoDataFrame, amostras: gpd.GeoDataFrame) -> Dict[str, object]:
    """
    Constrói geometrias unificadas de zonas positivas por tipo de depósito.
    Returns dict: {deposit_type: MultiPolygon}
    """
    zones = {}

    from shapely.geometry.base import BaseGeometry

    def _valid_geom(g):
        return isinstance(g, BaseGeometry) and not g.is_empty and g.is_valid

    def _safe_union(geom_list):
        """União robusta: reconstrói via WKB para garantir dtype correto no shapely."""
        import shapely
        from shapely import wkb as shapely_wkb
        # Reconstrói cada geometria via WKB para evitar problemas de dtype entre versões
        clean = []
        for g in geom_list:
            try:
                clean.append(shapely_wkb.loads(g.wkb))
            except Exception:
                pass
        if not clean:
            return None
        try:
            arr = shapely.from_wkb([g.wkb for g in clean])
            return shapely.union_all(arr)
        except Exception:
            # Último recurso: reduce pairwise
            from functools import reduce
            return reduce(lambda a, b: a.union(b), clean)

    # Das ocorrências (bufferizadas) — só tipos definidos no TIPO_MAP (exclui "outro")
    if len(occ) > 0 and "deposit_type" in occ.columns:
        valid_types = set(TIPO_MAP.values())
        for dtype in occ["deposit_type"].unique():
            if dtype not in valid_types:
                continue
            sub = occ[occ["deposit_type"] == dtype]
            polys = [g for g in sub["geometry_buffered"] if _valid_geom(g)]
            if not polys:
                continue
            union = _safe_union(polys)
            zones[dtype] = zones.get(dtype, None)
            zones[dtype] = union if zones[dtype] is None else zones[dtype].union(union)

    # Das amostras (polígonos diretos)
    if len(amostras) > 0:
        valid_geoms = [g for g in amostras.geometry.values if _valid_geom(g)]
        amostra_union = _safe_union(valid_geoms) if valid_geoms else None
        if amostra_union is not None:
            if "porfiro" in zones:
                zones["porfiro"] = zones["porfiro"].union(amostra_union)
            else:
                zones["porfiro"] = amostra_union

    for k, v in zones.items():
        area_km2 = v.area / 1e6
        logger.info(f"Zona positiva '{k}': {area_km2:.1f} km²")

    return zones


def build_exclusion_zone(zones: Dict) -> object:
    """Área que NÃO pode ser amostrada como negativo (muito próxima de positivos)."""
    import shapely
    from shapely import wkb as shapely_wkb
    from functools import reduce
    all_geoms = [g for g in zones.values() if g is not None]
    if not all_geoms:
        from shapely.geometry import Point
        return Point(0, 0).buffer(0)
    try:
        clean = [shapely_wkb.loads(g.wkb) for g in all_geoms]
        arr = shapely.from_wkb([g.wkb for g in clean])
        return shapely.union_all(arr)
    except Exception:
        return reduce(lambda a, b: a.union(b), all_geoms)


# ─────────────────────────────────────────────────────────────────────────────
# Matching patches → labels
# ─────────────────────────────────────────────────────────────────────────────

def assign_labels(
    embeddings_npz: Path,
    positive_zones: Dict[str, object],
    exclusion_zone,
    config: dict,
    negative_ratio: int = 5,
    patch_size_m: float = 6720.0,
    priority_type: str = "porfiro",
    label_zones: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Para cada patch no NPZ, determina se é positivo/negativo por tipo de depósito.
    Retorna DataFrame com colunas: patch_id, cx, cy, label, deposit_type, split
    """
    data = np.load(embeddings_npz, allow_pickle=True)
    ids = data["patch_ids"]
    cx = data["centers_x"]
    cy = data["centers_y"]

    records = []
    half = patch_size_m / 2.0
    label_mode = config["deposits"].get("positive_label_mode", "center")
    min_overlap = float(config["deposits"].get("positive_min_overlap", 0.10))
    if label_mode not in {"center", "overlap", "intersects"}:
        logger.warning(f"positive_label_mode inválido '{label_mode}' — usando center")
        label_mode = "center"
    logger.info(f"Modo de label positivo: {label_mode} | min_overlap={min_overlap:.2f}")

    for i in range(len(ids)):
        patch_box = box(cx[i] - half, cy[i] - half, cx[i] + half, cy[i] + half)
        patch_center = Point(cx[i], cy[i])

        matched_type = None
        matched_overlap = 0.0
        label = 0
        sample_weight = 0.7
        label_source = "background"

        if label_zones:
            # Prioridade conservadora: negativo explícito > núcleo > interpretado.
            for set_name in ["negativo", "nucleo", "interpretado"]:
                info = label_zones.get(set_name)
                if not info:
                    continue
                zone = info["zone"]
                if set_name == "negativo" or label_mode == "center":
                    is_match = zone.contains(patch_center) or zone.touches(patch_center)
                elif label_mode == "overlap":
                    if zone.intersects(patch_box):
                        matched_overlap = zone.intersection(patch_box).area / patch_box.area
                        is_match = matched_overlap >= min_overlap
                    else:
                        is_match = False
                else:
                    is_match = zone.intersects(patch_box)
                if is_match:
                    label = int(info["label"])
                    sample_weight = float(info["weight"])
                    label_source = set_name
                    matched_type = "porfiro" if label == 1 else "negative"
                    break
        else:
            # Para alvos discretos, evitar que um patch de 6.7 km vire positivo
            # só por tocar a borda de uma zona positiva.
            for dtype, zone in positive_zones.items():
                is_match = False
                overlap_fraction = 0.0
                if label_mode == "center":
                    is_match = zone.contains(patch_center) or zone.touches(patch_center)
                elif label_mode == "overlap":
                    if zone.intersects(patch_box):
                        overlap_fraction = zone.intersection(patch_box).area / patch_box.area
                        is_match = overlap_fraction >= min_overlap
                else:
                    is_match = zone.intersects(patch_box)

                if is_match:
                    matched_type = dtype
                    matched_overlap = overlap_fraction
                    label = 1
                    sample_weight = 1.0
                    label_source = dtype
                    if dtype == priority_type:
                        break

        records.append({
            "patch_id": str(ids[i]),
            "center_x": float(cx[i]),
            "center_y": float(cy[i]),
            "label": label,
            "deposit_type": matched_type or "negative",
            "label_source": label_source,
            "sample_weight": sample_weight,
            "positive_overlap": float(matched_overlap),
            "npz_path": str(embeddings_npz),
            "npz_idx": i,
        })

    df = pd.DataFrame(records)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info(f"Patches: {n_pos} positivos | {n_neg} negativos (ratio 1:{n_neg//max(n_pos,1)})")

    # Sub-amostrar negativos para manter ratio
    if n_neg > n_pos * negative_ratio:
        # Filtrar negativos dentro da exclusion zone (muito próximos)
        explicit_neg_df = df[(df["label"] == 0) & (df["label_source"] == "negativo")].copy()
        neg_df = df[(df["label"] == 0) & (df["label_source"] != "negativo")].copy()
        pos_df = df[df["label"] == 1].copy()

        # Excluir negativos dentro da exclusion zone expandida
        min_dist = config["deposits"]["negative_sampling"]["min_dist_from_deposit_m"]
        excl_expanded = exclusion_zone.buffer(min_dist)
        far_neg = neg_df[[
            not Point(r.center_x, r.center_y).within(excl_expanded)
            for _, r in neg_df.iterrows()
        ]]

        n_bg_keep = max(0, n_pos * negative_ratio - len(explicit_neg_df))
        n_keep = min(len(far_neg), n_bg_keep)
        neg_sampled = far_neg.sample(n=n_keep, random_state=42) if n_keep else far_neg.iloc[0:0]
        df = pd.concat([pos_df, explicit_neg_df, neg_sampled], ignore_index=True)
        logger.info(
            f"Após negative sampling: {len(pos_df)} pos | "
            f"{len(explicit_neg_df)} neg explícitos | {len(neg_sampled)} neg background"
        )

    return df


def spatial_train_val_test_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Split por blocos espaciais para reduzir vazamento entre patches vizinhos.
    """
    from sklearn.model_selection import train_test_split

    df = df.copy()
    block_m = 30000
    df["block_id"] = (
        np.floor(df["center_x"].values / block_m).astype(int).astype(str)
        + "_"
        + np.floor(df["center_y"].values / block_m).astype(int).astype(str)
    )
    blocks = df.groupby("block_id")["label"].max().reset_index(name="has_pos")

    if len(blocks) < 3 or blocks["has_pos"].nunique() < 2:
        logger.warning("Poucos blocos para split espacial robusto — usando split aleatório legado")
        pos_df = df[df["label"] == 1].copy()
        neg_df = df[df["label"] == 0].copy()

        def _split_group(grp, ts, vs, seed):
            if len(grp) < 3:
                grp["split"] = "train"
                return grp
            tr, tmp = train_test_split(grp, test_size=ts + vs, random_state=seed)
            if len(tmp) < 2:
                tmp["split"] = "val"
                tr["split"] = "train"
                return pd.concat([tr, tmp])
            relative_val = vs / (ts + vs)
            vl, te = train_test_split(tmp, test_size=relative_val, random_state=seed)
            tr["split"] = "train"
            vl["split"] = "val"
            te["split"] = "test"
            return pd.concat([tr, vl, te])

        df = pd.concat([
            _split_group(pos_df, test_size, val_size, random_state),
            _split_group(neg_df, test_size, val_size, random_state),
        ], ignore_index=True)
    else:
        tr_blocks, tmp_blocks = train_test_split(
            blocks,
            test_size=test_size + val_size,
            random_state=random_state,
            stratify=blocks["has_pos"],
        )
        relative_test = test_size / (test_size + val_size)
        strat = tmp_blocks["has_pos"] if tmp_blocks["has_pos"].nunique() > 1 else None
        vl_blocks, te_blocks = train_test_split(
            tmp_blocks,
            test_size=relative_test,
            random_state=random_state,
            stratify=strat,
        )
        df["split"] = "train"
        df.loc[df["block_id"].isin(vl_blocks["block_id"]), "split"] = "val"
        df.loc[df["block_id"].isin(te_blocks["block_id"]), "split"] = "test"
        logger.info(
            f"Split espacial por blocos: train={len(tr_blocks)} "
            f"val={len(vl_blocks)} test={len(te_blocks)} blocos"
        )

    splits = df["split"].value_counts()
    logger.info(f"Split: train={splits.get('train',0)} val={splits.get('val',0)} test={splits.get('test',0)}")
    return df


def run(config_path: str = None, force: bool = False):
    from loguru import logger
    logger.info("=" * 60)
    logger.info("FASE 5a · Preparação de Labels")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    embed_dir = Path(cfg["paths"]["embeddings"])
    out_dir = Path(cfg["paths"]["models"]).parent / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "labels.csv"
    if out_csv.exists() and not force:
        logger.info(f"Labels já existem: {out_csv} — pulando (use force=True para regenerar)")
        return

    label_zones = None
    if cfg["deposits"].get("use_label_sets", False):
        logger.info("Usando label_sets explícitos (nucleo / interpretado / negativo)")
        label_sets = load_label_sets(cfg)
        label_zones = build_label_zones(label_sets)
        positive_zones = {
            "porfiro": unary_union([
                info["zone"] for name, info in label_zones.items()
                if int(info["label"]) == 1
            ])
        }
        logger.info(f"Zona positiva ponderada 'porfiro': {positive_zones['porfiro'].area / 1e6:.1f} km²")
    else:
        # Carregar ground truth legado
        if cfg["deposits"].get("use_occurrences", True):
            occ = load_occurrences(cfg)
        else:
            logger.info("Ocorrências GPKG desativadas por config; usando apenas shapefiles de amostras")
            occ = gpd.GeoDataFrame(geometry=[], crs=cfg["project"]["crs"])
        amostras = load_amostras(cfg)
        positive_zones = build_positive_zone(occ, amostras)
    exclusion_zone = build_exclusion_zone(positive_zones)

    npz_files = sorted(embed_dir.glob("*.npz"))
    if not npz_files:
        logger.error(f"Nenhum NPZ de embeddings encontrado em {embed_dir}")
        return

    all_dfs = []
    for npz in npz_files:
        logger.info(f"Atribuindo labels: {npz.name}")
        df = assign_labels(
            embeddings_npz=npz,
            positive_zones=positive_zones,
            exclusion_zone=exclusion_zone,
            config=cfg,
            negative_ratio=cfg["deposits"]["negative_sampling"]["ratio"],
            patch_size_m=cfg["prithvi"]["patch_size"] * cfg["project"]["resolution"],
            priority_type=cfg["training"]["priority_model"],
            label_zones=label_zones,
        )
        all_dfs.append(df)

    labels_df = pd.concat(all_dfs, ignore_index=True)
    labels_df = spatial_train_val_test_split(
        labels_df,
        test_size=cfg["training"]["test_size"],
        val_size=cfg["training"]["val_size"],
        random_state=cfg["training"]["random_state"],
    )

    labels_df.to_csv(out_csv, index=False)
    logger.success(f"Labels salvos: {out_csv} | {len(labels_df)} patches")

    # Resumo
    summary = labels_df.groupby(["split", "deposit_type"]).size().unstack(fill_value=0)
    logger.info(f"\n{summary}")
    return labels_df


if __name__ == "__main__":
    run()
