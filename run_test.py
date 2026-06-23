"""
Script de teste end-to-end do pipeline Cu-pórfiro.

Usa uma área pequena (~50×50 km) dentro da cobertura S2 já existente
para validar GPU, bibliotecas e resumabilidade antes do run completo.

Área de teste: porção do centro_n com dados S2 confirmados
  UTM 19S: X 430000–480000 / Y 6185000–6235000  (~50km × 50km = 2500 km²)

Uso:
  cd D:\\argentina
  C:\\Users\\user\\anaconda3\\envs\\cu-targeting\\python.exe run_test.py
  C:\\Users\\user\\anaconda3\\envs\\cu-targeting\\python.exe run_test.py --from-phase 4
  C:\\Users\\user\\anaconda3\\envs\\cu-targeting\\python.exe run_test.py --force
"""

import sys
import shutil
import time
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuração de teste (área pequena, não sobrescreve dados reais)
# ─────────────────────────────────────────────────────────────────────────────

TEST_ROOT       = Path("D:/argentina/test_run")
TEST_DATA       = TEST_ROOT / "data"
TEST_LOGS       = TEST_ROOT / "logs"
TEST_CONFIG     = TEST_ROOT / "config_test.yaml"
CHECKPOINT_FILE = TEST_ROOT / "checkpoint.json"

# AOI de teste: área com S2 existente em centro_n
TEST_BBOX_UTM  = [430000, 6185000, 480000, 6235000]   # minx miny maxx maxy (UTM 19S)
TEST_BBOX_WGS84 = [-69.8, -34.5, -69.3, -34.05]       # WGS84 aproximado

MAIN_PIPELINE = Path("D:/argentina/pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Config YAML de teste
# ─────────────────────────────────────────────────────────────────────────────

TEST_CONFIG_YAML = f"""
project:
  name: "cu-porfiro-test"
  version: "test"
  crs: "EPSG:32719"
  resolution: 90
  log_level: "DEBUG"

paths:
  root: "{TEST_ROOT.as_posix()}"
  pipeline: "{MAIN_PIPELINE.as_posix()}"
  data: "{TEST_DATA.as_posix()}"
  aoi_shp: "{TEST_ROOT.as_posix()}/aoi_test.shp"
  aoi_tiles: "{TEST_ROOT.as_posix()}/aoi_test.shp"
  segemar_dir: "D:/argentina/02_SEGEMAR"
  amostras_gpkg: "D:/argentina/03_AMOSTRAS/Dados Argentina.gpkg"
  downloads: "{TEST_DATA.as_posix()}/04_MPC_DOWNLOADS"
  rasters: "{TEST_DATA.as_posix()}/05_RASTERS"
  embeddings: "{TEST_DATA.as_posix()}/06_EMBEDDINGS"
  models: "{TEST_DATA.as_posix()}/07_MODELS"
  output: "{TEST_DATA.as_posix()}/08_OUTPUT"
  logs: "{TEST_LOGS.as_posix()}"

aoi:
  bbox_utm19s: {TEST_BBOX_UTM}
  bbox_wgs84: {list(TEST_BBOX_WGS84)}
  tiles:
    - id: "test_tile"
      priority: 1

satellite:
  s2:
    collection: "sentinel-2-l2a"
    bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    date_start: "2022-03-01"
    date_end: "2024-10-31"
    cloud_cover_max: 20
    composite_method: "median"
    months_dry: [3, 4, 5, 6, 7, 8, 9, 10]
    max_scenes: 10
  dem:
    collection: "cop-dem-glo-30"
    resolution: 30

spectral_indices:
  clay_index:   {{b1: "B11", b2: "B12", formula: "B11/B12"}}
  ferric_iron:  {{b1: "B04", b2: "B03", formula: "B04/B03"}}
  iron_oxide:   {{b1: "B04", b2: "B02", formula: "B04/B02"}}
  ndvi:         {{b1: "B08", b2: "B04", formula: "(B08-B04)/(B08+B04)"}}
  ndwi:         {{b1: "B03", b2: "B08", formula: "(B03-B08)/(B03+B08)"}}
  swir_ratio:   {{b1: "B11", b2: "B08", formula: "B11/B08"}}

segemar_layers:
  - name: "geotectonico_clasif"
    file: "e2_5M_Geotectonico.shp"
    field: "clasif_tec"
    type: "categorical"
  - name: "falhas_densidade"
    file: "e5M_AMS_Fallas.shp"
    type: "density"
    density_radius_m: 15000
  - name: "geologia_ambiente"
    file: "e2_5M_UnidadesGeologicas.shp"
    field: "ambiente"
    type: "categorical"

deposits:
  ocorrencias_layer: "Ocorrencias_de_Cobre_Aoi_arg_if_v2"
  tipo_field: "cod_modelo"
  tipos:
    porfiro: ["4b", "4c"]
  buffer_m:
    porfiro: 4000
  amostras_dir: "D:/argentina/03_AMOSTRAS"
  amostras_shapefiles:
    - "amo01.shp"
    - "amo02.shp"
    - "amo03.shp"
    - "amo04.shp"
    - "amo05.shp"
    - "amo06.shp"
    - "amo07.shp"
    - "amo08.shp"
    - "amo09.shp"
    - "amo010.shp"
    - "amo011.shp"
  negative_sampling:
    ratio: 5
    min_dist_from_deposit_m: 10000

prithvi:
  model_id: "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
  patch_size: 224
  patch_size_m: 6720
  embedding_dim: 768
  batch_size: 128
  num_frames: 1
  bands_input: 6
  fp16: true
  device: "cuda"

training:
  priority_model: "porfiro"
  test_size: 0.20
  val_size: 0.20
  random_state: 42
  use_smote: true
  smote_k_neighbors: 3
  n_trials_optuna: 20
  lgbm_params_base:
    objective: "binary"
    metric: "auc"
    n_estimators: 300
    learning_rate: 0.05
    num_leaves: 31
    min_child_samples: 5
    subsample: 0.8
    colsample_bytree: 0.8
    class_weight: "balanced"
    device: "cpu"
    verbose: -1

mpc:
  max_workers: 4
  chunk_size_deg: 0.5
  retry_attempts: 3
  timeout_s: 180
  token: ""

output:
  formats: ["geotiff", "shapefile"]
  top_targets_n: 10
  colormap: "RdYlGn"
  nodata: -9999
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "  ", "OK": "✓ ", "FAIL": "✗ ", "SKIP": "→ ", "HEAD": ""}
    print(f"{ts} | {prefix.get(level,'  ')}{msg}", flush=True)


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(phase: int, status: str, elapsed: float):
    ck = load_checkpoint()
    ck[str(phase)] = {"status": status, "elapsed_min": round(elapsed, 2),
                      "ts": datetime.now().isoformat()}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ck, f, indent=2)


def _reuse_existing_data():
    """
    Copia S2/DEM já baixados em D:/argentina/data que sobrepõem o bbox de teste.
    Rename para 'test_tile' para que as fases seguintes os encontrem.
    """
    import shutil
    import rasterio
    from shapely.geometry import box as shp_box

    src_base = Path("D:/argentina/data/04_MPC_DOWNLOADS")
    dst_base = TEST_DATA / "04_MPC_DOWNLOADS"
    test_geom = shp_box(*TEST_BBOX_UTM)

    copied = 0
    for product in ("s2", "dem"):
        src_product = src_base / product
        if not src_product.exists():
            continue
        for tif in src_product.rglob("*.tif"):
            try:
                with rasterio.open(tif) as src:
                    b = src.bounds
                    tif_geom = shp_box(b.left, b.bottom, b.right, b.top)
                if not tif_geom.intersects(test_geom):
                    continue
                dst_dir = dst_base / product / "test_tile"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_dir / tif.name
                if not dst_file.exists():
                    shutil.copy2(tif, dst_file)
                    log(f"Copiado: {product}/{tif.name} ({tif.stat().st_size//1e6:.0f}MB)", "INFO")
                    copied += 1
            except Exception as e:
                log(f"Não foi possível copiar {tif.name}: {e}", "INFO")

    if copied == 0:
        log("Nenhum arquivo S2/DEM encontrado para reutilizar — Fase 2 precisará baixar.", "INFO")
    else:
        log(f"Reutilizados {copied} arquivo(s) de dados existentes.", "OK")

        # Replicar derivadas de terreno também
        src_terrain = Path("D:/argentina/data/05_RASTERS/terrain")
        dst_terrain = TEST_DATA / "05_RASTERS" / "terrain" / "test_tile"
        dst_terrain.mkdir(parents=True, exist_ok=True)
        for tif in src_terrain.rglob("*.tif"):
            try:
                with rasterio.open(tif) as src:
                    b = src.bounds
                    if shp_box(b.left, b.bottom, b.right, b.top).intersects(test_geom):
                        dst = dst_terrain / tif.name
                        if not dst.exists():
                            shutil.copy2(tif, dst)
            except Exception:
                pass


def create_test_aoi():
    """Cria AOI shapefile de teste cobrindo a área definida."""
    import fiona
    from fiona.crs import CRS
    from shapely.geometry import box, mapping

    aoi_shp = TEST_ROOT / "aoi_test.shp"
    if aoi_shp.exists():
        return aoi_shp

    schema = {"geometry": "Polygon", "properties": {"Id": "int"}}
    geom = box(*TEST_BBOX_UTM)

    with fiona.open(str(aoi_shp), "w", driver="ESRI Shapefile",
                    schema=schema, crs=CRS.from_epsg(32719)) as dst:
        dst.write({"geometry": mapping(geom), "properties": {"Id": 1}})

    log(f"AOI teste criada: {aoi_shp} | bbox={TEST_BBOX_UTM}")
    return aoi_shp


def setup_test_environment(force: bool):
    """Cria diretórios e config de teste."""
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    TEST_LOGS.mkdir(parents=True, exist_ok=True)

    if not TEST_CONFIG.exists() or force:
        TEST_CONFIG.write_text(TEST_CONFIG_YAML, encoding="utf-8")
        log(f"Config de teste: {TEST_CONFIG}")

    create_test_aoi()

    # Adicionar pipeline ao path
    sys.path.insert(0, str(MAIN_PIPELINE))
    sys.path.insert(0, str(MAIN_PIPELINE / "utils"))


# ─────────────────────────────────────────────────────────────────────────────
# Fases do teste
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(phase: int, force: bool = False) -> bool:
    ck = load_checkpoint()
    if ck.get(str(phase), {}).get("status") == "done" and not force:
        prev = ck[str(phase)]
        log(f"Fase {phase} já concluída ({prev['elapsed_min']:.1f} min) — pulando", "SKIP")
        return True

    log(f"{'='*55}", "HEAD")
    log(f"FASE {phase}", "HEAD")
    log(f"{'='*55}", "HEAD")

    t0 = time.time()
    ok = True
    cfg_path = str(TEST_CONFIG)

    try:
        if phase == 1:
            # SEGEMAR — rasterizar 3 camadas para a AOI de teste
            sys.path.insert(0, str(MAIN_PIPELINE / "01_segemar"))
            from importlib import import_module, reload
            import segemar_prep
            reload(segemar_prep)
            segemar_prep.run(cfg_path)

        elif phase == 2:
            # Download S2 + DEM para a área de teste (pequena → rápido)
            sys.path.insert(0, str(MAIN_PIPELINE / "02_mpc"))
            from importlib import import_module, reload
            import mpc_download_agent
            reload(mpc_download_agent)
            mpc_download_agent.run(config_path=cfg_path)

        elif phase == 3:
            # Índices espectrais
            sys.path.insert(0, str(MAIN_PIPELINE / "03_features"))
            from importlib import import_module, reload
            import spectral_indices
            reload(spectral_indices)
            spectral_indices.run(cfg_path)

        elif phase == 4:
            # Prithvi embeddings GPU
            import torch
            log(f"CUDA disponível: {torch.cuda.is_available()}", "INFO")
            if torch.cuda.is_available():
                log(f"GPU: {torch.cuda.get_device_name(0)}", "INFO")
                log(f"VRAM livre: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", "INFO")
            sys.path.insert(0, str(MAIN_PIPELINE / "04_embeddings"))
            from importlib import import_module, reload
            import prithvi_embed
            reload(prithvi_embed)
            prithvi_embed.run(config_path=cfg_path)

        elif phase == 5:
            # Labels + Treino
            sys.path.insert(0, str(MAIN_PIPELINE / "05_labels"))
            sys.path.insert(0, str(MAIN_PIPELINE / "06_train"))
            from importlib import import_module, reload
            import labels, train
            reload(labels); reload(train)
            labels.run(cfg_path, force=force)
            train.run(config_path=cfg_path, force=force)

        elif phase == 6:
            # Inferência + mosaico
            sys.path.insert(0, str(MAIN_PIPELINE / "07_inference"))
            sys.path.insert(0, str(MAIN_PIPELINE / "08_output"))
            from importlib import import_module, reload
            import predict, mosaic_export
            reload(predict); reload(mosaic_export)
            predict.run(config_path=cfg_path)
            mosaic_export.run(cfg_path)

    except Exception as e:
        log(f"FASE {phase} FALHOU: {e}", "FAIL")
        traceback.print_exc()
        ok = False

    elapsed = (time.time() - t0) / 60
    save_checkpoint(phase, "done" if ok else "failed", elapsed)
    status = "OK" if ok else "FAIL"
    log(f"Fase {phase} {status} em {elapsed:.1f} min", status)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Teste end-to-end do pipeline Cu-pórfiro")
    parser.add_argument("--from-phase", type=int, default=1,
                        help="Começar da fase N (1-6). Fases anteriores puladas.")
    parser.add_argument("--phase", type=int, default=None,
                        help="Executar só esta fase.")
    parser.add_argument("--force", action="store_true",
                        help="Ignorar checkpoint e re-executar todas as fases.")
    parser.add_argument("--clean", action="store_true",
                        help="Limpar pasta de teste antes de rodar.")
    parser.add_argument("--reuse-data", action="store_true",
                        help="Reutilizar S2/DEM já baixados em D:/argentina/data (pula download).")
    args = parser.parse_args()

    if args.clean and TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
        log("Pasta de teste removida.", "INFO")

    log("=" * 55, "HEAD")
    log("  TESTE END-TO-END · Cu Pórfiro Pipeline", "HEAD")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}", "HEAD")
    log(f"  AOI: {TEST_BBOX_UTM} (UTM 19S)", "HEAD")
    log(f"  Saída: {TEST_ROOT}", "HEAD")
    log("=" * 55, "HEAD")

    setup_test_environment(force=args.force)

    if args.reuse_data:
        _reuse_existing_data()

    phases = [args.phase] if args.phase else list(range(1, 7))
    phases = [p for p in phases if p >= args.from_phase]
    if args.reuse_data and 2 in phases:
        phases = [p for p in phases if p != 2]
        log("--reuse-data: Fase 2 pulada (dados copiados de D:/argentina/data)", "SKIP")

    t_total = time.time()
    for phase in phases:
        ok = run_phase(phase, force=args.force)
        if not ok:
            log(f"Pipeline de teste interrompido na fase {phase}. "
                f"Corrija o erro e rode novamente — fases anteriores serão puladas.", "FAIL")
            sys.exit(1)

    total = (time.time() - t_total) / 60
    log("=" * 55, "HEAD")
    log(f"  TESTE COMPLETO em {total:.0f} min", "OK")
    log(f"  Mapas: {TEST_ROOT / 'data' / '08_OUTPUT'}", "OK")
    log("=" * 55, "HEAD")

    # Mostrar resumo dos outputs gerados
    output_dir = TEST_DATA / "08_OUTPUT"
    if output_dir.exists():
        tifs = list(output_dir.rglob("*.tif"))
        shps = list(output_dir.rglob("*.shp"))
        log(f"  GeoTIFFs gerados: {len(tifs)}", "INFO")
        log(f"  Shapefiles gerados: {len(shps)}", "INFO")
        for f in sorted(tifs + shps):
            log(f"  → {f.relative_to(TEST_ROOT)}", "INFO")


if __name__ == "__main__":
    main()
