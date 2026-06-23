"""
Orquestrador Principal do Pipeline — Cu Targeting Argentina.

Executa as 6 fases em sequência, com suporte a:
  - Retomada de ponto de parada (--resume)
  - Execução de fase específica (--phase)
  - Tiles específicos (--tiles)

Uso:
  conda activate cu-targeting
  python run_pipeline.py                          # pipeline completo
  python run_pipeline.py --phase 1                # só SEGEMAR
  python run_pipeline.py --phase 2 --tiles norte  # download só tile norte
  python run_pipeline.py --resume                 # pula fases já concluídas
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

from loguru import logger

# Setup logging global
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<bold><green>{time:HH:mm:ss}</green></bold> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("D:/argentina/logs/pipeline_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="200 MB")

CONFIG_PATH = Path("D:/argentina/pipeline/00_config/config.yaml")
CHECKPOINT_FILE = Path("D:/argentina/logs/.pipeline_checkpoint.json")


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(phase: int, status: str, elapsed_min: float):
    ck = load_checkpoint()
    ck[str(phase)] = {"status": status, "elapsed_min": round(elapsed_min, 1),
                      "timestamp": datetime.now().isoformat()}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ck, f, indent=2)


def run_phase(phase: int, tiles: list = None, resume: bool = False, force: bool = False):
    ck = load_checkpoint()
    if resume and ck.get(str(phase), {}).get("status") == "done":
        logger.info(f"Fase {phase} já concluída (--resume) — pulando")
        return True

    t0 = time.time()
    ok = True

    try:
        if phase == 1:
            logger.info("▶ FASE 1 · Preparação SEGEMAR")
            sys.path.insert(0, str(Path(__file__).parent / "01_segemar"))
            from segemar_prep import run
            run(str(CONFIG_PATH))

        elif phase == 2:
            logger.info("▶ FASE 2 · Download Planetary Computer")
            sys.path.insert(0, str(Path(__file__).parent / "02_mpc"))
            from mpc_download_agent import run
            run(config_path=str(CONFIG_PATH), tiles=tiles)

        elif phase == 3:
            logger.info("▶ FASE 3 · Índices Espectrais")
            sys.path.insert(0, str(Path(__file__).parent / "03_features"))
            from spectral_indices import run
            run(str(CONFIG_PATH))

        elif phase == 4:
            logger.info("▶ FASE 4 · Embeddings Prithvi (GPU)")
            sys.path.insert(0, str(Path(__file__).parent / "04_embeddings"))
            from prithvi_embed import run
            run(config_path=str(CONFIG_PATH), tiles=tiles)

        elif phase == 5:
            logger.info("▶ FASE 5a · Labels")
            sys.path.insert(0, str(Path(__file__).parent / "05_labels"))
            from labels import run as run_labels
            run_labels(str(CONFIG_PATH), force=force)

            logger.info("▶ FASE 5b · Treino LightGBM")
            sys.path.insert(0, str(Path(__file__).parent / "06_train"))
            from train import run as run_train
            run_train(config_path=str(CONFIG_PATH), force=force)

        elif phase == 6:
            logger.info("▶ FASE 6a · Inferência")
            sys.path.insert(0, str(Path(__file__).parent / "07_inference"))
            from predict import run as run_predict
            run_predict(config_path=str(CONFIG_PATH), tiles=tiles)

            logger.info("▶ FASE 6b · Mosaico e Exportação")
            sys.path.insert(0, str(Path(__file__).parent / "08_output"))
            from mosaic_export import run as run_mosaic
            run_mosaic(str(CONFIG_PATH))

    except Exception as e:
        logger.error(f"FASE {phase} FALHOU: {e}")
        import traceback
        logger.error(traceback.format_exc())
        ok = False

    elapsed = (time.time() - t0) / 60
    save_checkpoint(phase, "done" if ok else "failed", elapsed)
    logger.info(f"Fase {phase} {'✓' if ok else '✗'} em {elapsed:.1f} min")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Pipeline Cu Targeting Argentina")
    parser.add_argument("--phase", type=int, default=None,
                        help="Executar só uma fase (1-6)")
    parser.add_argument("--tiles", nargs="+", default=None,
                        help="Segmentos específicos: seg_norte seg_centro seg_sul")
    parser.add_argument("--force", action="store_true",
                        help="Forçar re-execução mesmo se outputs existirem")
    parser.add_argument("--resume", action="store_true",
                        help="Pular fases já concluídas")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Começar da fase N (ignora --resume para fases anteriores)")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("  PIPELINE · TARGETING Cu · ARGENTINA")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 65)

    phases = [args.phase] if args.phase else list(range(1, 7))
    phases = [p for p in phases if p >= args.start_from]

    t_total = time.time()
    for phase in phases:
        ok = run_phase(phase, tiles=args.tiles, resume=args.resume, force=args.force)
        if not ok:
            logger.error(f"Pipeline interrompido na fase {phase}")
            sys.exit(1)

    total_min = (time.time() - t_total) / 60
    logger.success(f"\n{'='*65}")
    logger.success(f"  PIPELINE COMPLETO em {total_min:.0f} min ({total_min/60:.1f}h)")
    logger.success(f"  Mapas: D:/argentina/data/08_OUTPUT/")
    logger.success(f"{'='*65}")


if __name__ == "__main__":
    main()
