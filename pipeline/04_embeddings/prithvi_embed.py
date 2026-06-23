"""
Fase 4 · Extração de Embeddings — Prithvi-EO-2.0 (RTX 4090).

Usa o encoder do Prithvi-EO-2.0-300M como extrator de features:
  - Patches 224x224px dos composites S2 (6 bandas HLS-compatíveis)
  - Encoder ViT frozen → embedding 768-d por patch
  - Batch de 32-64 patches na RTX 4090 (24GB VRAM)
  - Saída: arquivos NPZ por tile com embeddings + coordenadas

Formato NPZ:
  embeddings: (N, 768)  float16
  patch_ids:  (N,)      str
  centers_x:  (N,)      float32  — centróide UTM X
  centers_y:  (N,)      float32  — centróide UTM Y

Saída: D:/argentina/data/06_EMBEDDINGS/<tile_name>.npz
"""

import sys
import time
import gc
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
from loguru import logger
from tqdm import tqdm
import rasterio
from rasterio.windows import Window

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.raster_utils import load_config
from utils.tiling import TileGrid, Tile
from utils.gpu_utils import get_device, log_gpu_memory, clear_gpu_cache, optimal_batch_size

logger.remove()
logger.add(sys.stderr, level="INFO", format="<magenta>{time:HH:mm:ss}</magenta> | <level>{message}</level>")
logger.add(Path("D:/argentina/logs/prithvi_embed_{time:YYYY-MM-DD}.log"), level="DEBUG", rotation="100 MB")

# Bandas S2 compatíveis com Prithvi (ordem HLS): Blue, Green, Red, NIR_Narrow, SWIR1, SWIR2
PRITHVI_BAND_ORDER = ["B02", "B03", "B04", "B8A", "B11", "B12"]
PRITHVI_N_BANDS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Loader do modelo Prithvi
# ─────────────────────────────────────────────────────────────────────────────

class PrithviEncoder:
    """Wrapper do Prithvi-EO-2.0 para extração de embeddings."""

    def __init__(self, config: dict, device=None):
        self.cfg = config
        self.prithvi_cfg = config["prithvi"]
        self.device = device or get_device()
        self.model = None
        self.model_cfg = None
        self._loaded = False
        self._timm_mode = False

    def load(self):
        """
        Carrega Prithvi com fallback automático:
          1. Prithvi-EO-2.0-300M via timm >= 1.0  (preferido)
          2. Prithvi-EO-1.0 / Prithvi-100M via transformers ViTMAE (fallback)
        """
        if self._loaded:
            return

        import torch
        model_id = self.prithvi_cfg["model_id"]
        self._timm_mode = False

        # ── Tentativa 1: terratorch + timm (Prithvi-EO-2.0-300M) ────────────
        try:
            import terratorch  # registra prithvi_eo_v2_300 no timm
            import timm
            logger.info(f"Carregando Prithvi-EO-2.0 via terratorch+timm {timm.__version__}")
            self.model = timm.create_model(
                f"hf_hub:{model_id}",
                pretrained=True,
                num_frames=1,
                in_chans=6,
            )
            self._timm_mode = True
            logger.success("Prithvi-EO-2.0-300M carregado")
        except Exception as e:
            logger.warning(f"Prithvi-EO-2.0 falhou ({e.__class__.__name__}: {e})")

        # ── Tentativa 2: Prithvi-100M — weights .pt via hf_hub_download ──────
        if not self._timm_mode:
            try:
                from huggingface_hub import hf_hub_download
                import timm
                logger.info("Carregando Prithvi-100M (weights direto do HuggingFace)")
                pt_path = hf_hub_download("ibm-nasa-geospatial/Prithvi-100M", "Prithvi_100M.pt")
                ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                raw_sd = ckpt.get("model", ckpt)

                # Inspecionar prefixos das keys para mapear para timm
                sample_keys = list(raw_sd.keys())[:5]
                logger.debug(f"Keys no checkpoint: {sample_keys}")

                # Prithvi-100M é ViT-Base (embed_dim=768, 12 blocks)
                # com patch_embed 3D [C_out, C_in, T, H, W] e pos_embed multi-frame
                self.model = timm.create_model(
                    "vit_base_patch16_224",
                    pretrained=False,
                    in_chans=6,
                    img_size=224,
                )

                # Filtrar só encoder (remover decoder MAE) e strip do prefixo "encoder."
                encoder_sd = {}
                for k, v in raw_sd.items():
                    if k.startswith("encoder."):
                        new_k = k[len("encoder."):]
                        encoder_sd[new_k] = v

                # Adaptar pesos temporais para ViT-Base 2D (single-frame)
                adapted_sd = {}
                for k, v in encoder_sd.items():
                    if k == "patch_embed.proj.weight" and v.ndim == 5:
                        # [C_out, C_in, T, H, W] → [C_out, C_in, H, W] (média temporal)
                        adapted_sd[k] = v.mean(dim=2)
                    elif k == "pos_embed" and v.shape[1] > 197:
                        # [1, T*196+1, D] → [1, 197, D] (cls + 1ª janela temporal)
                        cls_tok = v[:, :1, :]
                        spatial = v[:, 1:197, :]
                        adapted_sd[k] = torch.cat([cls_tok, spatial], dim=1)
                    else:
                        adapted_sd[k] = v

                logger.debug(f"Keys encoder após strip: {len(adapted_sd)}")

                missing, unexpected = self.model.load_state_dict(adapted_sd, strict=False)
                n_loaded = len(adapted_sd) - len(unexpected)
                logger.success(f"Prithvi-100M ViT-Base: {n_loaded}/{len(adapted_sd)} camadas carregadas")
                if missing:
                    logger.debug(f"Keys ausentes (normal p/ decoder MAE): {len(missing)}")

                self._timm_mode = True
            except Exception as e2:
                logger.warning(f"Prithvi-100M falhou ({e2.__class__.__name__}: {e2})")

        # ── Fallback final: ViT-Large ImageNet (sem pretraining geoespacial) ─
        if not self._timm_mode:
            import timm
            logger.warning("Usando ViT-Large/16 ImageNet como fallback (sem pretraining geoespacial)")
            self.model = timm.create_model(
                "vit_large_patch16_224.augreg_in21k",
                pretrained=True,
                in_chans=6,    # adapta 1ª camada para 6 bandas
                img_size=224,
            )
            self._timm_mode = True
            logger.warning("ATENÇÃO: embeddings sem pretraining S2 — qualidade reduzida")

        self.model.eval()
        if self.prithvi_cfg["fp16"]:
            self.model = self.model.half()
        self.model = self.model.to(self.device)

        log_gpu_memory("Prithvi carregado")
        self._loaded = True
        logger.success(f"Encoder pronto | device={self.device} | timm_mode={self._timm_mode}")

    @property
    def embed_dim(self) -> int:
        return self.prithvi_cfg["embedding_dim"]

    def encode_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        patches: (B, C, H, W) float32 — reflectância normalizada
        Retorna: (B, embed_dim) float16
        """
        import torch

        if not self._loaded:
            self.load()

        dtype = torch.float16 if self.prithvi_cfg["fp16"] else torch.float32
        x = torch.from_numpy(patches).to(dtype=dtype, device=self.device)

        with torch.no_grad():
            if self._timm_mode:
                # timm: forward_features(B, C, H, W) → (B, N, D) ou (B, D)
                feats = self.model.forward_features(x)
                if feats.ndim == 3:
                    embeddings = feats[:, 1:, :].mean(dim=1)  # média spatial, sem CLS
                else:
                    embeddings = feats
            else:
                # transformers ViTMAE: espera pixel_values (B, C, H, W)
                out = self.model(pixel_values=x)
                # last_hidden_state: (B, N+1, D) — índice 0 é CLS token
                embeddings = out.last_hidden_state[:, 0, :]

        return embeddings.cpu().float().numpy().astype(np.float16)


# ─────────────────────────────────────────────────────────────────────────────
# Extração de patches de rasters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatchInfo:
    patch_id: str
    center_x: float
    center_y: float
    row_off: int
    col_off: int
    valid: bool = True


def extract_patches_from_composite(
    tif_path: Path,
    patch_size: int,
    band_names: List[str],
    nodata: float = -9999,
    min_valid_ratio: float = 0.7,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, List[PatchInfo]]:
    """
    Extrai patches regulares de um GeoTIFF multi-banda.

    Args:
        tif_path: path para o composite GeoTIFF
        patch_size: tamanho do patch em pixels (224)
        band_names: bandas a extrair na ordem correta para Prithvi
        min_valid_ratio: descarta patches com mais de (1-ratio) de nodata
        stride: passo entre patches (default = patch_size = sem overlap)

    Returns:
        patches: (N, C, H, W) float32
        infos: lista de PatchInfo com metadados espaciais
    """
    if stride is None:
        stride = patch_size

    with rasterio.open(tif_path) as src:
        all_tags = {src.tags(i).get("name", f"B{i:02d}"): i for i in range(1, src.count + 1)}
        band_indices = []
        for bn in band_names:
            if bn in all_tags:
                band_indices.append(all_tags[bn])
            else:
                logger.warning(f"Banda '{bn}' não encontrada em {tif_path.name} — usando zeros")
                band_indices.append(None)

        height, width = src.height, src.width
        transform = src.transform

        patches = []
        infos = []
        patch_idx = 0

        rows = range(0, height - patch_size + 1, stride)
        cols = range(0, width - patch_size + 1, stride)

        for row_off in rows:
            for col_off in cols:
                window = Window(col_off, row_off, patch_size, patch_size)
                patch = np.zeros((len(band_names), patch_size, patch_size), dtype=np.float32)

                for c_idx, b_idx in enumerate(band_indices):
                    if b_idx is not None:
                        data = src.read(b_idx, window=window).astype(np.float32)
                        patch[c_idx] = data
                    # else: zeros já preenchidos

                # Verificar validade
                invalid = (patch[0] == nodata) | (patch[0] == 0) | np.isnan(patch[0])
                valid_ratio = 1.0 - invalid.mean()
                if valid_ratio < min_valid_ratio:
                    continue

                # Substituir nodata por 0 (Prithvi espera dado limpo)
                patch = np.where(patch == nodata, 0.0, patch)
                patch = np.nan_to_num(patch, nan=0.0)

                # Centróide em coordenadas UTM
                cx, cy = rasterio.transform.xy(
                    transform,
                    row_off + patch_size // 2,
                    col_off + patch_size // 2,
                )

                info = PatchInfo(
                    patch_id=f"{tif_path.stem}_r{row_off}_c{col_off}",
                    center_x=cx,
                    center_y=cy,
                    row_off=row_off,
                    col_off=col_off,
                    valid=True,
                )
                patches.append(patch)
                infos.append(info)
                patch_idx += 1

    if not patches:
        return np.empty((0, len(band_names), patch_size, patch_size)), []

    return np.stack(patches, axis=0), infos


def iter_patches_from_composite(
    tif_path: Path,
    patch_size: int,
    band_names: List[str],
    nodata: float = -9999,
    min_valid_ratio: float = 0.7,
    stride: Optional[int] = None,
):
    """Versão STREAMING: gera (patch CxHxW, PatchInfo) um a um.

    Evita montar milhares de patches na RAM (necessário a 30m + stride pequeno,
    onde um único chunk pode ter >10k patches × 1.2MB).
    """
    if stride is None:
        stride = patch_size

    with rasterio.open(tif_path) as src:
        all_tags = {src.tags(i).get("name", f"B{i:02d}"): i for i in range(1, src.count + 1)}
        band_indices = [all_tags.get(bn) for bn in band_names]
        height, width = src.height, src.width
        transform = src.transform

        for row_off in range(0, height - patch_size + 1, stride):
            for col_off in range(0, width - patch_size + 1, stride):
                window = Window(col_off, row_off, patch_size, patch_size)
                patch = np.zeros((len(band_names), patch_size, patch_size), dtype=np.float32)
                for c_idx, b_idx in enumerate(band_indices):
                    if b_idx is not None:
                        patch[c_idx] = src.read(b_idx, window=window).astype(np.float32)

                invalid = (patch[0] == nodata) | (patch[0] == 0) | np.isnan(patch[0])
                if (1.0 - invalid.mean()) < min_valid_ratio:
                    continue

                patch = np.where(patch == nodata, 0.0, patch)
                patch = np.nan_to_num(patch, nan=0.0)
                cx, cy = rasterio.transform.xy(
                    transform, row_off + patch_size // 2, col_off + patch_size // 2
                )
                yield patch, PatchInfo(
                    patch_id=f"{tif_path.stem}_r{row_off}_c{col_off}",
                    center_x=cx, center_y=cy, row_off=row_off, col_off=col_off, valid=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Normalização para Prithvi
# ─────────────────────────────────────────────────────────────────────────────

# Estatísticas HLS (Harmonized Landsat Sentinel-2) do pré-treino Prithvi
# Médias e desvios para as 6 bandas: Blue, Green, Red, NIR_Narrow, SWIR1, SWIR2
PRITHVI_MEANS = np.array([494.905, 815.239, 924.517, 2968.876, 2634.022, 1739.579], dtype=np.float32)
PRITHVI_STDS  = np.array([284.925, 357.298, 373.338,  955.523, 1061.328,  820.061], dtype=np.float32)
# Esses valores são para S2 em unidade de reflectância * 10000 (inteiro)


def normalize_for_prithvi(patches: np.ndarray, reflectance_scale: float = 10000.0) -> np.ndarray:
    """
    Normaliza patches para o espaço de entrada do Prithvi.
    Input: reflectância 0-1 (já dividida por 10000)
    Output: normalizado pelas médias/desvios do pré-treino HLS
    """
    # Escalar de volta para 0-10000 (unidade do pré-treino)
    patches_scaled = patches * reflectance_scale  # (B, C, H, W)

    means = PRITHVI_MEANS.reshape(1, -1, 1, 1)
    stds = PRITHVI_STDS.reshape(1, -1, 1, 1)

    normalized = (patches_scaled - means) / (stds + 1e-8)
    return normalized.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de extração
# ─────────────────────────────────────────────────────────────────────────────

def embed_tile(
    tile_name: str,
    s2_dir: Path,
    output_npz: Path,
    encoder: PrithviEncoder,
    config: dict,
    batch_size: int = 32,
):
    """
    Processa todos os composites S2 de um tile e salva embeddings NPZ.
    """
    if output_npz.exists():
        logger.info(f"Embeddings já existem: {output_npz.name} — pulando")
        return

    s2_files = sorted(s2_dir.glob("s2_composite_*.tif"))
    if not s2_files:
        logger.warning(f"Nenhum composite S2 em {s2_dir}")
        return

    patch_size = config["prithvi"]["patch_size"]
    stride = config["prithvi"].get("stride", patch_size)
    nodata = config["output"]["nodata"]

    all_embeddings = []
    all_ids = []
    all_cx = []
    all_cy = []

    logger.info(f"Tile '{tile_name}': {len(s2_files)} composites S2 | patch={patch_size} stride={stride}")

    # Buffers de lote — RAM limitada a batch_size patches (streaming)
    buf_patches, buf_infos = [], []

    def _flush():
        if not buf_patches:
            return
        arr = np.stack(buf_patches, axis=0)          # (B,C,H,W)
        arr = normalize_for_prithvi(arr)
        embs = encoder.encode_batch(arr)
        all_embeddings.append(embs.astype(np.float16))
        for inf in buf_infos:
            all_ids.append(inf.patch_id)
            all_cx.append(inf.center_x)
            all_cy.append(inf.center_y)
        buf_patches.clear()
        buf_infos.clear()

    n_total = 0
    for tif in tqdm(s2_files, desc=f"Patches {tile_name}"):
        for patch, info in iter_patches_from_composite(
            tif_path=tif, patch_size=patch_size, band_names=PRITHVI_BAND_ORDER,
            nodata=nodata, stride=stride,
        ):
            buf_patches.append(patch)
            buf_infos.append(info)
            n_total += 1
            if len(buf_patches) >= batch_size:
                _flush()
        _flush()  # resto do composite
        clear_gpu_cache()
        gc.collect()

    if not all_embeddings:
        logger.warning(f"Nenhum embedding gerado para tile {tile_name}")
        return
    logger.info(f"Tile '{tile_name}': {n_total} patches extraídos")

    embeddings = np.vstack(all_embeddings).astype(np.float16)
    ids = np.array(all_ids)
    cx = np.array(all_cx, dtype=np.float32)
    cy = np.array(all_cy, dtype=np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        embeddings=embeddings,
        patch_ids=ids,
        centers_x=cx,
        centers_y=cy,
    )
    size_mb = output_npz.stat().st_size / 1e6
    logger.success(
        f"NPZ salvo: {output_npz.name} | "
        f"{len(embeddings)} patches | {size_mb:.0f} MB"
    )


def run(config_path: str = None, tiles: List[str] = None):
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FASE 4 · Extração de Embeddings Prithvi")
    logger.info("=" * 60)

    cfg = load_config(config_path)
    device = get_device()

    encoder = PrithviEncoder(cfg, device)
    encoder.load()

    batch_size = int(cfg["prithvi"].get("batch_size") or optimal_batch_size(model_vram_gb=4.0))
    logger.info(f"Batch size Prithvi: {batch_size}")

    downloads = Path(cfg["paths"]["downloads"])
    embed_dir = Path(cfg["paths"]["embeddings"])
    embed_dir.mkdir(parents=True, exist_ok=True)

    # Descobrir tiles a partir dos diretórios existentes em downloads/s2/
    if tiles:
        tile_names = tiles
    else:
        s2_base = downloads / "s2"
        tile_names = sorted([d.name for d in s2_base.iterdir() if d.is_dir()]) if s2_base.exists() else []
        logger.info(f"Tiles S2 descobertos: {tile_names}")

    for tile_name in tile_names:
        s2_dir = downloads / "s2" / tile_name
        if not s2_dir.exists():
            logger.warning(f"S2 não encontrado para tile '{tile_name}' — pulando")
            continue

        output_npz = embed_dir / f"{tile_name}.npz"
        embed_tile(
            tile_name=tile_name,
            s2_dir=s2_dir,
            output_npz=output_npz,
            encoder=encoder,
            config=cfg,
            batch_size=batch_size,
        )

    elapsed = time.time() - t0
    logger.success(f"Fase 4 concluída em {elapsed/60:.0f} min")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--tiles", nargs="+", default=None)
    args = parser.parse_args()
    run(config_path=args.config, tiles=args.tiles)
