"""
Fase 2 · Agente de Download — Microsoft Planetary Computer.

Baixa de forma paralela e automatizada:
  - Sentinel-2 L2A  (composite mediana estação seca 2021-2024)
  - Copernicus DEM GLO-30
  - Sentinel-1 SAR RTC (VV, VH)
  - ASTER L1T (quando disponível via MPC)

Estratégia:
  1. Para cada tile primário (N/C/S), gera sub-bbox de ~1°x1°
  2. Busca cenas via STAC API do Planetary Computer
  3. Baixa bandas necessárias em paralelo (aiohttp)
  4. Gera composite mediana local por bbox
  5. Salva GeoTIFF por bbox em data/04_MPC_DOWNLOADS/

O agente retoma de onde parou (verifica arquivos existentes).
"""

import sys
import asyncio
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import requests
from loguru import logger
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import box
import planetary_computer
import pystac_client

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.raster_utils import load_config, save_raster, get_profile
from utils.tiling import TileGrid, Tile

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="<cyan>{time:HH:mm:ss}</cyan> | <level>{message}</level>")
logger.add(
    Path("D:/argentina/logs/mpc_download_{time:YYYY-MM-DD}.log"),
    level="DEBUG", rotation="100 MB",
)

# Earth Search (Element84) — Sentinel-2 + Copernicus DEM na AWS, sem autenticação
EARTH_SEARCH_CATALOG = "https://earth-search.aws.element84.com/v1"
# Manter MPC como fallback (quando disponível)
MPC_CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"
ACTIVE_CATALOG = EARTH_SEARCH_CATALOG  # trocar para MPC_CATALOG se preferir

# Bandas S2 necessárias para Prithvi (HLS-compatível) + índices extras
S2_BANDS_PRITHVI = ["B02", "B03", "B04", "B8A", "B11", "B12"]   # para Prithvi
S2_BANDS_EXTRA   = ["B05", "B06", "B07", "B08"]                   # índices adicionais
S2_ALL_BANDS     = S2_BANDS_PRITHVI + S2_BANDS_EXTRA

# Earth Search usa nomes de assets diferentes do MPC — mapear de/para
EARTH_SEARCH_BAND_MAP = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22",
}
# Mapa inverso para renomear de volta após download
EARTH_SEARCH_BAND_MAP_INV = {v: k for k, v in EARTH_SEARCH_BAND_MAP.items()}

# Meses estação seca Andes argentinos (menos nuvem)
DRY_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10]


# ─────────────────────────────────────────────────────────────────────────────
# Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DownloadJob:
    """Representa um job de download para um bbox."""
    job_id: str
    collection: str
    bbox_wgs84: Tuple[float, float, float, float]
    bbox_utm: Tuple[float, float, float, float]
    output_dir: Path
    date_start: str
    date_end: str
    bands: List[str] = field(default_factory=list)
    status: str = "pending"      # pending | running | done | failed
    n_scenes: int = 0
    output_path: Optional[Path] = None


class MPCDownloadAgent:
    """
    Agente autônomo de download via Microsoft Planetary Computer.
    Gerencia fila de jobs, retry automático e checkpoint de progresso.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.mpc_cfg = config["mpc"]
        self.out_base = Path(config["paths"]["downloads"])
        self.crs = config["project"]["crs"]
        self.resolution = config["project"]["resolution"]
        self._catalog = None
        self._jobs: List[DownloadJob] = []
        self._checkpoint_file = self.out_base / ".download_checkpoint.json"

    def _get_catalog(self) -> pystac_client.Client:
        if self._catalog is None:
            if ACTIVE_CATALOG == EARTH_SEARCH_CATALOG:
                # Earth Search (AWS) — sem autenticação
                self._catalog = pystac_client.Client.open(EARTH_SEARCH_CATALOG)
                logger.info(f"Conectado ao Earth Search (AWS): {EARTH_SEARCH_CATALOG}")
            else:
                # Microsoft Planetary Computer — requer assinatura de URLs
                self._catalog = pystac_client.Client.open(
                    MPC_CATALOG,
                    modifier=planetary_computer.sign_inplace,
                )
                logger.info(f"Conectado ao Planetary Computer: {MPC_CATALOG}")
        return self._catalog

    # ── Geração de bounding boxes ─────────────────────────────────────────────

    def generate_download_bboxes(self, tile: Tile, chunk_deg: float = 1.0) -> List[Tuple]:
        """
        Divide um tile em bboxes WGS84 de ~chunk_deg graus para download.
        Retorna lista de (minx, miny, maxx, maxy) em WGS84.
        """
        b = tile.bounds_wgs84
        xs = np.arange(b[0], b[2], chunk_deg)
        ys = np.arange(b[1], b[3], chunk_deg)
        bboxes = []
        for x in xs:
            for y in ys:
                bboxes.append((
                    round(x, 4),
                    round(y, 4),
                    round(min(x + chunk_deg, b[2]), 4),
                    round(min(y + chunk_deg, b[3]), 4),
                ))
        return bboxes

    # ── Sentinel-2 ────────────────────────────────────────────────────────────

    def _stac_search_with_retry(self, max_retries: int = 5, wait_s: float = 10.0, **kwargs) -> List:
        """Busca STAC com retry exponencial para timeouts do MPC.
        Usa paginação manual (pages) para evitar timeout em item_collection() grande.
        """
        import time as _time
        from pystac_client.exceptions import APIError

        # Limitar no servidor: evita o MPC varrer milhares de itens antes de responder
        kwargs.setdefault("max_items", kwargs.pop("max_items", 100))

        catalog = self._get_catalog()
        for attempt in range(1, max_retries + 1):
            try:
                results = []
                search = catalog.search(**kwargs)
                # Iterar página por página em vez de item_collection() tudo de uma vez
                for page in search.pages():
                    results.extend(page.items)
                    if len(results) >= kwargs.get("max_items", 100):
                        break
                return results
            except (APIError, Exception) as e:
                if attempt == max_retries:
                    raise
                wait = wait_s * (2 ** (attempt - 1))
                logger.warning(f"STAC search falhou (tentativa {attempt}/{max_retries}): {e} — aguardando {wait:.0f}s")
                _time.sleep(wait)
        return []

    def search_s2_scenes(
        self,
        bbox: Tuple,
        date_start: str,
        date_end: str,
        cloud_max: int = 20,
        dry_months_only: bool = True,
        max_scenes: int = 40,
    ) -> List:
        """
        Busca cenas S2 L2A — limita a max_scenes para controlar memória.
        Seleciona cenas distribuídas por ano/mês para melhor composite.
        """
        # Servidor: buscar cenas suficientes de TODOS os MGRS tiles que cobrem o bbox.
        # Limite alto pois precisamos ver todos os tiles, não só os mais recentes.
        server_limit = max(max_scenes * 8, 800)
        items = self._stac_search_with_retry(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lt": cloud_max}},
            max_items=server_limit,
        )

        items = list(items)
        if dry_months_only:
            items = [
                it for it in items
                if datetime.fromisoformat(it.datetime.isoformat()).month in DRY_MONTHS
            ]

        if not items:
            logger.warning(f"Sem cenas S2 para bbox {bbox}")
            return []

        # ── Seleção GULOSA por cobertura de footprint ──────────────────────────
        # A footprint REAL de cada cena S2 é diagonal (borda do swath de órbita),
        # não o quadrado do tile MGRS. Selecionar por "menor nuvem" enviesa para
        # cenas da mesma órbita cujas faixas diagonais se sobrepõem → metade do
        # bbox fica sem cena. Aqui escolhemos cenas que MAXIMIZAM a área coberta
        # do bbox e só então completamos o orçamento com as mais limpas (mediana).
        from shapely.geometry import box as _box, shape as _shape
        from shapely.ops import unary_union

        target = _box(*bbox)
        tgt_area = target.area
        cand = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))

        # footprint de cada candidato (interseção com o bbox)
        foots = []
        for it in cand:
            try:
                g = _shape(it.geometry).intersection(target)
                foots.append(g if not g.is_empty else None)
            except Exception:
                foots.append(None)

        chosen, chosen_idx = [], set()
        covered = None
        cov_area = 0.0
        # 1) Guloso: adiciona a cena que mais aumenta a cobertura, até ~99.5%
        while cov_area < 0.995 * tgt_area:
            best_i, best_gain, best_union = -1, 0.0, None
            for i, g in enumerate(foots):
                if i in chosen_idx or g is None:
                    continue
                u = g if covered is None else unary_union([covered, g])
                gain = u.area - cov_area
                if gain > best_gain:
                    best_i, best_gain, best_union = i, gain, u
            if best_i < 0 or best_gain < tgt_area * 0.001:
                break  # nada relevante a adicionar
            chosen_idx.add(best_i)
            chosen.append(cand[best_i])
            covered, cov_area = best_union, best_union.area

        cov_pct = 100.0 * cov_area / tgt_area

        # 2) Completar orçamento com as cenas mais limpas (melhora a mediana)
        for i, it in enumerate(cand):
            if len(chosen) >= max_scenes:
                break
            if i not in chosen_idx:
                chosen_idx.add(i)
                chosen.append(it)

        logger.info(
            f"S2: {len(chosen)} cenas | cobertura do bbox por footprints: {cov_pct:.1f}% "
            f"(guloso: {len(chosen_idx)} avaliadas)"
        )
        return chosen

    def download_s2_composite(
        self,
        bbox_wgs84: Tuple,
        items: List,
        output_path: Path,
        bands: List[str] = None,
        resampling: Resampling = Resampling.bilinear,
    ) -> Optional[Path]:
        """
        Cria composite mediana S2 via stackstac (substitui odc-stac).
        stackstac é mais estável com versões recentes do dask.
        """
        if not items:
            logger.warning(f"Sem cenas S2 para bbox {bbox_wgs84}")
            return None

        if output_path.exists():
            logger.debug(f"Já existe: {output_path.name} — pulando")
            return output_path

        if bands is None:
            bands = S2_ALL_BANDS

        # Earth Search usa nomes de asset diferentes (blue/green/... em vez de B02/B03/...)
        if ACTIVE_CATALOG == EARTH_SEARCH_CATALOG:
            asset_names = [EARTH_SEARCH_BAND_MAP.get(b, b) for b in bands]
        else:
            asset_names = bands

        try:
            import stackstac
            import dask
            import os
            from rasterio.transform import from_origin

            # GDAL/S3 sem autenticação (dados públicos Earth Search / AWS)
            os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
            os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
            os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff"
            os.environ["GDAL_HTTP_TIMEOUT"] = "120"

            # Limitar uso de memória dask
            dask.config.set({"distributed.worker.memory.target": 0.7,
                             "array.chunk-size": "256MiB"})

            # Filtrar itens sem geometria/bbox válida antes de passar ao stackstac
            valid_items = [it for it in items if it.bbox is not None]
            if not valid_items:
                logger.warning(f"Nenhuma cena S2 com bbox válido para {bbox_wgs84}")
                return None
            if len(valid_items) < len(items):
                logger.warning(f"Removidas {len(items)-len(valid_items)} cenas sem bbox")

            target_epsg = int(self.crs.split(":")[1])

            def _build_stack(use_epsg: bool):
                kwargs = dict(
                    assets=asset_names,
                    bounds_latlon=bbox_wgs84,
                    resolution=self.resolution,
                    resampling=resampling,
                    rescale=False,
                    chunksize=1024,
                )
                if use_epsg:
                    kwargs["epsg"] = target_epsg
                return stackstac.stack(valid_items, **kwargs)

            logger.info(f"Carregando {len(valid_items)} cenas S2 via stackstac...")
            try:
                stack = _build_stack(use_epsg=True)
            except Exception as e_inner:
                if "out_bounds" in str(e_inner):
                    logger.warning(f"out_bounds=None com epsg={target_epsg} — retentando sem epsg fixo")
                    stack = _build_stack(use_epsg=False)
                else:
                    raise

            logger.info(f"Calculando mediana de {len(valid_items)} cenas...")
            # Retry com backoff para erros S3 transientes (RasterioIOError)
            import time as _time
            for _attempt in range(3):
                try:
                    # 0 = nodata do S2 (fora do swath diagonal de cada granule).
                    # Mascarar ANTES da mediana para o skipna ignorar por cena —
                    # senão a borda diagonal vira mediana≈0 e abre buraco.
                    composite = stack.where(stack > 0).median(dim="time", skipna=True).compute()
                    break
                except Exception as _e:
                    if _attempt == 2:
                        raise
                    wait = 15 * (2 ** _attempt)
                    logger.warning(f"Erro no compute() tentativa {_attempt+1}/3 — aguardando {wait}s: {_e}")
                    _time.sleep(wait)
                    # Rebuild stack com menos cenas (excluir cenas mais nubladas)
                    valid_items = valid_items[:-max(1, len(valid_items)//4)]
                    if not valid_items:
                        raise RuntimeError("Sem cenas válidas após retry")
                    stack = _build_stack(use_epsg=True)

            arr = composite.values.astype(np.float32)  # (C, H, W)
            # nodata: reflectância 0 (S2 nodata) OU NaN (fora do footprint do tile)
            nodata_mask = (arr == 0) | ~np.isfinite(arr)
            arr = arr / 10000.0
            arr = np.clip(arr, 0, 1)
            arr = np.where(nodata_mask, -9999, arr)

            x = composite.x.values
            y = composite.y.values
            res_x = abs(float(x[1] - x[0])) if len(x) > 1 else self.resolution
            res_y = abs(float(y[1] - y[0])) if len(y) > 1 else self.resolution
            out_crs = str(composite.rio.crs) if hasattr(composite, "rio") else self.crs

            profile = {
                "driver": "GTiff", "dtype": "float32", "nodata": -9999,
                "width": len(x), "height": len(y), "count": arr.shape[0],
                "crs": out_crs,
                "transform": from_origin(float(x[0]), float(y[0]), res_x, res_y),
                "compress": "lzw", "tiled": True, "blockxsize": 512, "blockysize": 512,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_raster(arr, output_path, profile, descriptions=bands)  # sempre nomes B02/B03/...
            logger.success(f"S2 composite: {output_path.name} | shape={arr.shape}")
            return output_path

        except Exception as e:
            logger.error(f"Erro ao processar S2 para {output_path.name}: {e}")
            import traceback; logger.debug(traceback.format_exc())
            return None

    # ── Copernicus DEM ─────────────────────────────────────────────────────────

    def download_dem(
        self,
        bbox_wgs84: Tuple,
        output_path: Path,
    ) -> Optional[Path]:
        """Baixa Copernicus DEM 30m via stackstac."""
        if output_path.exists():
            logger.debug(f"DEM já existe: {output_path.name}")
            return output_path

        # Earth Search: "cop-dem-glo-30" | MPC: "cop-dem-glo-30" — mesmo nome
        items = self._stac_search_with_retry(
            collections=["cop-dem-glo-30"],
            bbox=bbox_wgs84,
            max_items=50,
        )

        if not items:
            logger.warning(f"DEM não encontrado para bbox {bbox_wgs84}")
            return None

        try:
            import stackstac
            import os
            from rasterio.transform import from_origin

            os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
            os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
            os.environ["GDAL_HTTP_TIMEOUT"] = "120"

            stack = stackstac.stack(
                items,
                assets=["data"],
                bounds_latlon=bbox_wgs84,
                resolution=self.resolution,
                epsg=int(self.crs.split(":")[1]),
                rescale=False,
            )
            arr = stack.median(dim="time", skipna=True).compute().values.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            arr = np.where(arr == 0, -9999, arr)

            x = stack.x.values
            y = stack.y.values
            res_x = abs(float(x[1] - x[0])) if len(x) > 1 else self.resolution
            res_y = abs(float(y[1] - y[0])) if len(y) > 1 else self.resolution
            profile = {
                "driver": "GTiff", "dtype": "float32", "nodata": -9999,
                "width": len(x), "height": len(y), "count": 1,
                "crs": self.crs,
                "transform": from_origin(float(x[0]), float(y[0]), res_x, res_y),
                "compress": "lzw", "tiled": True, "blockxsize": 512, "blockysize": 512,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_raster(arr, output_path, profile, descriptions=["elevation_m"])
            logger.success(f"DEM: {output_path.name}")
            return output_path

        except Exception as e:
            logger.error(f"Erro DEM para {output_path.name}: {e}")
            import traceback; logger.debug(traceback.format_exc())
            return None

    # ── Sentinel-1 SAR ────────────────────────────────────────────────────────

    def download_sar(
        self,
        bbox_wgs84: Tuple,
        output_path: Path,
        date_start: str = "2021-01-01",
        date_end: str = "2024-12-31",
    ) -> Optional[Path]:
        """Baixa composite mediana SAR (VV, VH) para o bbox."""
        if output_path.exists():
            logger.debug(f"SAR já existe: {output_path.name}")
            return output_path

        # sentinel-1-rtc só está no MPC, não no Earth Search
        if ACTIVE_CATALOG != MPC_CATALOG:
            logger.warning("SAR (Sentinel-1) não disponível no Earth Search — pulando")
            return None

        items = self._stac_search_with_retry(
            collections=["sentinel-1-rtc"],
            bbox=bbox_wgs84,
            datetime=f"{date_start}/{date_end}",
            max_items=60,
        )

        if not items:
            logger.warning(f"SAR não encontrado para bbox {bbox_wgs84}")
            return None

        try:
            import stackstac
            import os
            from rasterio.transform import from_origin

            os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
            os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
            os.environ["GDAL_HTTP_TIMEOUT"] = "120"

            stack = stackstac.stack(
                items,
                assets=["vv", "vh"],
                bounds_latlon=bbox_wgs84,
                resolution=self.resolution,
                epsg=int(self.crs.split(":")[1]),
                rescale=False,
            )
            composite = stack.median(dim="time", skipna=True).compute()
            vv = composite.sel(band="vv").values.astype(np.float32)
            vh = composite.sel(band="vh").values.astype(np.float32)
            ratio = np.where(vh != 0, vv / (vh + 1e-10), 0.0)

            arr = np.stack([
                np.where(vv == 0, -9999, vv),
                np.where(vh == 0, -9999, vh),
                np.where(ratio == 0, -9999, ratio),
            ], axis=0)

            x = composite.x.values
            y = composite.y.values
            res_x = abs(float(x[1] - x[0])) if len(x) > 1 else self.resolution
            res_y = abs(float(y[1] - y[0])) if len(y) > 1 else self.resolution
            profile = {
                "driver": "GTiff", "dtype": "float32", "nodata": -9999,
                "width": len(x), "height": len(y), "count": 3,
                "crs": self.crs,
                "transform": from_origin(float(x[0]), float(y[0]), res_x, res_y),
                "compress": "lzw", "tiled": True, "blockxsize": 512, "blockysize": 512,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_raster(arr, output_path, profile, descriptions=["SAR_VV", "SAR_VH", "SAR_VV_VH_ratio"])
            logger.success(f"SAR: {output_path.name}")
            return output_path

        except Exception as e:
            logger.error(f"Erro SAR para {output_path.name}: {e}")
            import traceback; logger.debug(traceback.format_exc())
            return None

    # ── Orquestrador de tiles ─────────────────────────────────────────────────

    def run_tile(self, tile: Tile, chunk_deg: float = 1.0):
        """Executa todos os downloads para um tile primário."""
        logger.info(f"{'='*50}")
        logger.info(f"Tile: {tile.parent_tile} | {tile}")
        logger.info(f"{'='*50}")

        s2_cfg = self.cfg["satellite"]["s2"]
        sar_cfg = self.cfg["satellite"].get("sar", {})
        chunk_deg = self.mpc_cfg.get("chunk_size_deg", chunk_deg)

        bboxes = self.generate_download_bboxes(tile, chunk_deg)
        logger.info(f"Bboxes para download: {len(bboxes)} chunks de {chunk_deg}°")

        s2_dir = self.out_base / "s2" / tile.parent_tile
        dem_dir = self.out_base / "dem" / tile.parent_tile
        sar_dir = self.out_base / "sar" / tile.parent_tile

        failed_chunks = []
        for i, bbox in enumerate(tqdm(bboxes, desc=f"Downloading {tile.parent_tile}")):
            bbox_str = f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
            try:
                # Sentinel-2
                s2_out = s2_dir / f"s2_composite_{bbox_str}.tif"
                if not s2_out.exists():
                    items = self.search_s2_scenes(
                        bbox=bbox,
                        date_start=s2_cfg["date_start"],
                        date_end=s2_cfg["date_end"],
                        cloud_max=s2_cfg["cloud_cover_max"],
                        dry_months_only=True,
                        max_scenes=s2_cfg.get("max_scenes", 20),
                    )
                    self.download_s2_composite(bbox, items, s2_out, bands=S2_ALL_BANDS)

                # DEM
                dem_out = dem_dir / f"dem_{bbox_str}.tif"
                self.download_dem(bbox, dem_out)

                # SAR (opcional — só se configurado)
                if sar_cfg:
                    sar_out = sar_dir / f"sar_{bbox_str}.tif"
                    self.download_sar(bbox, sar_out, sar_cfg.get("date_start", "2021-01-01"), sar_cfg.get("date_end", "2024-12-31"))

            except Exception as e:
                logger.error(f"Chunk {i+1}/{len(bboxes)} falhou: {bbox_str} — {e}")
                failed_chunks.append(bbox_str)
                continue

        if failed_chunks:
            logger.warning(f"Tile {tile.parent_tile}: {len(failed_chunks)} chunks com falha — rode novamente para reprocessar: {failed_chunks}")
        logger.success(f"Tile {tile.parent_tile} concluído: {len(bboxes) - len(failed_chunks)}/{len(bboxes)} chunks ok")

    def run_all(self, priority_first: bool = True):
        """Executa downloads para todos os segmentos na ordem norte→sul."""
        import pyproj as _pyproj

        t0 = time.time()
        logger.info("FASE 2 · Agente MPC Download — iniciando")

        grid = TileGrid(self.cfg)
        primary_tiles = grid.load_primary_tiles()

        order = (
            ["seg_norte", "seg_centro", "seg_sul"]
            if priority_first
            else primary_tiles["tile_name"].tolist()
        )

        proj_to_wgs = _pyproj.Transformer.from_crs(
            self.cfg["project"]["crs"], "EPSG:4326", always_xy=True
        )

        for tile_name in order:
            row = primary_tiles[primary_tiles["tile_name"] == tile_name]
            if len(row) == 0:
                logger.warning(f"Segmento '{tile_name}' não encontrado")
                continue
            geom = row.iloc[0].geometry
            bounds_utm = geom.bounds
            xs = [bounds_utm[0], bounds_utm[2]]
            ys = [bounds_utm[1], bounds_utm[3]]
            lons, lats = proj_to_wgs.transform(xs, ys)
            bounds_wgs84 = (min(lons), min(lats), max(lons), max(lats))

            tile = Tile(
                tile_id=tile_name,
                bounds=bounds_utm,
                bounds_wgs84=bounds_wgs84,
                parent_tile=tile_name,
                resolution_m=self.cfg["project"]["resolution"],
            )
            self.run_tile(tile)

        elapsed = time.time() - t0
        logger.success(f"Download completo em {elapsed/60:.0f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Derivadas de terreno (pós-download DEM)
# ─────────────────────────────────────────────────────────────────────────────

def compute_terrain_derivatives(config: dict):
    """
    Calcula slope, aspect, TPI, curvatura e índice topográfico (TWI)
    a partir do DEM baixado. Salva em data/05_RASTERS/terrain/.
    """
    from pathlib import Path
    import numpy as np
    import rasterio
    from scipy.ndimage import generic_filter, convolve

    dem_dir = Path(config["paths"]["downloads"]) / "dem"
    out_dir = Path(config["paths"]["rasters"]) / "terrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_files = list(dem_dir.rglob("*.tif"))
    if not dem_files:
        logger.warning("Nenhum DEM encontrado para calcular derivadas")
        return

    # Processar cada tile de DEM
    for dem_path in tqdm(dem_files, desc="Terrain derivatives"):
        stem = dem_path.stem
        tile_dir = out_dir / dem_path.parent.name
        tile_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)
            res = src.res[0]
            profile = src.profile.copy()
            profile.update(dtype="float32", count=1, nodata=-9999)

        nodata_mask = dem == -9999

        def _slope_aspect(dem, res):
            # Gradiente via diferenças centrais
            dz_dy, dz_dx = np.gradient(dem, res)
            slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
            aspect = np.degrees(np.arctan2(-dz_dx, dz_dy)) % 360
            return slope.astype(np.float32), aspect.astype(np.float32)

        slope, aspect = _slope_aspect(dem, res)
        slope[nodata_mask] = -9999
        aspect[nodata_mask] = -9999

        # TPI (Topographic Position Index) — janela 1km
        win = max(3, int(1000 / res) | 1)  # ímpar
        dem_smooth = convolve(
            np.where(nodata_mask, np.nan, dem),
            np.ones((win, win)) / win**2,
            mode="reflect",
        )
        tpi = (dem - dem_smooth).astype(np.float32)
        tpi[nodata_mask] = -9999

        # Curvatura (Laplaciano simplificado)
        from scipy.ndimage import laplace
        curv = laplace(np.where(nodata_mask, 0, dem)).astype(np.float32)
        curv[nodata_mask] = -9999

        for name, arr in [("slope", slope), ("aspect", aspect), ("tpi", tpi), ("curvature", curv)]:
            out_path = tile_dir / f"{name}_{stem}.tif"
            save_raster(arr, out_path, profile, descriptions=[name])

    logger.success(f"Derivadas de terreno salvas em {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str = None, tiles: List[str] = None):
    cfg = load_config(config_path)
    agent = MPCDownloadAgent(cfg)

    if tiles:
        import pyproj as _pyproj
        grid = TileGrid(cfg)
        primary = grid.load_primary_tiles()
        proj_to_wgs = _pyproj.Transformer.from_crs(cfg["project"]["crs"], "EPSG:4326", always_xy=True)
        for tile_name in tiles:
            row = primary[primary["tile_name"] == tile_name]
            if len(row) == 0:
                logger.warning(f"Segmento '{tile_name}' não encontrado")
                continue
            geom = row.iloc[0].geometry
            bounds_utm = geom.bounds
            xs = [bounds_utm[0], bounds_utm[2]]
            ys = [bounds_utm[1], bounds_utm[3]]
            lons, lats = proj_to_wgs.transform(xs, ys)
            tile = Tile(
                tile_id=tile_name,
                bounds=bounds_utm,
                bounds_wgs84=(min(lons), min(lats), max(lons), max(lats)),
                parent_tile=tile_name,
                resolution_m=cfg["project"]["resolution"],
            )
            agent.run_tile(tile)
    else:
        agent.run_all()

    logger.info("Calculando derivadas de terreno...")
    compute_terrain_derivatives(cfg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agente de download MPC")
    parser.add_argument("--config", default=None, help="Path para config.yaml")
    parser.add_argument("--tiles", nargs="+", default=None,
                        help="Tiles específicos: norte centro_n centro_s sul")
    args = parser.parse_args()
    run(config_path=args.config, tiles=args.tiles)
