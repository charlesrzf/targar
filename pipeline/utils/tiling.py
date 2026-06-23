"""Gerenciamento de tiles para processamento de área grande."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
from dataclasses import dataclass, field

import fiona
import pyproj
from shapely.geometry import box, shape
from shapely import wkb as swkb
from shapely.ops import transform as shp_transform
import geopandas as gpd
from loguru import logger


@dataclass
class Tile:
    tile_id: str
    bounds: Tuple[float, float, float, float]  # minx miny maxx maxy em UTM
    bounds_wgs84: Tuple[float, float, float, float]
    crs: str = "EPSG:32719"
    patch_size_px: int = 224
    resolution_m: float = 30.0
    row: int = 0
    col: int = 0
    parent_tile: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def width_m(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height_m(self) -> float:
        return self.bounds[3] - self.bounds[1]

    @property
    def width_px(self) -> int:
        return int(self.width_m / self.resolution_m)

    @property
    def height_px(self) -> int:
        return int(self.height_m / self.resolution_m)

    @property
    def geometry(self):
        return box(*self.bounds)

    def output_dir(self, base_dir: Path) -> Path:
        return base_dir / self.parent_tile / self.tile_id

    def __str__(self):
        return (
            f"Tile({self.tile_id} | {self.width_px}x{self.height_px}px "
            f"| {self.width_m/1000:.1f}x{self.height_m/1000:.1f}km)"
        )


class TileGrid:
    """
    Usa AOI_Segmentos.shp (3 faixas cobrindo a AOI completa) como tiles primários.
    Gera patches 224×224px em resolução S2 nativa (30m) para extração de embeddings.
    """

    # Nomes dos 3 segmentos na ordem norte→sul (como apparecem no shapefile)
    SEGMENT_NAMES = ["seg_norte", "seg_centro", "seg_sul"]
    SEGMENT_PRIORITY = {"seg_norte": 0, "seg_centro": 1, "seg_sul": 2}

    def __init__(self, config: dict):
        self.config = config
        self.resolution = config["project"]["resolution"]
        self.crs = config["project"]["crs"]
        self.patch_px = config["prithvi"]["patch_size"]
        # Patches são sempre extraídos a 30m (resolução nativa S2/Prithvi)
        self.patch_m = self.patch_px * 30  # 224 * 30 = 6720m
        self._primary_tiles: Optional[gpd.GeoDataFrame] = None
        self._all_patches: List[Tile] = []

    def load_primary_tiles(self) -> gpd.GeoDataFrame:
        """Carrega AOI_Segmentos.shp via fiona (evita bug shapely create_collection)."""
        shp_path = self.config["paths"]["aoi_tiles"]
        target_crs = self.crs

        records = []
        with fiona.open(shp_path) as src:
            src_crs = src.crs or "EPSG:32719"
            transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
            for feat in src:
                try:
                    geom = shape(feat["geometry"])
                    geom = swkb.loads(geom.wkb)
                    geom_t = shp_transform(transformer.transform, geom)
                    records.append({"geometry": geom_t})
                except Exception as e:
                    logger.warning(f"Segmento ignorado: {e}")

        if not records:
            raise RuntimeError(f"AOI_Segmentos vazio ou não lido: {shp_path}")

        # Ordenar por Y decrescente (norte → sul) para atribuir nomes corretos
        records.sort(key=lambda r: -r["geometry"].bounds[3])

        names = self.SEGMENT_NAMES[: len(records)]
        for r, n in zip(records, names):
            r["tile_name"] = n

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
        self._primary_tiles = gdf
        for _, row in gdf.iterrows():
            b = row.geometry.bounds
            logger.info(
                f"Segmento '{row['tile_name']}': "
                f"X {b[0]:.0f}–{b[2]:.0f}  Y {b[1]:.0f}–{b[3]:.0f}"
            )
        return gdf

    def generate_patches(self, primary_tile_name: str = None) -> List[Tile]:
        """
        Gera grid de patches 224×224 @ 30m sobre os segmentos.
        Patches são gerados sempre a 30m (resolução Prithvi); saída depois é reamostrada.
        """
        if self._primary_tiles is None:
            self.load_primary_tiles()

        tiles_df = self._primary_tiles
        if primary_tile_name:
            tiles_df = tiles_df[tiles_df["tile_name"] == primary_tile_name]

        # Conversor UTM→WGS84 para gerar bounds_wgs84 por patch
        proj_to_wgs = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

        patches = []
        for _, row in tiles_df.iterrows():
            tile_name = row["tile_name"]
            bounds = row.geometry.bounds  # minx miny maxx maxy

            # Alinhar ao grid de patches
            minx = np.floor(bounds[0] / self.patch_m) * self.patch_m
            miny = np.floor(bounds[1] / self.patch_m) * self.patch_m
            maxx = np.ceil(bounds[2] / self.patch_m) * self.patch_m
            maxy = np.ceil(bounds[3] / self.patch_m) * self.patch_m

            cols = np.arange(minx, maxx, self.patch_m)
            rows = np.arange(miny, maxy, self.patch_m)

            n_patches = 0
            for r_idx, y0 in enumerate(rows):
                for c_idx, x0 in enumerate(cols):
                    patch_bounds = (x0, y0, x0 + self.patch_m, y0 + self.patch_m)
                    patch_geom = box(*patch_bounds)

                    if not patch_geom.intersects(row.geometry):
                        continue

                    # bounds WGS84 via pyproj (sem geopandas.to_crs)
                    corners_x = [patch_bounds[0], patch_bounds[2]]
                    corners_y = [patch_bounds[1], patch_bounds[3]]
                    lons, lats = proj_to_wgs.transform(corners_x, corners_y)
                    b84 = (min(lons), min(lats), max(lons), max(lats))

                    patch_id = f"{tile_name}_r{r_idx:03d}_c{c_idx:03d}"
                    patch = Tile(
                        tile_id=patch_id,
                        bounds=patch_bounds,
                        bounds_wgs84=b84,
                        crs=self.crs,
                        patch_size_px=self.patch_px,
                        resolution_m=30.0,  # sempre 30m para Prithvi
                        row=r_idx,
                        col=c_idx,
                        parent_tile=tile_name,
                    )
                    patches.append(patch)
                    n_patches += 1

            logger.info(f"Segmento '{tile_name}': {n_patches} patches gerados")

        self._all_patches = patches
        logger.info(f"Total de patches: {len(patches)}")
        return patches

    def patches_to_geodataframe(self, patches: List[Tile] = None) -> gpd.GeoDataFrame:
        if patches is None:
            patches = self._all_patches
        records = []
        for p in patches:
            records.append({
                "tile_id": p.tile_id,
                "parent": p.parent_tile,
                "row": p.row,
                "col": p.col,
                "width_px": p.width_px,
                "height_px": p.height_px,
                "geometry": p.geometry,
            })
        return gpd.GeoDataFrame(records, crs=self.crs)

    def save_tile_index(self, output_path: Path) -> Path:
        gdf = self.patches_to_geodataframe()
        gdf.to_file(output_path, driver="GPKG")
        logger.info(f"Índice de tiles salvo: {output_path} ({len(gdf)} patches)")
        return output_path

    def iter_tiles_by_priority(self) -> Iterator[Tile]:
        return iter(sorted(
            self._all_patches,
            key=lambda t: self.SEGMENT_PRIORITY.get(t.parent_tile, 9)
        ))
