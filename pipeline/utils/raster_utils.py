"""Utilitários de raster compartilhados entre todos os módulos do pipeline."""

import yaml
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional

import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from loguru import logger


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "00_config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_profile(
    bounds: Tuple[float, float, float, float],
    resolution: float,
    crs: str,
    count: int = 1,
    dtype: str = "float32",
    nodata: float = -9999,
) -> dict:
    """Gera profile rasterio padrão do projeto."""
    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }


def save_raster(
    array: np.ndarray,
    path: Union[str, Path],
    profile: dict,
    descriptions: List[str] = None,
) -> Path:
    """Salva array numpy como GeoTIFF com o profile dado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[np.newaxis, :]
    profile = profile.copy()
    profile["count"] = array.shape[0]

    # Force remove se arquivo existe (contorna locks no Windows)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        if descriptions:
            for i, desc in enumerate(descriptions, 1):
                dst.update_tags(i, name=desc)
    logger.info(f"Salvo: {path} | shape={array.shape} | dtype={array.dtype}")
    return path


def reproject_to_profile(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    dst_profile: dict,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reprojeta e reamostra raster para o profile de destino."""
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_profile["transform"],
                    dst_crs=dst_profile["crs"],
                    resampling=resampling,
                )
    return dst_path


def align_rasters(
    paths: List[Union[str, Path]],
    reference_path: Union[str, Path],
    output_dir: Union[str, Path],
    resampling: Resampling = Resampling.bilinear,
) -> List[Path]:
    """Alinha lista de rasters ao profile de referência."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile
    aligned = []
    for p in paths:
        p = Path(p)
        out = output_dir / p.name
        reproject_to_profile(p, out, ref_profile, resampling)
        aligned.append(out)
        logger.debug(f"Alinhado: {p.name}")
    return aligned


def load_aoi_geometry(config: dict) -> gpd.GeoDataFrame:
    """Carrega AOI principal via fiona (evita bug shapely create_collection)."""
    import fiona, pyproj
    from shapely.geometry import shape
    from shapely import wkb as swkb
    from shapely.ops import transform as shp_transform

    shp_path = config["paths"]["aoi_shp"]
    target_crs = config["project"]["crs"]
    records = []
    with fiona.open(str(shp_path)) as src:
        src_crs = src.crs or target_crs
        transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
        for feat in src:
            try:
                geom = shape(feat["geometry"])
                geom = swkb.loads(geom.wkb)
                geom_t = shp_transform(transformer.transform, geom)
                records.append({"geometry": geom_t})
            except Exception:
                pass
    return gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)


def load_tile_geometries(config: dict) -> gpd.GeoDataFrame:
    """Carrega os 4 tiles N/C/S como GeoDataFrame. Retorna vazio se shapely falhar."""
    try:
        import pyproj
        from shapely.ops import transform as shp_transform
        import fiona

        gpkg = config["paths"]["aoi_tiles"]
        layer = config["paths"]["aoi_tiles_layer"]
        target_crs = config["project"]["crs"]

        with fiona.open(gpkg, layer=layer) as src:
            src_crs = src.crs_wkt or src.crs
            transformer = pyproj.Transformer.from_crs(
                src_crs, target_crs, always_xy=True
            )
            records = []
            for feat in src:
                try:
                    from shapely.geometry import shape
                    from shapely import wkb as swkb
                    geom = shape(feat["geometry"])
                    geom_repr = swkb.loads(geom.wkb)
                    geom_t = shp_transform(transformer.transform, geom_repr)
                    props = dict(feat["properties"])
                    props["geometry"] = geom_t
                    records.append(props)
                except Exception:
                    pass
        if not records:
            return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)
        return gdf
    except Exception as e:
        import warnings
        warnings.warn(f"load_tile_geometries falhou ({e}) — bounds calculados dos patches")
        return gpd.GeoDataFrame()


def clip_raster_to_geometry(
    src_path: Union[str, Path],
    geometry,
    dst_path: Union[str, Path],
    nodata: float = -9999,
) -> Path:
    """Recorta raster ao polígono de geometria."""
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        geoms = [geometry] if hasattr(geometry, "geoms") else [geometry]
        out_image, out_transform = rio_mask(src, geoms, crop=True, nodata=nodata)
        out_profile = src.profile.copy()
        out_profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            nodata=nodata,
        )
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(out_image)
    return dst_path


def mosaic_rasters(
    paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    nodata: float = -9999,
) -> Path:
    """Mosaica lista de rasters GeoTIFF em um único arquivo."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [rasterio.open(p) for p in paths]
    mosaic, transform = merge(datasets, nodata=nodata)
    profile = datasets[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        nodata=nodata,
    )
    for ds in datasets:
        ds.close()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)
    logger.info(f"Mosaico salvo: {output_path}")
    return output_path


def stack_rasters(
    paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    descriptions: List[str] = None,
    nodata: float = -9999,
) -> Path:
    """Empilha rasters single-band em um multi-band GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    profile = None
    for p in paths:
        with rasterio.open(p) as src:
            arrays.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile.copy()
    stack = np.stack(arrays, axis=0)
    stack = np.where(np.isnan(stack), nodata, stack)
    profile.update(count=len(arrays), dtype="float32", nodata=nodata)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stack)
        if descriptions:
            for i, desc in enumerate(descriptions, 1):
                dst.update_tags(i, name=desc)
    logger.info(f"Stack salvo: {output_path} | {len(arrays)} bandas")
    return output_path
