"""Utilitários de GPU para RTX 4090."""

import torch
from loguru import logger


def get_device(force_cpu: bool = False) -> torch.device:
    """Retorna device CUDA se disponível, com log de info da GPU."""
    if force_cpu:
        logger.warning("Forçando CPU (force_cpu=True)")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        logger.warning("CUDA não disponível — usando CPU")
        return torch.device("cpu")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory / 1e9
    logger.info(
        f"GPU: {props.name} | VRAM: {vram_gb:.1f} GB | "
        f"CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}"
    )
    torch.backends.cudnn.benchmark = True
    return device


def log_gpu_memory(prefix: str = "") -> dict:
    """Loga uso atual de VRAM e retorna dict com valores."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved
    info = {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
    }
    label = f"[{prefix}] " if prefix else ""
    logger.debug(
        f"{label}VRAM: {allocated:.1f}/{total:.1f} GB alocado | {free:.1f} GB livre"
    )
    return info


def clear_gpu_cache():
    """Limpa cache CUDA entre processamentos de tiles."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def optimal_batch_size(model_vram_gb: float = 4.0, total_vram_gb: float = 24.0) -> int:
    """
    Estima batch size ótimo para Prithvi na RTX 4090.
    Reserva 20% para overhead e outros tensores.
    """
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    available = total_vram_gb * 0.80 - model_vram_gb
    # Prithvi patch 224x224x6 em fp16 + ativações ≈ ~150MB por item
    patch_mb = 150
    batch = max(1, int((available * 1024) / patch_mb))
    batch = min(batch, 128)  # RTX 4090 25.8GB suporta até 128
    logger.info(f"Batch size estimado para Prithvi: {batch}")
    return batch
