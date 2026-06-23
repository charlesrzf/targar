# Setup e Execução — Pipeline Cu Targeting Argentina

## 1. Criar ambiente Conda

```bat
cd D:\argentina\pipeline
conda env create -f environment.yml
conda activate cu-targeting
```

> Primeira vez: ~15 min (PyTorch CUDA 12.1 + dependências geoespaciais)

## 2. Verificar GPU

```python
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')"
```

Esperado: `NVIDIA GeForce RTX 4090 | 24.0 GB`

## 3. Verificar dados locais

```python
python -c "
from utils.raster_utils import load_config, load_aoi_geometry
cfg = load_config()
aoi = load_aoi_geometry(cfg)
print('AOI ok:', aoi.total_bounds)
"
```

## 4. Executar Pipeline Completo

```bat
conda activate cu-targeting
cd D:\argentina\pipeline
python run_pipeline.py
```

### Executar fase por fase (recomendado para primeira execução):

```bat
# Fase 1: Processar dados SEGEMAR locais (~20-40 min)
python run_pipeline.py --phase 1

# Fase 2: Download via Planetary Computer (~2-6h dependendo da internet)
# Dica: começar pelo tile norte (prioridade máxima)
python run_pipeline.py --phase 2 --tiles norte centro_n

# Fase 3: Índices espectrais (~30 min)
python run_pipeline.py --phase 3

# Fase 4: Embeddings Prithvi na RTX 4090 (~2-4h por tile)
python run_pipeline.py --phase 4 --tiles norte

# Fase 5: Labels + Treino (~1-2h com Optuna 50 trials)
python run_pipeline.py --phase 5

# Fase 6: Inferência + Exportação (~30 min)
python run_pipeline.py --phase 6
```

### Retomar de onde parou:

```bat
python run_pipeline.py --resume
python run_pipeline.py --start-from 4
```

## 5. Saídas

```
D:\argentina\data\08_OUTPUT\
  geotiff\
    favorabilidade_porfiro_AOI.tif     ← PRIORIDADE
    favorabilidade_skarn_AOI.tif
    favorabilidade_epitermal_AOI.tif
    favorabilidade_manto_AOI.tif
    favorabilidade_COMPOSTA_AOI.tif    ← máximo entre tipos
  shapefiles\
    top_targets.gpkg                   ← 50 melhores alvos
  reports\
    ranking_targets.csv
    model_metrics.csv
    mapa_interativo.html               ← abrir no browser
```

## 6. Carregar no QGIS

1. Arrastar `favorabilidade_porfiro_AOI.tif` para o QGIS
2. Simbologia: Pseudocolor → RdYlGn → valores 0-1
3. Adicionar `top_targets.gpkg` como camada de polígonos
4. Sobrepor dados SEGEMAR de `02_SEGEMAR\` para contexto geológico

## Notas

- **Prithvi download**: ~1.2 GB do HuggingFace na primeira execução da Fase 4
- **Espaço em disco estimado**: ~80-150 GB para dados S2/DEM/SAR de toda a AOI
- **Checkpoint**: progresso salvo em `D:\argentina\logs\.pipeline_checkpoint.json`
- **Logs**: `D:\argentina\logs\` — verificar em caso de erro
