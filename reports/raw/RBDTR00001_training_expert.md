# @TrainingExpert — Informe RBDTR00001

## Trabajo realizado
- Modificado `train_pytorch.py`: max_steps 20000 → 100000, warmup 2000 → 4000
- Creado `kaggle_push_100k.py` para subir y ejecutar notebook
- Notebook: diegovelandiabarajas/rubidium-pytorch-train

## Cambios específicos
- BS=8, GA=4 (igual)
- max_steps=100000 (antes 20000)
- warmup=4000 (antes 2000)
- lr=3e-4 con cosine schedule (igual)
- Estimado: ~53 min en GPU P100

## Reportes generados
- `reports/raw/RBDTR00001_training_expert.md`
