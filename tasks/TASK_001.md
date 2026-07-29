# TASK_001: Monitorear y optimizar entrenamiento Kaggle

Responsable: Training Lead
Dependencias: []
Estado: WORKING

## Objetivo
Monitorear el notebook `rubidium-pytorch-train` en Kaggle. Si falla o supera 27 min, optimizar para <10 min.

## Contexto
- GPU: Tesla P100 (sm_60) — incompatible con PyTorch CUDA 12.x
- Solución actual: instalar `torch==2.4.0+cu118` vía subprocess
- Modelo: 14.4M params, 20K steps, batch=8, grad_accum=4
- Dataset: 2.1M chars, 44 archivos

## Resultado esperado
Training completo en GPU con modelo guardado en `/kaggle/working/model_10m_final.pkl`

## Informes
- reports/TASK_001_kaggle_status.md
