# Rust Expert — Informe TASK_002

## Trabajo realizado
- Cargado modelo 14.4M params en Rust
- Corregida transposición de pesos FFN (w1_w, w2_w) y lm_w
- Recompilado rubidium-core

## Resultados
| Métrica | Valor |
|---------|-------|
| Carga de modelo | ✅ Éxito |
| Generación | ✅ Funciona |
| Velocidad | ~11.6 chars/s (CPU) |
| Calidad | ❌ Mala (loss=3.02, sin entrenar suficiente) |

## Problemas
- El modelo con loss=3.02 no genera texto coherente. Necesita entrenar 50K-100K+ steps
- La inferencia en Rust (CPU) es lenta para el modelo de 14.4M

## Recomendaciones
1. Entrenar más steps en Kaggle (100K steps)
2. Optimizar inferencia Rust (paralelizar, matmul optimizado)
3. O desplegar modelo parcial y re-entrenar después
