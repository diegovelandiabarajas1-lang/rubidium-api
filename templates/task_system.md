# Sistema de Tareas — Rubidium AI

## Formato de ID
`XXXXXX00000` — 6 letras (A-Z sin Ñ) + 5 dígitos (sin 00000)

Las letras identifican el tipo de tarea. Ejemplos:
- `RBDTR00001` → Rubidium Training
- `RBDUI00001` → Rubidium UI
- `RBDIN00001` → Rubidium Inference
- `RBDDP00001` → Rubidium Deployment

## Flujo
1. Usuario pide tarea → @CTO crea en `queue/pending/` con ID
2. @CTO asigna Director → mueve a `queue/working/`
3. Director asigna Lead → Lead asigna Especialista(s)
4. Especialista ejecuta → escribe `reports/raw/<ID>_<expert>.md`
5. Lead lee raw → escribe `reports/summaries/<ID>_<lead>.md`
6. Director lee summary → escribe `reports/final/<ID>_<director>.md`
7. CTO lee final → reporta al Usuario
8. Tarea se mueve a `queue/completed/`

## Reportes
- `reports/raw/` — Informes técnicos detallados (especialistas)
- `reports/summaries/` — Resúmenes por Lead
- `reports/final/` — Decisión final por Director
