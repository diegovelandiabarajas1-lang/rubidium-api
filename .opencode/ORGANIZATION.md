# Organización Rubidium — Arquitectura

## Jerarquía

```
Usuario → CEO → CTO
                 ├── Architecture Director
                 │    └── Rust Expert
                 ├── Training Director
                 │    ├── Data Lead
                 │    └── Cuda Expert
                 ├── Performance Director
                 │    ├── Cuda Expert
                 │    ├── Memory Expert
                 │    └── Profiler Expert
                 └── Deployment Director
                      └── DevOps Lead
```

## Flujo de trabajo

1. **CEO** recibe objetivo del usuario → crea tarea en `queue/pending/`
2. **CTO** toma la tarea → la asigna al Director correspondiente
3. **Director** → asigna a Lead → Lead asigna a Especialistas
4. **Especialista** → escribe en `reports/raw/`
5. **Lead** → lee `reports/raw/` → escribe `reports/summaries/`
6. **Director** → lee `reports/summaries/` → escribe `reports/final/`
7. **CTO** → lee `reports/final/` → reporta al CEO

## Reglas
- Nadie salta niveles
- Cada nivel solo lee el nivel inmediato inferior
- Los informes son el único medio de comunicación
- Las decisiones se documentan en `reports/decisions/`
- Las tareas se mueven: pending → working → review → completed
