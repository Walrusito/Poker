# Poker CFR AI

Proyecto de investigacion y entrenamiento para un agente de No-Limit Texas Hold'em inspirado en Deep CFR, con soporte multiway (`2-9` jugadores), evaluacion contra varios baselines y ejecucion principal via Docker.

No es un solver GTO exacto. El repositorio implementa una aproximacion basada en:

- abstraccion discreta de acciones,
- features ingenierizadas en lugar de informacion perfecta cruda,
- estimacion de equity mediante LUT + Monte Carlo,
- dos redes compartidas entre asientos (`RegretNet` y `PolicyNet`),
- self-play con rollouts para estimar utilidades de acciones no muestreadas.

## Estado actual (abril 2026)

El proyecto esta en una fase **estable para investigacion aplicada**, con pipeline de entrenamiento/evaluacion operativo en Docker, pero aun lejos de un solver teoricamente canonico o una plataforma de produccion.

### Lo que ya esta consolidado

- Entrenamiento end-to-end con `main.py` + `DeepCFRTrainer`, incluyendo warmup preflop de LUT al inicio de la run.
- Reanudacion de runs via `CheckpointManager` con validaciones de compatibilidad de entorno, arquitectura y parametros clave.
- Evaluacion contra multiples baselines (`random`, `heuristic`, `snapshot_pool`, `population`) con registro de metricas por iteracion.
- Cross-play entre runs con `cross_play_matrix.py`, seleccion de top-runs y export opcional a CSV.
- Persistencia dual de checkpoints:
  - `*.pt` completos para continuar entrenamiento.
  - `*_policy.pt` ligeros para evaluacion/cross-play mas rapido.
- Integracion con MLflow (tags, params, artefactos, estado final `FINISHED`/`FAILED`) y trazabilidad de comando en `artifacts/last_run_command.txt`.

### Limites actuales (importante)

- Sigue siendo una aproximacion estilo Deep CFR con abstracciones y aproximaciones de equity; no hay garantia de convergencia Nash multiway.
- `exploitability_proxy`, `robust_score` y metricas similares son indicadores operativos, no medidas teoricas exactas.
- El rendimiento todavia esta condicionado por copias profundas del entorno, caches no siempre acotadas y coste de I/O de LUT/checkpoints.

## Objetivo del proyecto

El objetivo practico es entrenar, evaluar y comparar politicas aproximadas de poker bajo un framework reproducible:

- entrenar politicas por self-play,
- evaluar contra agentes random, heuristicas y snapshots historicos,
- guardar checkpoints y reanudar runs,
- comparar runs y construir matrices de cross-play.

## Arquitectura

### Flujo de alto nivel

1. `main.py` parsea argumentos, fija seeds y configura runtime de Torch.
2. `CheckpointManager` prepara la run y decide si se reanuda.
3. `PokerEnv` crea el entorno NLHE con stacks, ciegas, side pots y orden de accion.
4. `DeepCFRTrainer` instancia:
   - `InformationSetBuilder` para features
   - `CardAbstraction` / `EquityLUT` / `HandEquity` para equity
   - `RegretNet` para regrets por accion
   - `PolicyNet` para distribuciones de accion
   - buffers reservoir para datos de regret y policy
5. El trainer ejecuta self-play y rollouts.
6. Se entrenan ambas redes con PyTorch.
7. Se evalua la politica contra distintos baselines.
8. Se registran metricas y se guardan checkpoints.

### Modulos principales

- `main.py`
  Entry point de entrenamiento y warmup de LUT.

- `train/train_deep_cfr.py`
  Orquestacion principal de self-play, rollouts, entrenamiento, evaluacion y checkpointing.

- `env/poker_env.py`
  Entorno de una mano de NLHE con `preflop`, `flop`, `turn`, `river`, `showdown`, all-ins y side pots.

- `env/vectorized_poker_env.py`
  Wrapper por lotes para inferencia y rollouts sobre multiples entornos.

- `utils/information_set.py`
  Construccion del vector de features usado por las redes.

- `utils/equity_lut.py`
  LUT incremental para equity preflop y postflop bucketizada.

- `utils/hand_equity.py`
  Fallback de equity por Monte Carlo y calculo exacto en river heads-up.

- `models/regret_net.py`
  MLP para predecir regrets por accion.

- `models/policy_net.py`
  MLP para predecir una distribucion de accion.

- `utils/checkpointing.py`
  Gestion de checkpoints completos, snapshots ligeros de politica y `run_summary.json`.

- `compare_runs.py`
  Ranking tabular de runs.

- `cross_play_matrix.py`
  Matriz hero-vs-field entre checkpoints.

## Estructura del repositorio

```text
.
|-- main.py
|-- train/
|   `-- train_deep_cfr.py
|-- env/
|   |-- poker_env.py
|   |-- vectorized_poker_env.py
|   |-- rules.py
|   `-- deck.py
|-- models/
|   |-- regret_net.py
|   `-- policy_net.py
|-- data/
|   |-- buffers.py
|   |-- dataset.py
|   `-- lut/
|-- utils/
|   |-- information_set.py
|   |-- equity_lut.py
|   |-- hand_equity.py
|   |-- card_abstraction.py
|   |-- bet_sizing_abstraction.py
|   |-- checkpointing.py
|   |-- command_persistence.py
|   `-- run_comparison.py
|-- tests/
|-- Dockerfile
`-- docker-compose.yml
```

## Reglas de poker soportadas

- No-Limit Texas Hold'em
- `2-9` jugadores
- ciegas y boton rotativo
- `preflop`, `flop`, `turn`, `river`, `showdown`
- side pots por capas para all-ins multiway
- utilidades terminales en big blinds (`reward_unit="bb"`)

No incluye:

- rake
- antes
- torneos / ICM
- abstraccion continua de sizings
- solving exacto del arbol completo

## Espacio de acciones

El proyecto usa una abstraccion discreta. Las acciones legales dependen de `street`, stack, `to_call`, `min_raise` y SPR.

Acciones base:

- `fold`
- `check`
- `call`
- `bet_25`
- `bet_50`
- `bet_75`
- `bet_100`
- `bet_125`
- `bet_200`
- `all_in`

`BetSizingAbstraction` traduce estas etiquetas a cantidades concretas.

## Features del estado

`InformationSetBuilder` produce un vector de dimension fija con señales como:

- equity estimada
- bucket de equity
- pot odds
- implied odds
- SPR normalizado
- pot size, `to_call` y `last_raise_size` normalizados
- street y posicion relativa
- numero de jugadores activos y por actuar
- flags de agresor, boton, SB y BB
- stack y contribucion del hero
- textura del board
- blockers y heuristica de nut potential
- agresion reciente

Esto reduce el problema, pero tambien limita la fidelidad estrategica frente al juego real.

## Entrenamiento

### Self-play

Durante cada episodio:

- se recorre la mano siguiendo la politica derivada de regrets
- se estima la utilidad de la accion muestreada
- se estiman acciones alternativas por rollout
- se guardan muestras en `advantage_buffer`
- se guardan muestras en `policy_buffer`

### Redes

- `RegretNet`
  Aprende regrets por accion desde `advantage_buffer`.

- `PolicyNet`
  Aprende una distribucion objetivo desde `policy_buffer`.

### Evaluacion

El trainer calcula:

- self-play
- `vs_random`
- `vs_heuristic`
- `vs_heuristic_pool`
- `vs_snapshot_pool`
- `vs_population`

Metricas destacadas:

- `vs_random_bb_per_100`
- `vs_heuristic_bb_per_100`
- `vs_snapshot_bb_per_100`
- `vs_population_bb_per_100`
- `avg_policy_entropy`
- `avg_abs_regret`
- `robust_score`
- `exploitability_proxy`

## Como interpretar `exploitability_proxy` y `robust_score`

Estas dos metricas se deben leer en **2 capas**:

1. **Capa practica (operativa)**: sirven para detectar degradaciones, comparar runs y priorizar decisiones de entrenamiento.
2. **Capa teorica (limite actual)**: no equivalen a exploitability exacta ni demuestran convergencia a equilibrio de Nash.

### Regla de oro

- No tomes decisiones con una sola metrica aislada.
- Evalua siempre junto con `vs_random_bb_per_100`, `vs_heuristic_bb_per_100`, `vs_snapshot_bb_per_100` y `vs_population_bb_per_100`.
- Si puedes, compara con el mismo benchmark/seed para reducir varianza.

### Reach weighting y clipping (nuevo)

- El trainer ahora admite weighting CFR-style por reach para regrets/policy (`--disable-reach-weighting`, `--reach-weight-mode`).
- El clipping puede ser fijo (`--reach-weight-clip`) o automatico por cuantiles (`--reach-auto-clip-quantile`, activo salvo `--disable-reach-auto-clip`).
- **Valor optimo recomendado**: en esta implementacion los pesos de reach estan acotados en `[0,1]`, por lo que el clip teorico optimo es `1.0` (cualquier valor mayor no aporta recorte real).
- En ejecucion se reporta `reach_weight_clip_optimal` y percentiles (`reach_weight_raw_p50/p95/p99`) para validar si conviene mantener `1.0` o bajar el umbral en modo conservador.

### Rangos recomendados para `exploitability_proxy`

**Interpretacion general**: menor suele ser mejor, porque sugiere menos vulnerabilidad relativa bajo el proxy actual.

- `< 0.10`:
  - Lectura: muy buena senal operativa.
  - Por que: la politica aparenta consistencia y menos huecos explotables bajo la evaluacion actual.
  - Que hacer: validar que tambien mejora o mantiene `bb/100` en `snapshot/population` antes de promoverla.
- `0.10 - 0.30`:
  - Lectura: razonable para investigacion; aun hay margen de mejora.
  - Por que: suele reflejar compromiso entre exploracion y estabilidad de politica.
  - Que hacer: continuar entrenamiento y revisar tendencia por iteraciones (no solo valor puntual).
- `0.30 - 0.60`:
  - Lectura: zona de alerta moderada.
  - Por que: puede indicar abstraccion insuficiente, sobreajuste al self-play o ruido de estimacion.
  - Que hacer: revisar regimen (entropia, smoothing, epochs), y comprobar si cross-play tambien cae.
- `> 0.60`:
  - Lectura: alerta alta.
  - Por que: la politica probablemente presenta patrones explotables fuertes bajo este esquema de medida.
  - Que hacer: no usar como candidato principal; auditar entrenamiento, evaluacion y calidad de muestras.

### Rangos recomendados para `robust_score`

**Interpretacion general**: mayor suele ser mejor, porque resume robustez empirica frente a baselines/pools.

- `> 0.65`:
  - Lectura: robustez empirica alta.
  - Por que: la politica suele sostener rendimiento en varios escenarios, no solo en uno.
  - Que hacer: candidata a `best_robust`, pero confirmar con cross-play y varianza.
- `0.45 - 0.65`:
  - Lectura: robustez intermedia aceptable.
  - Por que: suele haber fortalezas parciales, con sensibilidad a tipo de rival o configuracion.
  - Que hacer: mejorar generalizacion (pool mas diverso, evaluacion estable, no-regresion).
- `0.25 - 0.45`:
  - Lectura: robustez debil.
  - Por que: el agente puede estar aprendiendo lineas fragiles o demasiado dependientes del entorno de entrenamiento.
  - Que hacer: revisar diversidad de oponentes, abstraccion de acciones/features y estabilidad de entrenamiento.
- `< 0.25`:
  - Lectura: robustez muy baja.
  - Por que: alta probabilidad de degradacion fuera del caso entrenado.
  - Que hacer: tratar como run experimental/no promotable hasta corregir causas.

### Como usar ambos juntos (decision practica)

- **Caso A (bueno)**: `exploitability_proxy` baja + `robust_score` alta -> candidato fuerte.
- **Caso B (inestable)**: `exploitability_proxy` baja + `robust_score` baja -> posible sobreajuste al proxy; mirar cross-play y pools.
- **Caso C (ruidoso)**: `exploitability_proxy` alta + `robust_score` alta -> rendimiento empirico bueno pero con vulnerabilidades detectadas; vigilar regresiones.
- **Caso D (malo)**: `exploitability_proxy` alta + `robust_score` baja -> descartar para promocion.

### Siguiente salto recomendado (capa teorica)

Para acercar estas lecturas a teoria de juegos:

- introducir una aproximacion de best response por sampling,
- reforzar weighting por reach probability en el trainer,
- mantener benchmarks congelados (misma seed y mismos rivales) para comparar runs con menos ruido.

## Requisitos

### Docker

La forma recomendada de ejecutar el proyecto es Docker Compose.

- Docker Desktop con soporte para GPU si se va a usar CUDA
- `docker compose`

### Dependencias Python

`requirements.txt`:

- `torch`
- `numpy`
- `mlflow`
- `tqdm`
- `pytest`
- `eval7`

## Uso

### Ejecutar tests

```bash
docker compose run --rm --build poker-tests
```

### Entrenar

Comando base:

```bash
docker compose run --rm --build poker-ai --iterations 20 --episodes 2000 --eval-hands 5000
```

Ejemplo corto:

```bash
docker compose run --rm poker-ai --iterations 3 --episodes 50 --eval-hands 100 --players 2
```

Smoke test usado en validacion:

```bash
docker compose run --rm --build poker-ai --iterations 1 --episodes 2 --eval-hands 20
```

### Abrir MLflow UI

```bash
docker compose up mlflow-ui
```

Luego abre `http://localhost:5000`.

### Script auxiliar en Windows

```cmd
train_with_mlflow.cmd --iterations 10 --episodes 200 --players 4
```

### Comparar runs

```bash
docker compose run --rm --entrypoint python poker-ai compare_runs.py --checkpoint-dir artifacts/checkpoints --experiment poker_cfr_ai
```

### Construir matriz de cross-play

```bash
docker compose run --rm --entrypoint python poker-ai cross_play_matrix.py --checkpoint-dir artifacts/checkpoints --experiment poker_cfr_ai --top-runs 4 --hands 120
```

## Flags utiles

- `--players`
- `--starting-stack`
- `--small-blind`
- `--big-blind`
- `--mc-simulations`
- `--lut-simulations`
- `--rollouts-per-action`
- `--feature-cache-size`
- `--batch-size`
- `--regret-epochs`
- `--policy-epochs`
- `--parallel-workers`
- `--dataloader-workers`
- `--rollout-batch-size`
- `--snapshot-pool-size`
- `--population-run-limit`
- `--population-mix-prob`
- `--policy-smoothing-alpha`
- `--entropy-regularization`
- `--disable-reach-weighting`
- `--reach-weight-mode` (`linear` / `sqrt`)
- `--reach-weight-clip`
- `--disable-reach-auto-clip`
- `--reach-auto-clip-quantile`
- `--checkpoint-dir`
- `--resume-mode`
- `--seed`
- `--experiment`

## Checkpoints y artefactos

Cada run escribe en `artifacts/checkpoints/<experimento>/<run_name>/`:

- `latest.pt`
- `best.pt`
- `best_robust.pt`
- `iter_XXXX.pt`
- `latest_policy.pt`
- `best_policy.pt`
- `best_robust_policy.pt`
- `policy_iter_XXXX.pt`
- `run_summary.json`

Notas:

- Los `*.pt` completos conservan redes, optimizadores, buffers y RNG para reanudar entrenamiento.
- Los `*_policy.pt` guardan solo lo necesario para reconstruir `PolicyNet` y acelerar evaluacion, snapshot pool y cross-play.
- En checkpoints nuevos ya no se persiste `env_state` completo; se guarda `env_config`, que es suficiente para validar compatibilidad sin pagar el coste de serializar el entorno entero.
- Tambien se persiste el comando ejecutado en `artifacts/last_run_command.txt` o en la ruta configurada con `--command-file`.

## Limitaciones conocidas

### Correctitud

- El proyecto sigue usando una aproximacion de juego y de regret, no un solving exacto del arbol.
- La validacion de `resume` ya bloquea incompatibilidades de entorno, arquitectura y varios hiperparametros criticos, pero todavia no cubre absolutamente toda la configuracion operativa posible.

### Teoria de juegos / poker

- No hay calculo real de exploitability ni best response.
- El algoritmo no implementa Deep CFR canonico con todas sus ponderaciones de reach y counterfactual values.
- La abstraccion de acciones y de features puede inducir estrategias explotables fuera del dominio entrenado.

### Rendimiento

- El recorrido usa muchos `copy.deepcopy`.
- Los caches de equity exacta no estan acotados en memoria.
- Los checkpoints completos siguen guardando buffers completos, lo que puede crecer bastante en disco, aunque ahora se evita persistir `env_state` y se generan snapshots ligeros de politica para evaluacion.
- La LUT se reescribe como JSON completo de forma periodica, con alto coste de I/O.

### Operacion

- `resume_mode=auto` favorece conveniencia; aunque ahora valida mas configuracion del trainer, sigue siendo recomendable abrir una run nueva si cambias el regimen de entrenamiento de forma deliberada.
- `mlflow-ui` y el contenedor de entrenamiento deben mantenerse con la misma version de MLflow para evitar inconsistencias visuales o de backend.

## Mejoras futuras recomendadas (roadmap)

### Prioridad alta: operacion y confianza experimental

- Endurecer aun mas la observabilidad en MLflow (fallos de tracking visibles, metadatos de resume mas explicitos).
- Añadir smoke tests del flujo real de `main.py` en CI (inicio run -> entrenamiento corto -> checkpoint -> cierre run).
- Expandir validaciones de compatibilidad de `resume` (incluyendo regimen de entrenamiento y configuracion completa de redes/optimizadores).
- Homogeneizar versiones de runtime (entrenamiento y `mlflow-ui`) para evitar divergencias de visualizacion/tracking.

### Prioridad media: calidad de senal y estabilidad numerica

- Mejorar evaluacion (benchmarks mas estables y mejor separacion de metricas por baseline).
- Evolucionar `PolicyNet` a esquema de logits + `log_softmax` si encaja con el pipeline actual.
- Reforzar pruebas de no-regresion en metricas clave y escenarios multiway.
- Afinar abstraccion de acciones/sizings por `street`, `SPR` y contexto de apuesta.

### Prioridad media-baja: teoria y escalabilidad

- Incorporar weighting mas fiel por reach probability y acercar el trainer a variantes mas canonicas de Deep CFR.
- Evaluar politicas separadas por rol/posicion efectiva para reducir mezcla estrategica excesiva.
- Reducir coste de `copy.deepcopy` con clonado estructural del entorno.
- Migrar persistencia de LUT a formato incremental/binario para bajar I/O y tiempo de guardado.

## Desarrollo y calidad

La suite actual cubre:

- reglas del entorno
- side pots y zero-sum multiway
- features matematicas
- equity
- checkpointing
- command persistence
- partes del trainer
- compatibilidad del resume por feature schema y configuracion del entorno
- invalidacion correcta de la cache de features

Para llevar el proyecto mas cerca de produccion todavia hacen falta:

- benchmarks reproducibles
- smoke tests del entrypoint en CI
- pruebas de compatibilidad entre versiones de checkpoint
- observabilidad mas estricta en MLflow

## Resumen

Este repositorio es una buena base experimental para estudiar self-play aproximado en poker multiway. Tras las correcciones actuales, el flujo principal, la suite y los checkpoints del entorno quedaron mas robustos, pero el proyecto sigue necesitando mejoras teoricas, de rendimiento y de operacion antes de considerarse una plataforma fiable de investigacion avanzada o una base de produccion.
