# Poker CFR AI

Proyecto de investigacion y entrenamiento para un agente de No-Limit Texas Hold'em inspirado en Deep CFR, con soporte multiway (`2-9` jugadores), evaluacion contra varios baselines y ejecucion principal via Docker.

No es un solver GTO exacto. El repositorio implementa una aproximacion basada en:

- abstraccion discreta de acciones,
- features ingenierizadas en lugar de informacion perfecta cruda,
- estimacion de equity mediante LUT + Monte Carlo,
- dos redes compartidas entre asientos (`RegretNet` y `PolicyNet`),
- self-play con rollouts para estimar utilidades de acciones no muestreadas.

## Estado actual

La base es funcional para experimentacion, pero sigue siendo un sistema aproximado y no una plataforma de produccion.

- La suite `pytest` pasa completa (`43` tests).
- El entrypoint `main.py` arranca y ejecuta el warmup preflop correctamente.
- Las utilidades terminales se preservan en bruto, manteniendo zero-sum en entrenamiento y evaluacion multiway.
- La cache de `InformationSetBuilder` invalida correctamente los campos relevantes del estado.
- La reanudacion de checkpoints valida la configuracion del entorno antes de cargar `env_state`.
- La implementacion sigue siendo una aproximacion estilo Deep CFR y no mantiene garantias teoricas de convergencia a equilibrio de Nash multiway.
- Metricas como `robust_score` y `exploitability_proxy` son proxies operativos, no medidas teoricas exactas.

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
  Gestion de `latest.pt`, `best.pt`, `best_robust.pt` y `run_summary.json`.

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
- `run_summary.json`

Tambien se persiste el comando ejecutado en `artifacts/last_run_command.txt` o en la ruta configurada con `--command-file`.

## Limitaciones conocidas

### Correctitud

- El proyecto sigue usando una aproximacion de juego y de regret, no un solving exacto del arbol.
- La validacion de `resume` protege la configuracion del entorno, pero no bloquea todavia todos los cambios posibles de hiperparametros de entrenamiento.

### Teoria de juegos / poker

- No hay calculo real de exploitability ni best response.
- El algoritmo no implementa Deep CFR canonico con todas sus ponderaciones de reach y counterfactual values.
- La abstraccion de acciones y de features puede inducir estrategias explotables fuera del dominio entrenado.

### Rendimiento

- El recorrido usa muchos `copy.deepcopy`.
- Los caches de equity exacta no estan acotados en memoria.
- Los checkpoints guardan buffers completos y estado del entorno, lo que escala mal en disco y tiempo de carga.
- La LUT se reescribe como JSON completo de forma periodica, con alto coste de I/O.

### Operacion

- El wrapper de MLflow silencia errores y puede hacer que una run parezca sana aunque el tracking haya fallado.
- `resume_mode=auto` favorece conveniencia; hoy valida la compatibilidad del entorno, pero no toda la configuracion del trainer.

## Mejoras futuras recomendadas

### Correctitud y robustez

- Extender la validacion de checkpoints para cubrir tambien hiperparametros del trainer.
- Añadir tests de humo automaticos para el entrypoint real.
- Añadir pruebas de no-regresion sobre metricas y evaluacion multiway.

### ML y optimizacion

- Pasar `PolicyNet` a logits + `log_softmax` para mejorar estabilidad numerica.
- Sustituir `copy.deepcopy` por clonado estructural del entorno.
- Separar checkpoints completos de snapshots ligeros solo-con-pesos.
- Acotar caches y medir su impacto con profiling.
- Mover la persistencia de LUT a un formato incremental mas eficiente.

### Poker y teoria de juegos

- Separar politicas por jugador o, como minimo, por rol y posicion efectiva.
- Introducir weighting correcto por reach probability.
- Incorporar evaluacion aproximada de best response.
- Mejorar la abstraccion de sizings por street y por SPR.
- Añadir features de rangos, blockers y textura del board mas fieles a NLHE real.

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
