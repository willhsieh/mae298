# Battleship Experiments

This folder now separates reusable search logic from notebooks so you can compare models quickly.

## Layout
- `battleship/`: core Python modules (board generation, priors, algorithms, benchmarking).
- `scripts/benchmark.py`: CLI-style benchmark runner.
- `scripts/run_benchmark.sh`: convenience wrapper that uses the local `.env` venv.
- `battleship-final.ipynb`: notebook wired to the new modules.

## Run (using the local venv)
```bash
./battleship-test/scripts/run_benchmark.sh
```

If the venv is elsewhere:
```bash
BATTLESHIP_VENV=/path/to/venv ./battleship-test/scripts/run_benchmark.sh
```

## Environment knobs
- `BATTLESHIP_ALGORITHMS` (comma list): `adjacent,orientation,placement,hunt_target_focus,hunt_target_strict,hunt_target_eig` (plus `monte_carlo` if needed)
- `BATTLESHIP_BOARDS`: number of boards in the benchmark
- `BATTLESHIP_PRIOR`: `uniform`, `center`, `edge`
- `BATTLESHIP_SEED`: RNG seed for reproducibility
- `BATTLESHIP_MC_SAMPLES`, `BATTLESHIP_MC_MAX_ATTEMPTS`: Monte Carlo posterior control
- `BATTLESHIP_SHIP_SIZES`: e.g., `5x1,4x1,3x1,3x1,2x1`
