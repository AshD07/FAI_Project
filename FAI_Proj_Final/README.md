# Multi-Tier Explainable AI Pilot for Asteroid-Field Survival

CS 502 (Fundamentals of Artificial Intelligence) project, South Asian University.

**Team:** Ashutosh Diwan, Tanishq Ajmeria, Vithul, Himanshu

An autonomous spacecraft pilot for a Kessler/XFC-style asteroid game. Each layer uses a classical AI technique:

| Stage | Technique | Code |
|---|---|---|
| 1–2 | Physics engine and noisy radar model | `src/environment/`, `src/sensors/` |
| 3 | Kalman filter multi-object tracking | `src/tracking/` |
| 4 | Forward-chaining rule engine (time-to-impact evasion) | `src/reasoning/` |
| 5 | Local A* path planning | `src/navigation/` |
| 6 | CSP firing gate with ballistic lead solver | `src/firing/` |
| 7 | Minimax with alpha-beta pruning | `src/strategy/` |
| 8 | Genetic algorithm parameter tuning | `src/optimization/` |
| 9 | Explainability log and HUD | `src/explainability/` |

The full write-up is in [`CS502_Project_Report.pdf`](CS502_Project_Report.pdf).

## Results (summary)

| Agent | Accuracy | Mean survival |
|---|---|---|
| Autonomous pilot (default) | 32.6% | 20.2 s |
| Autonomous pilot (GA-tuned) | 40.1% | 18.2 s |
| Naive reactive baseline | 16.1% | 8.6 s |
| Random baseline | 3.3% | 17.9 s |

100 benchmark matches (5 scenarios × 4 agents × 5 seeds). See `EVALUATION_REPORT.md`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_survival.py                     # interactive solo survival (SPACE pause, G grid, E export logs, R reset)
python run_survival.py --headless --seed 1 # headless run with decision log
python run_evaluation.py --runs 5          # benchmark matrix, report and charts
python train_ga.py --generations 10 --pop 12 --ticks 600   # retrain the GA
python -m unittest discover tests          # 32 unit tests
```

Stage demos: `demo_stage1_3.py`, `demo_stage4_6.py`, `demo_stage7_9.py`.
