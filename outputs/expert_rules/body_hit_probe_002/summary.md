# Rule Expert Evaluation

- timestamp: `2026-06-18T09:53:26.019568+00:00`
- seed: `30000`
- episodes: `5`
- radius: `0.16`
- waypoint hit mode: `body`

## Base Landing

| mean return | std | min | max |
|---:|---:|---:|---:|
| 267.14 | 41.94 | 185.67 | 304.67 |

## Waypoint Routes

| task | waypoints | mean return | route completion | both-leg land | touchdown | settled | final target dist | episode length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| two_waypoint | 2.00/2 | 446.78 | 1.00 | 0.60 | 1.00 | 1.00 | 0.134 | 713.4 |
| orbit | 4.00/4 | 493.46 | 1.00 | 0.80 | 1.00 | 1.00 | 0.081 | 875.4 |
| figure_eight | 5.20/8 | 74.57 | 0.00 | 0.00 | 0.00 | 0.00 | 0.525 | 1000.0 |
| drawn_diamond | 3.60/4 | 290.10 | 0.60 | 0.60 | 0.60 | 0.60 | 0.072 | 953.2 |
