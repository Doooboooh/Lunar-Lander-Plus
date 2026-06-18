# Rule Expert Evaluation

- timestamp: `2026-06-18T09:53:51.751686+00:00`
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
| two_waypoint | 2.00/2 | 376.65 | 1.00 | 0.40 | 0.80 | 0.80 | 0.312 | 691.0 |
| orbit | 4.00/4 | 512.73 | 1.00 | 0.60 | 1.00 | 1.00 | 0.090 | 784.4 |
| figure_eight | 6.20/8 | 119.42 | 0.00 | 0.00 | 0.00 | 0.00 | 0.384 | 1000.0 |
| drawn_diamond | 4.00/4 | 533.58 | 1.00 | 0.60 | 1.00 | 1.00 | 0.084 | 722.0 |
