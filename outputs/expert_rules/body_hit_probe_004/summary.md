# Rule Expert Evaluation

- timestamp: `2026-06-18T09:54:41.742880+00:00`
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
| two_waypoint | 2.00/2 | 256.11 | 1.00 | 0.00 | 0.20 | 0.20 | 0.771 | 444.8 |
| orbit | 4.00/4 | 559.82 | 1.00 | 0.80 | 0.80 | 0.80 | 0.053 | 739.2 |
| figure_eight | 7.00/8 | 197.70 | 0.20 | 0.00 | 0.00 | 0.00 | 0.577 | 1000.0 |
| drawn_diamond | 4.00/4 | 547.95 | 1.00 | 0.40 | 0.80 | 0.80 | 0.091 | 679.6 |
