| rank | config | eval mean | eval std | route completion | waypoints | final dist | lr | hidden | layers | act | entropy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | tanh_256_low_lr | -153.11 | 32.78 | - | - | - | 1e-04 | 256 | 2 | tanh | 0.005 |
| 2 | relu_256_low_lr | -340.66 | 150.05 | - | - | - | 1e-04 | 256 | 2 | relu | 0.003 |
| 3 | relu_256_longer_rollout | -365.42 | 248.54 | - | - | - | 1e-04 | 256 | 2 | relu | 0.003 |
| 4 | relu_128_low_entropy | -441.16 | 35.24 | - | - | - | 3e-04 | 128 | 2 | relu | 0.003 |
| 5 | conservative_clip | -653.16 | 115.57 | - | - | - | 1e-04 | 128 | 2 | tanh | 0.003 |
