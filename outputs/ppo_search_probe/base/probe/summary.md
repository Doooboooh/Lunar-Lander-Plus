| rank | config | eval mean | eval std | route completion | waypoints | final dist | lr | hidden | layers | act | entropy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | relu_128_low_entropy | -137.91 | 20.88 | - | - | - | 3e-04 | 128 | 2 | relu | 0.003 |
| 2 | baseline_tanh_128 | -200.12 | 70.87 | - | - | - | 3e-04 | 128 | 2 | tanh | 0.010 |
| 3 | tanh_256_low_lr | -237.18 | 61.21 | - | - | - | 1e-04 | 256 | 2 | tanh | 0.005 |
| 4 | elu_256 | -248.29 | 74.35 | - | - | - | 3e-04 | 256 | 2 | elu | 0.010 |
