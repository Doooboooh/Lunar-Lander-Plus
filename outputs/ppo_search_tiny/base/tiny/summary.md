| rank | config | eval mean | eval std | route completion | waypoints | final dist | lr | hidden | layers | act | entropy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | tanh_256_low_lr | -225.91 | 91.37 | - | - | - | 1e-04 | 256 | 2 | tanh | 0.005 |
| 2 | elu_256 | -269.99 | 86.83 | - | - | - | 3e-04 | 256 | 2 | elu | 0.010 |
| 3 | relu_128_low_entropy | -526.16 | 57.11 | - | - | - | 3e-04 | 128 | 2 | relu | 0.003 |
| 4 | baseline_tanh_128 | -578.56 | 4.71 | - | - | - | 3e-04 | 128 | 2 | tanh | 0.010 |
