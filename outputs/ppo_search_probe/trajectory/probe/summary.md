| rank | config | eval mean | eval std | route completion | waypoints | final dist | lr | hidden | layers | act | entropy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | relu_128_low_entropy | -180.79 | 2.91 | 0.00 | 0.00/2 | 1.30 | 3e-04 | 128 | 2 | relu | 0.003 |
| 2 | elu_256 | -192.36 | 5.64 | 0.00 | 0.00/2 | 1.29 | 3e-04 | 256 | 2 | elu | 0.010 |
| 3 | tanh_256_low_lr | -263.95 | 11.70 | 0.00 | 0.00/2 | 1.21 | 1e-04 | 256 | 2 | tanh | 0.005 |
| 4 | baseline_tanh_128 | -276.01 | 65.49 | 0.00 | 0.00/2 | 1.29 | 3e-04 | 128 | 2 | tanh | 0.010 |
