| rank | config | eval mean | eval std | route completion | waypoints | final dist | lr | hidden | layers | act | entropy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | relu_128_low_entropy | -191.19 | 6.61 | 0.00 | 0.00/2 | 1.30 | 3e-04 | 128 | 2 | relu | 0.003 |
| 2 | baseline_tanh_128 | -271.24 | 6.75 | 0.00 | 0.00/2 | 1.21 | 3e-04 | 128 | 2 | tanh | 0.010 |
| 3 | tanh_256_low_lr | -271.24 | 6.75 | 0.00 | 0.00/2 | 1.21 | 1e-04 | 256 | 2 | tanh | 0.005 |
| 4 | elu_256 | -271.24 | 6.75 | 0.00 | 0.00/2 | 1.21 | 3e-04 | 256 | 2 | elu | 0.010 |
