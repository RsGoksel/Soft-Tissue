# Multi-seed FocusPointNet comparison

- epochs per run: 120
- seeds: [0, 1, 2]
- batch size: 4
- base_channels: 16
- total wall time: 52.9 min

| arch | n_runs | test RMS mean ± std | X mean ± std | Y mean ± std | Z mean ± std |
|------|--------|---------------------|--------------|--------------|--------------|
| cnn | 3 | 26.25 ± 4.61 | 4.63 ± 2.21 | 2.69 ± 1.21 | 25.65 ± 4.17 |
| resnet | 3 | 34.48 ± 3.74 | 5.90 ± 3.76 | 3.10 ± 1.36 | 33.64 ± 3.21 |
| unet | 3 | 34.17 ± 8.01 | 3.93 ± 0.70 | 3.02 ± 1.05 | 33.71 ± 8.32 |