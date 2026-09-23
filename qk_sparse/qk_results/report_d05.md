# Q/K vertical-vector sparsity across modalities

### Verdict

- **audio** (audio): _yes_ -- a shared key set per 128-query block retains 0.80 of per-row oracle mass; adjacent-query key overlap 0.744 vs chance 0.026 (29.0x); practical gain over block-sparsity +0.276, over horizontal +0.252.
- **image** (image): _partial_ -- a shared key set per 64-query block retains 0.68 of per-row oracle mass; adjacent-query key overlap 0.572 vs chance 0.025 (22.5x); practical gain over block-sparsity +0.173, over horizontal +0.123.
- **synthetic-high** (synthetic): _yes_ -- a shared key set per 128-query block retains 1.00 of per-row oracle mass; adjacent-query key overlap 0.880 vs chance 0.026 (34.5x); practical gain over block-sparsity +0.207, over horizontal +0.207. **Caveat:** rows spread their mass over 63% of available keys, so no sparse pattern helps much here regardless of shape.
- **synthetic-low** (synthetic): _no_ -- a shared key set per 128-query block retains 0.27 of per-row oracle mass; adjacent-query key overlap 0.025 vs chance 0.026 (1.0x); practical gain over block-sparsity +0.036, over horizontal +0.024. **Caveat:** rows spread their mass over 61% of available keys, so no sparse pattern helps much here regardless of shape.
- **text** (text): _yes_ -- a shared key set per 128-query block retains 0.91 of per-row oracle mass; adjacent-query key overlap 0.494 vs chance 0.026 (19.2x); practical gain over block-sparsity +0.225, over horizontal +0.144.
- **timeseries** (timeseries): density 0.05 selects only 2 key(s) out of 43 -- too coarse to measure. Re-report this model at a higher density (`--density 0.2`).
- **video-dit-long** (video-dit): _yes_ -- a shared key set per 128-query block retains 0.86 of per-row oracle mass; adjacent-query key overlap 0.612 vs chance 0.026 (23.9x); practical gain over block-sparsity +0.108, over horizontal +0.006.
- **video** (video): _partial_ -- a shared key set per 128-query block retains 0.51 of per-row oracle mass; adjacent-query key overlap 0.368 vs chance 0.026 (14.4x); practical gain over block-sparsity +0.222, over horizontal +0.116.
- **vlm** (video-llm): _yes_ -- a shared key set per 128-query block retains 0.88 of per-row oracle mass; adjacent-query key overlap 0.175 vs chance 0.026 (6.8x); practical gain over block-sparsity +0.184, over horizontal +0.159. **Mechanism:** a single global key set already recovers 87% of what per-block sets do, so this is key salience (sinks), not neighbouring-query agreement.


### Recovered attention mass at density 0.05

| model | modality | seq | eff keys | line | block | horizontal | **vertical** | oracle | vert-block | vert/oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| audio | audio | 1500 | 0.26 | 0.268 | 0.152 | 0.176 | **0.429** | 0.533 | +0.276 | 0.804 |
| image | image | 1370 | 0.14 | 0.284 | 0.258 | 0.309 | **0.432** | 0.632 | +0.173 | 0.683 |
| synthetic-high | synthetic | 2048 | 0.63 | 0.091 | 0.043 | 0.043 | **0.250** | 0.250 | +0.207 | 0.999 |
| synthetic-low | synthetic | 2048 | 0.61 | 0.061 | 0.032 | 0.044 | **0.069** | 0.258 | +0.036 | 0.265 |
| text | text | 4096 | 0.01 | 0.568 | 0.623 | 0.704 | **0.848** | 0.937 | +0.225 | 0.905 |
| timeseries (synthetic input) | timeseries | 43 | 0.95 | 0.054 | - | - | **0.060** | 0.075 | - | 0.806 |
| video-dit-long | video-dit | 32760 | 0.01 | 0.247 | 0.716 | 0.818 | **0.824** | 0.962 | +0.108 | 0.856 |
| video | video | 1568 | 0.12 | 0.214 | 0.143 | 0.249 | **0.364** | 0.721 | +0.222 | 0.505 |
| vlm | video-llm | 1612 | 0.04 | 0.615 | 0.522 | 0.548 | **0.706** | 0.800 | +0.184 | 0.883 |


### Selector quality at density 0.05 (mass recovered; `vertical` is the oracle ceiling)

| model | vertical | pooled | centre row | strided max | strided mean | random row | pooled/ceiling | cos_to_mean |
|---|---|---|---|---|---|---|---|---|
| audio | 0.429 | 0.398 | 0.296 | 0.343 | 0.352 | 0.273 | 0.928 | 0.851 |
| image | 0.432 | 0.398 | 0.288 | 0.352 | 0.367 | 0.316 | 0.922 | 0.782 |
| synthetic-high | 0.250 | 0.250 | 0.249 | 0.249 | 0.250 | 0.250 | 1.000 | 0.999 |
| synthetic-low | 0.069 | 0.061 | 0.053 | 0.055 | 0.053 | 0.053 | 0.887 | 0.087 |
| text | 0.848 | 0.768 | 0.716 | 0.744 | 0.754 | 0.715 | 0.906 | 0.917 |
| timeseries | 0.060 | 0.058 | 0.056 | 0.057 | 0.057 | 0.055 | 0.969 | 0.499 |
| video-dit-long | 0.824 | 0.713 | 0.488 | 0.549 | 0.608 | 0.423 | 0.865 | 0.743 |
| video | 0.364 | 0.275 | 0.201 | 0.241 | 0.256 | 0.182 | 0.755 | 0.623 |
| vlm | 0.706 | 0.669 | 0.633 | 0.650 | 0.661 | 0.635 | 0.947 | 0.881 |


### Contiguous vs strided query grouping (retention of per-row oracle, density 0.05)

Both groupings use the *same number of query rows*, so the difference is grouping shape rather than group size.

| model | rows | stride | contiguous | strided | difference | n |
|---|---|---|---|---|---|---|
| image | 35 | 37 | 0.816 | 0.778 | -0.038 | 384 |
| video-dit-long | 128 | 52 | 0.917 | 0.868 | -0.048 | 288 |
| video-dit-long | 19 | 1560 | 0.963 | 0.962 | -0.001 | 288 |
| video | 98 | 14 | 0.723 | 0.609 | -0.114 | 192 |
| vlm | 83 | 11 | 0.894 | 0.948 | +0.055 | 96 |
| vlm | 124 | 21 | 0.901 | 0.942 | +0.041 | 96 |


### Top-key agreement vs query distance (Jaccard, median)

| model | chance | d=1 | d=4 | d=16 | d=64 | d=256 |
|---|---|---|---|---|---|---|
| audio | 0.026 | 0.744 | 0.456 | 0.471 | 0.500 | 0.376 |
| image | 0.025 | 0.572 | 0.214 | 0.054 | - | - |
| synthetic-high | 0.026 | 0.880 | 0.889 | 0.889 | 0.025 | 0.025 |
| synthetic-low | 0.026 | 0.025 | 0.025 | 0.028 | 0.028 | 0.025 |
| text | 0.026 | 0.494 | 0.301 | 0.233 | 0.151 | 0.104 |
| timeseries | 0.024 | 0.000 | 0.000 | 0.000 | - | - |
| video-dit-long | 0.026 | 0.612 | 0.354 | 0.178 | 0.193 | 0.079 |
| video | 0.026 | 0.368 | 0.219 | - | - | - |
| vlm | 0.026 | 0.175 | 0.130 | 0.122 | 0.135 | 0.130 |


### Query-vector cosine vs distance (median)

| model | d=1 | d=4 | d=16 | d=64 | d=256 |
|---|---|---|---|---|---|
| audio | 0.955 | 0.736 | 0.698 | 0.812 | 0.583 |
| image | 0.891 | 0.703 | 0.544 | - | - |
| synthetic-high | 0.997 | 0.997 | 0.998 | 0.015 | 0.004 |
| synthetic-low | -0.013 | 0.016 | 0.013 | -0.011 | -0.016 |
| text | 0.960 | 0.893 | 0.877 | 0.825 | 0.765 |
| timeseries | -0.009 | 0.742 | 0.604 | - | - |
| video-dit-long | 0.877 | 0.651 | 0.506 | 0.561 | 0.377 |
| video | 0.713 | 0.502 | - | - | - |
| vlm | 0.825 | 0.790 | 0.765 | 0.764 | 0.765 |


### Relative output error at density 0.05 (lower is better)

`||O_sparse - O_dense|| / ||O_dense||`, renormalised over kept keys.

| model | line | block | horizontal | **vertical** | oracle | pooled |
|---|---|---|---|---|---|---|
| audio | - | 1.331 | 1.296 | 0.931 | 0.767 | 0.993 |
| image | - | 0.824 | 0.764 | 0.558 | 0.336 | 0.623 |
| synthetic-high | - | 6.470 | 6.470 | 2.487 | 2.477 | 2.487 |
| synthetic-low | - | 5.459 | 6.629 | 4.923 | 2.428 | 4.244 |
| text | - | 0.514 | 0.396 | 0.227 | 0.089 | 0.269 |
| timeseries | - | - | - | 4.962 | 4.902 | 4.985 |
| video-dit-long | - | 0.162 | 0.109 | 0.115 | 0.027 | 0.191 |
| video | - | 0.989 | 0.823 | 0.618 | 0.254 | 0.751 |
| vlm | - | 0.828 | 0.763 | 0.504 | 0.252 | 0.558 |


### Retention (vertical/oracle) vs budget

| model | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| audio | 0.62 | 0.71 | 0.80 | 0.89 | 0.94 |
| image | 0.59 | 0.62 | 0.68 | 0.78 | 0.87 |
| synthetic-high | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| synthetic-low | 0.17 | 0.20 | 0.27 | 0.33 | 0.44 |
| text | 0.77 | 0.84 | 0.91 | 0.94 | 0.97 |
| timeseries | 0.81 | 0.81 | 0.81 | 0.81 | 0.83 |
| video-dit-long | 0.54 | 0.68 | 0.86 | 0.93 | 0.98 |
| video | 0.32 | 0.38 | 0.51 | 0.62 | 0.76 |
| vlm | 0.88 | 0.87 | 0.88 | 0.88 | 0.91 |


### MinP selected density (VecAttention's own rule)

| model | gap=4 | gap=6 | gap=8 |
|---|---|---|---|
| audio | 0.398 | 0.997 | 1.000 |
| image | 0.014 | 0.263 | 0.790 |
| synthetic-high | 0.765 | 0.997 | 1.000 |
| synthetic-low | 1.000 | 1.000 | 1.000 |
| text | 0.001 | 0.008 | 0.088 |
| timeseries | 1.000 | 1.000 | 1.000 |
| video-dit-long | 0.050 | 0.171 | 0.402 |
| video | 0.458 | 0.846 | 0.992 |
| vlm | 0.001 | 0.007 | 0.188 |


### Per-layer breakdown (density 0.05)

| model | depth | eff keys | line | block | **vertical** | oracle | vert/oracle | line/vert | n |
|---|---|---|---|---|---|---|---|---|---|
| audio | 0 | 0.69 | 0.192 | 0.121 | **0.235** | 0.238 | 0.987 | 0.820 | 176 |
| audio | 1 | 0.26 | 0.333 | 0.218 | **0.513** | 0.552 | 0.930 | 0.649 | 176 |
| audio | 2 | 0.64 | 0.203 | 0.122 | **0.228** | 0.251 | 0.907 | 0.889 | 176 |
| audio | 3 | 0.24 | 0.277 | 0.143 | **0.438** | 0.521 | 0.840 | 0.632 | 176 |
| audio | 4 | 0.25 | 0.448 | 0.112 | **0.522** | 0.573 | 0.912 | 0.857 | 176 |
| audio | 5 | 0.15 | 0.491 | 0.238 | **0.580** | 0.700 | 0.830 | 0.845 | 176 |
| image | 0 | 0.24 | 0.380 | 0.161 | **0.422** | 0.485 | 0.871 | 0.899 | 256 |
| image | 1 | 0.07 | 0.325 | 0.500 | **0.592** | 0.638 | 0.928 | 0.549 | 256 |
| image | 2 | 0.20 | 0.200 | 0.241 | **0.323** | 0.534 | 0.605 | 0.619 | 256 |
| image | 3 | 0.08 | 0.226 | 0.270 | **0.486** | 0.795 | 0.611 | 0.466 | 256 |
| image | 4 | 0.16 | 0.270 | 0.186 | **0.388** | 0.634 | 0.611 | 0.696 | 256 |
| image | 5 | 0.18 | 0.378 | 0.269 | **0.454** | 0.594 | 0.764 | 0.834 | 256 |
| synthetic-high | 0 | 0.63 | 0.091 | 0.043 | **0.250** | 0.250 | 0.999 | 0.363 | 128 |
| synthetic-low | 0 | 0.61 | 0.061 | 0.032 | **0.069** | 0.258 | 0.265 | 0.890 | 128 |
| text | 0 | 0.04 | 0.034 | 0.290 | **0.657** | 0.928 | 0.708 | 0.052 | 256 |
| text | 1 | 0.01 | 0.762 | 0.765 | **0.847** | 0.937 | 0.904 | 0.900 | 256 |
| text | 2 | 0.00 | 0.852 | 0.862 | **0.956** | 0.984 | 0.972 | 0.891 | 256 |
| text | 3 | 0.03 | 0.479 | 0.536 | **0.834** | 0.887 | 0.940 | 0.574 | 256 |
| text | 4 | 0.00 | 0.924 | 0.935 | **0.975** | 0.987 | 0.987 | 0.948 | 256 |
| text | 5 | 0.14 | 0.166 | 0.169 | **0.466** | 0.656 | 0.710 | 0.357 | 256 |
| timeseries | 0 | 0.95 | 0.050 | - | **0.056** | 0.070 | 0.797 | 0.893 | 80 |
| timeseries | 1 | 0.97 | 0.056 | - | **0.061** | 0.068 | 0.894 | 0.916 | 80 |
| timeseries | 2 | 0.79 | 0.066 | - | **0.107** | 0.111 | 0.959 | 0.616 | 80 |
| video-dit-long | 0 | 0.73 | 0.085 | 0.148 | **0.177** | 0.190 | 0.932 | 0.481 | 192 |
| video-dit-long | 1 | 0.05 | 0.259 | 0.526 | **0.683** | 0.870 | 0.785 | 0.379 | 192 |
| video-dit-long | 2 | 0.01 | 0.318 | 0.607 | **0.804** | 0.954 | 0.843 | 0.396 | 192 |
| video-dit-long | 3 | 0.00 | 0.453 | 0.951 | **0.964** | 0.997 | 0.967 | 0.470 | 192 |
| video-dit-long | 4 | 0.00 | 0.256 | 0.809 | **0.874** | 0.981 | 0.891 | 0.293 | 192 |
| video-dit-long | 5 | 0.08 | 0.103 | 0.443 | **0.480** | 0.799 | 0.600 | 0.215 | 192 |
| video | 0 | 0.61 | 0.147 | 0.077 | **0.161** | 0.219 | 0.736 | 0.913 | 96 |
| video | 1 | 0.17 | 0.158 | 0.168 | **0.310** | 0.615 | 0.504 | 0.510 | 96 |
| video | 2 | 0.09 | 0.200 | 0.145 | **0.299** | 0.766 | 0.391 | 0.667 | 96 |
| video | 3 | 0.05 | 0.231 | 0.149 | **0.400** | 0.862 | 0.464 | 0.579 | 96 |
| video | 4 | 0.18 | 0.347 | 0.116 | **0.398** | 0.613 | 0.650 | 0.871 | 96 |
| video | 5 | 0.10 | 0.686 | 0.300 | **0.693** | 0.771 | 0.899 | 0.991 | 96 |
| vlm | 0 | 0.11 | 0.586 | 0.479 | **0.669** | 0.672 | 0.995 | 0.876 | 96 |
| vlm | 1 | 0.01 | 0.718 | 0.662 | **0.846** | 0.894 | 0.947 | 0.848 | 96 |
| vlm | 2 | 0.07 | 0.496 | 0.401 | **0.581** | 0.776 | 0.749 | 0.854 | 96 |
| vlm | 3 | 0.04 | 0.385 | 0.350 | **0.642** | 0.780 | 0.823 | 0.599 | 96 |
| vlm | 4 | 0.01 | 0.753 | 0.756 | **0.853** | 0.952 | 0.896 | 0.882 | 96 |
| vlm | 5 | 0.01 | 0.701 | 0.633 | **0.792** | 0.889 | 0.891 | 0.885 | 96 |


### Warnings

- timeseries: time-series windows were synthetic (seasonal + AR); pass --data-dir with an ETTh1-style CSV for real data
- video-dit-long: latent token grid 21x30x52 = 32760 tokens; strides row=52, frame=1560
- video: MCG-NJU/videomae-base-finetuned-kinetics: repaired 24 q/v biases from the checkpoint and zeroed 12 key biases (BEiT-style layout); Q/K now match the pretrained model.
- vlm: sample 0: 1612 prefill tokens, video_grid_thw=[16, 18, 22]
- vlm: sample 1: 4060 prefill tokens, video_grid_thw=[16, 24, 42]