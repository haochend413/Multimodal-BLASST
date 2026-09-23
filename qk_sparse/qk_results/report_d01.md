# Q/K vertical-vector sparsity across modalities

### Verdict

- **audio** (audio): _partial_ -- a shared key set per 128-query block retains 0.62 of per-row oracle mass; adjacent-query key overlap 0.667 vs chance 0.005 (132.7x); practical gain over block-sparsity +nan, over horizontal +nan.
- **image** (image): _partial_ -- a shared key set per 64-query block retains 0.59 of per-row oracle mass; adjacent-query key overlap 0.474 vs chance 0.005 (92.2x); practical gain over block-sparsity +nan, over horizontal +nan.
- **synthetic-high** (synthetic): _yes_ -- a shared key set per 128-query block retains 1.00 of per-row oracle mass; adjacent-query key overlap 0.818 vs chance 0.005 (166.7x); practical gain over block-sparsity +nan, over horizontal +nan. **Caveat:** rows spread their mass over 63% of available keys, so no sparse pattern helps much here regardless of shape.
- **synthetic-low** (synthetic): _no_ -- a shared key set per 128-query block retains 0.17 of per-row oracle mass; adjacent-query key overlap 0.000 vs chance 0.005 (0.0x); practical gain over block-sparsity +nan, over horizontal +nan. **Caveat:** rows spread their mass over 61% of available keys, so no sparse pattern helps much here regardless of shape.
- **text** (text): _partial_ -- a shared key set per 128-query block retains 0.77 of per-row oracle mass; adjacent-query key overlap 0.466 vs chance 0.005 (92.3x); practical gain over block-sparsity +nan, over horizontal +nan.
- **timeseries** (timeseries): density 0.01 selects only 0 key(s) out of 43 -- too coarse to measure. Re-report this model at a higher density (`--density 0.2`).
- **video-dit-long** (video-dit): _partial_ -- a shared key set per 128-query block retains 0.54 of per-row oracle mass; adjacent-query key overlap 0.515 vs chance 0.005 (102.4x); practical gain over block-sparsity +0.148, over horizontal -0.013.
- **video** (video): _no_ -- a shared key set per 128-query block retains 0.32 of per-row oracle mass; adjacent-query key overlap 0.280 vs chance 0.005 (54.6x); practical gain over block-sparsity +nan, over horizontal +nan.
- **vlm** (video-llm): _yes_ -- a shared key set per 128-query block retains 0.88 of per-row oracle mass; adjacent-query key overlap 0.111 vs chance 0.005 (22.2x); practical gain over block-sparsity +nan, over horizontal +nan. **Mechanism:** a single global key set already recovers 90% of what per-block sets do, so this is key salience (sinks), not neighbouring-query agreement.


### Recovered attention mass at density 0.01

| model | modality | seq | eff keys | line | block | horizontal | **vertical** | oracle | vert-block | vert/oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| audio | audio | 1500 | 0.26 | 0.077 | - | - | **0.121** | 0.195 | - | 0.620 |
| image | image | 1370 | 0.14 | 0.171 | - | - | **0.229** | 0.385 | - | 0.594 |
| synthetic-high | synthetic | 2048 | 0.63 | 0.019 | - | - | **0.085** | 0.085 | - | 0.997 |
| synthetic-low | synthetic | 2048 | 0.61 | 0.013 | - | - | **0.015** | 0.091 | - | 0.167 |
| text | text | 4096 | 0.01 | 0.519 | - | - | **0.638** | 0.832 | - | 0.767 |
| timeseries (synthetic input) | timeseries | 43 | 0.95 | 0.027 | - | - | **0.031** | 0.038 | - | 0.809 |
| video-dit-long | video-dit | 32760 | 0.01 | 0.053 | 0.312 | 0.473 | **0.460** | 0.859 | +0.148 | 0.536 |
| video | video | 1568 | 0.12 | 0.069 | - | - | **0.132** | 0.414 | - | 0.318 |
| vlm | video-llm | 1612 | 0.04 | 0.537 | - | - | **0.598** | 0.677 | - | 0.884 |


### Selector quality at density 0.01 (mass recovered; `vertical` is the oracle ceiling)

| model | vertical | pooled | centre row | strided max | strided mean | random row | pooled/ceiling | cos_to_mean |
|---|---|---|---|---|---|---|---|---|
| audio | 0.121 | 0.115 | 0.088 | 0.094 | 0.102 | 0.082 | 0.952 | 0.851 |
| image | 0.229 | 0.206 | 0.161 | 0.180 | 0.194 | 0.174 | 0.899 | 0.782 |
| synthetic-high | 0.085 | 0.085 | 0.085 | 0.085 | 0.085 | 0.085 | 1.000 | 0.999 |
| synthetic-low | 0.015 | 0.013 | 0.011 | 0.011 | 0.011 | 0.011 | 0.845 | 0.087 |
| text | 0.638 | 0.623 | 0.577 | 0.602 | 0.616 | 0.583 | 0.976 | 0.917 |
| timeseries | 0.031 | 0.030 | 0.029 | 0.030 | 0.029 | 0.028 | 0.977 | 0.499 |
| video-dit-long | 0.460 | 0.329 | 0.201 | 0.220 | 0.272 | 0.166 | 0.715 | 0.743 |
| video | 0.132 | 0.091 | 0.065 | 0.073 | 0.080 | 0.059 | 0.690 | 0.623 |
| vlm | 0.598 | 0.574 | 0.564 | 0.569 | 0.571 | 0.563 | 0.960 | 0.881 |


### Contiguous vs strided query grouping (retention of per-row oracle, density 0.01)

Both groupings use the *same number of query rows*, so the difference is grouping shape rather than group size.

| model | rows | stride | contiguous | strided | difference | n |
|---|---|---|---|---|---|---|
| image | 35 | 37 | 0.759 | 0.755 | -0.005 | 384 |
| video-dit-long | 128 | 52 | 0.680 | 0.610 | -0.070 | 288 |
| video-dit-long | 19 | 1560 | 0.894 | 0.861 | -0.032 | 288 |
| video | 98 | 14 | 0.569 | 0.470 | -0.099 | 192 |
| vlm | 83 | 11 | 0.923 | 0.955 | +0.032 | 96 |
| vlm | 124 | 21 | 0.907 | 0.951 | +0.045 | 96 |


### Top-key agreement vs query distance (Jaccard, median)

| model | chance | d=1 | d=4 | d=16 | d=64 | d=256 |
|---|---|---|---|---|---|---|
| audio | 0.005 | 0.667 | 0.250 | 0.200 | 0.304 | 0.154 |
| image | 0.005 | 0.474 | 0.167 | 0.037 | - | - |
| synthetic-high | 0.005 | 0.818 | 0.818 | 0.818 | 0.000 | 0.000 |
| synthetic-low | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| text | 0.005 | 0.466 | 0.231 | 0.152 | 0.083 | 0.056 |
| timeseries | 0.012 | 0.000 | 0.000 | 0.000 | - | - |
| video-dit-long | 0.005 | 0.515 | 0.195 | 0.062 | 0.074 | 0.014 |
| video | 0.005 | 0.280 | 0.067 | - | - | - |
| vlm | 0.005 | 0.111 | 0.091 | 0.083 | 0.082 | 0.083 |


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


### Relative output error at density 0.01 (lower is better)

`||O_sparse - O_dense|| / ||O_dense||`, renormalised over kept keys.

| model | line | block | horizontal | **vertical** | oracle | pooled |
|---|---|---|---|---|---|---|
| audio | - | - | - | 1.606 | 1.487 | 1.609 |
| image | - | - | - | 0.971 | 0.639 | 1.055 |
| synthetic-high | - | - | - | 6.251 | 6.194 | 6.251 |
| synthetic-low | - | - | - | 10.605 | 6.242 | 9.052 |
| text | - | - | - | 0.494 | 0.243 | 0.514 |
| timeseries | - | - | - | 5.543 | 5.550 | 5.306 |
| video-dit-long | - | 0.403 | 0.334 | 0.375 | 0.095 | 0.473 |
| video | - | - | - | 1.151 | 0.554 | 1.213 |
| vlm | - | - | - | 0.749 | 0.503 | 0.790 |


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


### Per-layer breakdown (density 0.01)

| model | depth | eff keys | line | block | **vertical** | oracle | vert/oracle | line/vert | n |
|---|---|---|---|---|---|---|---|---|---|
| audio | 0 | 0.69 | 0.056 | - | **0.076** | 0.078 | 0.981 | 0.728 | 176 |
| audio | 1 | 0.26 | 0.101 | - | **0.142** | 0.201 | 0.708 | 0.709 | 176 |
| audio | 2 | 0.64 | 0.055 | - | **0.072** | 0.082 | 0.885 | 0.756 | 176 |
| audio | 3 | 0.24 | 0.084 | - | **0.133** | 0.180 | 0.736 | 0.629 | 176 |
| audio | 4 | 0.25 | 0.126 | - | **0.144** | 0.206 | 0.698 | 0.880 | 176 |
| audio | 5 | 0.15 | 0.180 | - | **0.224** | 0.320 | 0.699 | 0.803 | 176 |
| image | 0 | 0.24 | 0.171 | - | **0.181** | 0.218 | 0.830 | 0.946 | 256 |
| image | 1 | 0.07 | 0.250 | - | **0.270** | 0.516 | 0.524 | 0.925 | 256 |
| image | 2 | 0.20 | 0.096 | - | **0.184** | 0.325 | 0.567 | 0.520 | 256 |
| image | 3 | 0.08 | 0.127 | - | **0.217** | 0.483 | 0.449 | 0.583 | 256 |
| image | 4 | 0.16 | 0.163 | - | **0.205** | 0.339 | 0.604 | 0.797 | 256 |
| image | 5 | 0.18 | 0.242 | - | **0.265** | 0.345 | 0.768 | 0.913 | 256 |
| synthetic-high | 0 | 0.63 | 0.019 | - | **0.085** | 0.085 | 0.997 | 0.227 | 128 |
| synthetic-low | 0 | 0.61 | 0.013 | - | **0.015** | 0.091 | 0.167 | 0.843 | 128 |
| text | 0 | 0.04 | 0.009 | - | **0.271** | 0.780 | 0.347 | 0.031 | 256 |
| text | 1 | 0.01 | 0.731 | - | **0.767** | 0.860 | 0.892 | 0.953 | 256 |
| text | 2 | 0.00 | 0.843 | - | **0.918** | 0.961 | 0.956 | 0.918 | 256 |
| text | 3 | 0.03 | 0.404 | - | **0.611** | 0.695 | 0.878 | 0.662 | 256 |
| text | 4 | 0.00 | 0.920 | - | **0.955** | 0.968 | 0.986 | 0.963 | 256 |
| text | 5 | 0.14 | 0.079 | - | **0.217** | 0.373 | 0.581 | 0.365 | 256 |
| timeseries | 0 | 0.95 | 0.025 | - | **0.028** | 0.036 | 0.799 | 0.865 | 80 |
| timeseries | 1 | 0.97 | 0.027 | - | **0.031** | 0.034 | 0.913 | 0.859 | 80 |
| timeseries | 2 | 0.79 | 0.033 | - | **0.054** | 0.058 | 0.941 | 0.600 | 80 |
| video-dit-long | 0 | 0.73 | 0.019 | 0.033 | **0.047** | 0.053 | 0.884 | 0.410 | 192 |
| video-dit-long | 1 | 0.05 | 0.065 | 0.265 | **0.388** | 0.679 | 0.572 | 0.167 | 192 |
| video-dit-long | 2 | 0.01 | 0.154 | 0.236 | **0.475** | 0.847 | 0.562 | 0.323 | 192 |
| video-dit-long | 3 | 0.00 | 0.106 | 0.768 | **0.806** | 0.967 | 0.833 | 0.131 | 192 |
| video-dit-long | 4 | 0.00 | 0.069 | 0.381 | **0.472** | 0.909 | 0.520 | 0.146 | 192 |
| video-dit-long | 5 | 0.08 | 0.016 | 0.127 | **0.148** | 0.420 | 0.352 | 0.107 | 192 |
| video | 0 | 0.61 | 0.036 | - | **0.042** | 0.057 | 0.736 | 0.872 | 96 |
| video | 1 | 0.17 | 0.047 | - | **0.109** | 0.303 | 0.359 | 0.432 | 96 |
| video | 2 | 0.09 | 0.060 | - | **0.102** | 0.462 | 0.221 | 0.593 | 96 |
| video | 3 | 0.05 | 0.076 | - | **0.163** | 0.616 | 0.265 | 0.467 | 96 |
| video | 4 | 0.18 | 0.128 | - | **0.163** | 0.338 | 0.484 | 0.785 | 96 |
| video | 5 | 0.10 | 0.464 | - | **0.474** | 0.534 | 0.887 | 0.979 | 96 |
| vlm | 0 | 0.11 | 0.452 | - | **0.530** | 0.534 | 0.992 | 0.853 | 96 |
| vlm | 1 | 0.01 | 0.681 | - | **0.747** | 0.801 | 0.933 | 0.912 | 96 |
| vlm | 2 | 0.07 | 0.427 | - | **0.456** | 0.538 | 0.848 | 0.934 | 96 |
| vlm | 3 | 0.04 | 0.305 | - | **0.447** | 0.647 | 0.690 | 0.682 | 96 |
| vlm | 4 | 0.01 | 0.709 | - | **0.753** | 0.883 | 0.853 | 0.942 | 96 |
| vlm | 5 | 0.01 | 0.651 | - | **0.688** | 0.773 | 0.890 | 0.946 | 96 |


### Warnings

- timeseries: time-series windows were synthetic (seasonal + AR); pass --data-dir with an ETTh1-style CSV for real data
- video-dit-long: latent token grid 21x30x52 = 32760 tokens; strides row=52, frame=1560
- video: MCG-NJU/videomae-base-finetuned-kinetics: repaired 24 q/v biases from the checkpoint and zeroed 12 key biases (BEiT-style layout); Q/K now match the pretrained model.
- vlm: sample 0: 1612 prefill tokens, video_grid_thw=[16, 18, 22]
- vlm: sample 1: 4060 prefill tokens, video_grid_thw=[16, 24, 42]