| variant                              | target   | split      |   alpha |   az_mae |   az_ci_low |   az_ci_high |   el_mae |   el_ci_low |   el_ci_high |
|:-------------------------------------|:---------|:-----------|--------:|---------:|------------:|-------------:|---------:|------------:|-------------:|
| full image                           | global   | test       |    3.19 |    43.91 |       38.88 |        49.20 |     1.87 |        1.70 |         2.04 |
| full image                           | global   | test_cubes |    3.19 |    42.53 |       40.38 |        44.67 |     1.94 |        1.86 |         2.02 |
| full image                           | local    | test       |    3.19 |    43.91 |       38.88 |        49.20 |     1.87 |        1.70 |         2.04 |
| full image                           | local    | test_cubes |    3.19 |    42.53 |       40.38 |        44.67 |     1.94 |        1.86 |         2.02 |
| floor only (object removed)          | global   | test       |    0.09 |    25.46 |       21.75 |        29.55 |     1.81 |        1.66 |         1.95 |
| floor only (object removed)          | global   | test_cubes |    0.09 |    21.45 |       20.13 |        22.85 |     1.89 |        1.82 |         1.96 |
| floor only (object removed)          | local    | test       |    0.09 |    25.46 |       21.75 |        29.55 |     1.81 |        1.66 |         1.95 |
| floor only (object removed)          | local    | test_cubes |    0.09 |    21.45 |       20.13 |        22.85 |     1.89 |        1.82 |         1.96 |
| object only (crop)                   | global   | test       |    6.88 |     5.65 |        4.95 |         6.42 |     2.49 |        2.18 |         2.82 |
| object only (crop)                   | global   | test_cubes |    6.88 |    33.82 |       31.95 |        35.73 |    10.68 |       10.18 |        11.19 |
| object only (crop)                   | local    | test       |    6.88 |     5.65 |        4.95 |         6.42 |     2.49 |        2.18 |         2.82 |
| object only (crop)                   | local    | test_cubes |    6.88 |    33.82 |       31.95 |        35.73 |    10.68 |       10.18 |        11.19 |
| floor highlight position (2 numbers) | global   | test       |    0.00 |    38.08 |       33.80 |        42.42 |    11.79 |       10.86 |        12.66 |
| floor highlight position (2 numbers) | global   | test_cubes |    0.00 |    37.35 |       35.58 |        39.20 |    11.41 |       11.01 |        11.84 |
| floor highlight position (2 numbers) | local    | test       |    0.00 |    38.08 |       33.80 |        42.42 |    11.79 |       10.86 |        12.66 |
| floor highlight position (2 numbers) | local    | test_cubes |    0.00 |    37.35 |       35.58 |        39.20 |    11.41 |       11.01 |        11.84 |
