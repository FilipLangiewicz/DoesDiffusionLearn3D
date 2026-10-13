| variant                              | target   | split      |   alpha |   az_mae |   az_ci_low |   az_ci_high |   el_mae |   el_ci_low |   el_ci_high |
|:-------------------------------------|:---------|:-----------|--------:|---------:|------------:|-------------:|---------:|------------:|-------------:|
| full image                           | global   | test       |    3.16 |    13.02 |       10.97 |        15.41 |     5.74 |        5.14 |         6.33 |
| full image                           | global   | test_cubes |    3.16 |    15.23 |       13.99 |        16.45 |     6.34 |        6.01 |         6.68 |
| full image                           | local    | test       |  100.00 |    13.66 |       10.93 |        16.59 |     9.27 |        8.27 |        10.28 |
| full image                           | local    | test_cubes |  100.00 |    12.94 |       11.89 |        14.11 |     8.15 |        7.69 |         8.61 |
| floor only (object removed)          | global   | test       |    0.01 |     8.29 |        7.30 |         9.40 |     3.96 |        3.55 |         4.38 |
| floor only (object removed)          | global   | test_cubes |    0.01 |     9.47 |        8.85 |        10.09 |     4.39 |        4.17 |         4.63 |
| floor only (object removed)          | local    | test       |  316.23 |    14.85 |       12.75 |        17.09 |     7.53 |        6.56 |         8.59 |
| floor only (object removed)          | local    | test_cubes |  316.23 |    13.37 |       12.20 |        14.65 |     6.95 |        6.54 |         7.35 |
| object only (crop)                   | global   | test       |    0.01 |    20.13 |       17.67 |        22.78 |     7.99 |        7.15 |         8.88 |
| object only (crop)                   | global   | test_cubes |    0.01 |    50.45 |       47.84 |        52.88 |    13.54 |       12.90 |        14.16 |
| object only (crop)                   | local    | test       |    1.00 |     7.96 |        6.99 |         9.00 |     5.50 |        4.87 |         6.16 |
| object only (crop)                   | local    | test_cubes |    1.00 |    27.59 |       26.07 |        29.18 |    10.16 |        9.66 |        10.65 |
| floor highlight position (2 numbers) | global   | test       |    0.01 |    12.25 |       10.34 |        14.53 |     7.39 |        6.75 |         8.06 |
| floor highlight position (2 numbers) | global   | test_cubes |    0.01 |    17.59 |       16.11 |        19.31 |     8.11 |        7.71 |         8.57 |
| floor highlight position (2 numbers) | local    | test       |    0.01 |    30.44 |       26.68 |        34.49 |    13.81 |       12.54 |        15.19 |
| floor highlight position (2 numbers) | local    | test_cubes |    0.01 |    30.70 |       29.05 |        32.44 |    14.13 |       13.51 |        14.74 |
