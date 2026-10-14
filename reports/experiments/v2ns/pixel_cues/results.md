| variant                              | target   | split      |   alpha |   az_mae |   az_ci_low |   az_ci_high |   el_mae |   el_ci_low |   el_ci_high |
|:-------------------------------------|:---------|:-----------|--------:|---------:|------------:|-------------:|---------:|------------:|-------------:|
| full image                           | global   | test       |    0.07 |     4.19 |        3.70 |         4.71 |     1.26 |        1.14 |         1.39 |
| full image                           | global   | test_cubes |    0.07 |    69.90 |       67.01 |        72.95 |    21.13 |       19.95 |        22.32 |
| full image                           | local    | test       |    0.07 |     4.19 |        3.70 |         4.71 |     1.26 |        1.14 |         1.39 |
| full image                           | local    | test_cubes |    0.07 |    69.90 |       67.01 |        72.95 |    21.13 |       19.95 |        22.32 |
| floor only (object removed)          | global   | test       |    0.04 |     4.61 |        4.00 |         5.34 |     1.58 |        1.44 |         1.72 |
| floor only (object removed)          | global   | test_cubes |    0.04 |    10.22 |        9.72 |        10.66 |     1.77 |        1.70 |         1.84 |
| floor only (object removed)          | local    | test       |    0.04 |     4.61 |        4.00 |         5.34 |     1.58 |        1.44 |         1.72 |
| floor only (object removed)          | local    | test_cubes |    0.04 |    10.22 |        9.72 |        10.66 |     1.77 |        1.70 |         1.84 |
| object only (crop)                   | global   | test       |    0.29 |     3.25 |        2.80 |         3.78 |     0.92 |        0.81 |         1.04 |
| object only (crop)                   | global   | test_cubes |    0.29 |    39.80 |       37.59 |        42.07 |     9.88 |        9.44 |        10.33 |
| object only (crop)                   | local    | test       |    0.29 |     3.25 |        2.80 |         3.78 |     0.92 |        0.81 |         1.04 |
| object only (crop)                   | local    | test_cubes |    0.29 |    39.80 |       37.59 |        42.07 |     9.88 |        9.44 |        10.33 |
| floor highlight position (2 numbers) | global   | test       |    0.11 |    11.01 |        9.35 |        12.81 |    11.79 |       10.88 |        12.64 |
| floor highlight position (2 numbers) | global   | test_cubes |    0.11 |    17.87 |       16.99 |        18.77 |    11.26 |       10.86 |        11.69 |
| floor highlight position (2 numbers) | local    | test       |    0.11 |    11.01 |        9.35 |        12.81 |    11.79 |       10.88 |        12.64 |
| floor highlight position (2 numbers) | local    | test_cubes |    0.11 |    17.87 |       16.99 |        18.77 |    11.26 |       10.86 |        11.69 |
