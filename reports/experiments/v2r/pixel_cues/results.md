| variant                              | target   | split      |   alpha |   az_mae |   az_ci_low |   az_ci_high |   el_mae |   el_ci_low |   el_ci_high |
|:-------------------------------------|:---------|:-----------|--------:|---------:|------------:|-------------:|---------:|------------:|-------------:|
| full image                           | global   | test       |   30.35 |    25.06 |       21.63 |        28.71 |     1.82 |        1.65 |         1.99 |
| full image                           | global   | test_cubes |   30.35 |    25.81 |       24.42 |        27.25 |     1.97 |        1.89 |         2.05 |
| full image                           | local    | test       |   30.35 |    25.06 |       21.63 |        28.71 |     1.82 |        1.65 |         1.99 |
| full image                           | local    | test_cubes |   30.35 |    25.81 |       24.42 |        27.25 |     1.97 |        1.89 |         2.05 |
| floor only (object removed)          | global   | test       |    6.10 |    31.07 |       26.96 |        35.67 |     1.87 |        1.70 |         2.04 |
| floor only (object removed)          | global   | test_cubes |    6.10 |    30.45 |       28.61 |        32.32 |     1.96 |        1.88 |         2.04 |
| floor only (object removed)          | local    | test       |    6.10 |    31.07 |       26.96 |        35.67 |     1.87 |        1.70 |         2.04 |
| floor only (object removed)          | local    | test_cubes |    6.10 |    30.45 |       28.61 |        32.32 |     1.96 |        1.88 |         2.04 |
| object only (crop)                   | global   | test       |    1.86 |     4.45 |        3.94 |         4.99 |     2.30 |        1.96 |         2.66 |
| object only (crop)                   | global   | test_cubes |    1.86 |    22.23 |       21.02 |        23.43 |    11.97 |       11.33 |        12.63 |
| object only (crop)                   | local    | test       |    1.86 |     4.45 |        3.94 |         4.99 |     2.30 |        1.96 |         2.66 |
| object only (crop)                   | local    | test_cubes |    1.86 |    22.23 |       21.02 |        23.43 |    11.97 |       11.33 |        12.63 |
| floor highlight position (2 numbers) | global   | test       |    0.00 |    36.76 |       32.75 |        40.84 |    11.78 |       10.84 |        12.65 |
| floor highlight position (2 numbers) | global   | test_cubes |    0.00 |    37.09 |       35.32 |        38.92 |    11.38 |       10.99 |        11.81 |
| floor highlight position (2 numbers) | local    | test       |    0.00 |    36.76 |       32.75 |        40.84 |    11.78 |       10.84 |        12.65 |
| floor highlight position (2 numbers) | local    | test_cubes |    0.00 |    37.09 |       35.32 |        38.92 |    11.38 |       10.99 |        11.81 |
