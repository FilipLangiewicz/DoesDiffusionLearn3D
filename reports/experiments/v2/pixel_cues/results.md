| variant                              | target   | split      |   alpha |   az_mae |   az_ci_low |   az_ci_high |   el_mae |   el_ci_low |   el_ci_high |
|:-------------------------------------|:---------|:-----------|--------:|---------:|------------:|-------------:|---------:|------------:|-------------:|
| full image                           | global   | test       |    8.39 |     2.51 |        2.16 |         2.87 |     0.91 |        0.81 |         1.01 |
| full image                           | global   | test_cubes |    8.39 |    13.67 |       13.01 |        14.36 |     3.36 |        3.17 |         3.53 |
| full image                           | local    | test       |    8.39 |     2.51 |        2.16 |         2.87 |     0.91 |        0.81 |         1.01 |
| full image                           | local    | test_cubes |    8.39 |    13.67 |       13.01 |        14.36 |     3.36 |        3.17 |         3.53 |
| floor only (object removed)          | global   | test       |    5.87 |     6.38 |        5.46 |         7.44 |     1.60 |        1.45 |         1.74 |
| floor only (object removed)          | global   | test_cubes |    5.87 |    12.43 |       11.75 |        13.15 |     1.72 |        1.64 |         1.80 |
| floor only (object removed)          | local    | test       |    5.87 |     6.38 |        5.46 |         7.44 |     1.60 |        1.45 |         1.74 |
| floor only (object removed)          | local    | test_cubes |    5.87 |    12.43 |       11.75 |        13.15 |     1.72 |        1.64 |         1.80 |
| object only (crop)                   | global   | test       |    1.00 |     3.19 |        2.84 |         3.58 |     1.21 |        1.06 |         1.36 |
| object only (crop)                   | global   | test_cubes |    1.00 |    24.75 |       23.51 |        26.00 |     7.68 |        7.33 |         8.04 |
| object only (crop)                   | local    | test       |    1.00 |     3.19 |        2.84 |         3.58 |     1.21 |        1.06 |         1.36 |
| object only (crop)                   | local    | test_cubes |    1.00 |    24.75 |       23.51 |        26.00 |     7.68 |        7.33 |         8.04 |
| floor highlight position (2 numbers) | global   | test       |    0.13 |    11.97 |        9.61 |        14.80 |    11.74 |       10.82 |        12.63 |
| floor highlight position (2 numbers) | global   | test_cubes |    0.13 |    17.95 |       17.08 |        18.83 |    11.18 |       10.77 |        11.62 |
| floor highlight position (2 numbers) | local    | test       |    0.13 |    11.97 |        9.61 |        14.80 |    11.74 |       10.82 |        12.63 |
| floor highlight position (2 numbers) | local    | test_cubes |    0.13 |    17.95 |       17.08 |        18.83 |    11.18 |       10.77 |        11.62 |
