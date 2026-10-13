| variant                              | target   | split      |   alpha |   az_mae |   az_ci_low |   az_ci_high |   el_mae |   el_ci_low |   el_ci_high |
|:-------------------------------------|:---------|:-----------|--------:|---------:|------------:|-------------:|---------:|------------:|-------------:|
| full image                           | global   | test       |    6.20 |    11.84 |        9.86 |        14.16 |     5.61 |        5.03 |         6.16 |
| full image                           | global   | test_cubes |    6.20 |    14.32 |       13.11 |        15.51 |     6.24 |        5.91 |         6.56 |
| full image                           | local    | test       |   28.79 |    11.29 |        9.33 |        13.30 |     8.75 |        7.82 |         9.69 |
| full image                           | local    | test_cubes |   28.79 |    11.56 |       10.56 |        12.68 |     7.71 |        7.31 |         8.14 |
| floor only (object removed)          | global   | test       |    0.94 |     7.94 |        7.01 |         9.00 |     3.94 |        3.54 |         4.36 |
| floor only (object removed)          | global   | test_cubes |    0.94 |     9.39 |        8.77 |        10.04 |     4.34 |        4.12 |         4.56 |
| floor only (object removed)          | local    | test       |    4.36 |    13.52 |       11.55 |        15.49 |     7.31 |        6.37 |         8.31 |
| floor only (object removed)          | local    | test_cubes |    4.36 |    12.25 |       11.18 |        13.37 |     7.19 |        6.82 |         7.60 |
| object only (crop)                   | global   | test       |    1.23 |    19.75 |       17.49 |        22.27 |     7.97 |        7.18 |         8.81 |
| object only (crop)                   | global   | test_cubes |    1.23 |    44.34 |       41.89 |        46.63 |    11.81 |       11.27 |        12.31 |
| object only (crop)                   | local    | test       |    1.23 |     7.67 |        6.76 |         8.66 |     5.58 |        5.00 |         6.18 |
| object only (crop)                   | local    | test_cubes |    1.23 |    21.50 |       20.27 |        22.84 |     8.36 |        7.94 |         8.79 |
| floor highlight position (2 numbers) | global   | test       |    0.00 |    12.25 |       10.34 |        14.53 |     7.39 |        6.75 |         8.06 |
| floor highlight position (2 numbers) | global   | test_cubes |    0.00 |    17.59 |       16.11 |        19.31 |     8.11 |        7.71 |         8.57 |
| floor highlight position (2 numbers) | local    | test       |    0.00 |    30.44 |       26.68 |        34.49 |    13.81 |       12.54 |        15.19 |
| floor highlight position (2 numbers) | local    | test_cubes |    0.00 |    30.70 |       29.05 |        32.44 |    14.13 |       13.51 |        14.74 |
