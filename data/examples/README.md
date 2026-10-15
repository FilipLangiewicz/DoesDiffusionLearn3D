# Examples of the data set versions

This folder has three images of each data set version (256 px JPEG) and one overview sheet, `overview.jpg`. The full data sets are on Kaggle only. `docs/KAGGLE_DATA.md` gives the links.

The new versions use the same scene numbers. Thus the rows of the sheet for v2, v2ns, v2r, v2rns, v2rso and v2rot differ from the v2 row in one property only.

| Version | What changes | Images | Labels and splits |
|---|---|---:|---|
| v1 | The original data of the ZZSN project: an area lamp turns around the scene centre, and the sun lamp of the base scene is on. | 3000 | `data/master_labels.csv`, `data/splits/` |
| v2 | One sun lamp (only a direction), a flat floor, the object at the scene centre. | 3000 | `data/v2/` |
| v2ns | As v2, but the object casts no shadow. | 3000 | `data/v2ns/` |
| v2pos | As v2, but the object stands at a random position. Test only, with its own scenes. | 300 | `data/v2pos/` |
| v2r | As v2, but the object stands at a random position. | 3000 | `data/v2r/` |
| v2rns | As v2r, but the object casts no shadow. | 3000 | `data/v2rns/` |
| v2rso | As v2r, but the camera does not see the object. Only its shadow stays. | 3000 | `data/v2rso/` |
| v2rot | As v2, but the light azimuth is the v2 azimuth + 90 deg. Pairs for the patch test. | 48 | `configs/kaggle/patch_pairs.txt` |

The script `scripts/data/make_examples.py` makes this folder from the full images.
