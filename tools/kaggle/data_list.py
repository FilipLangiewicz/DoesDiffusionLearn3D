"""Write docs/KAGGLE_DATA.md: the Kaggle data sets and job outputs, with their state."""
# ruff: noqa: E501

from pathlib import Path

PART = {
    0: "kule 0–499",
    1: "kule 500–999",
    2: "walce 0–499",
    3: "walce 500–999",
    4: "sześciany 0–499",
    5: "sześciany 500–999",
}
NEW = "nowa podłoga i światło"
GROUPS = [
    ("render", "v2", 6, f"obiekt na środku; {NEW}; obrazy + etykiety"),
    ("render", "v2pos", 1, f"losowa pozycja; {NEW}; tylko test; 300 obrazów (kule, walce, sześciany)"),
    ("render", "v2ns", 6, f"obiekt na środku; {NEW}; bez cienia obiektu; obrazy"),
    ("render", "v2r", 6, f"losowa pozycja; {NEW}; obrazy"),
    ("render", "v2rns", 6, f"losowa pozycja; {NEW}; bez cienia obiektu; obrazy"),
    ("render", "v2rso", 6, f"losowa pozycja; {NEW}; sam cień, obiekt niewidoczny; obrazy"),
    (
        "extract",
        "v1",
        6,
        "oryginalny zbiór v1 (ZZSN); aktywacje: SANA, SANA losowe wagi, SANA stary szum, SD 1.5, SDXL",
    ),
    ("extract", "v2", 6, f"obiekt na środku; {NEW}; aktywacje: SANA, SD 1.5"),
    (
        "extract",
        "v2pos",
        1,
        f"losowa pozycja; {NEW}; tylko test; aktywacje: SANA, SANA losowe wagi, SD 1.5, SDXL",
    ),
    ("extract", "v2r", 6, f"losowa pozycja; {NEW}; aktywacje: SANA, SANA losowe wagi, SD 1.5, SDXL"),
    ("extract", "v2rns", 6, f"losowa pozycja; {NEW}; bez cienia obiektu; aktywacje: SANA"),
    ("extract", "v2rso", 6, f"losowa pozycja; {NEW}; sam cień; aktywacje: SANA"),
    ("intervene", "test", 0, "test przyczynowy, próba techniczna; obiekt na środku; 4 obrazy"),
]


def state(d: Path) -> str:
    if (d / ".done").exists() or (d / ".complete").exists():
        return "gotowe"
    if (d / ".failed").exists():
        return "błąd"
    return "liczy się" if (d / ".pushed").exists() else "w kolejce"


rows = [
    "# Dane projektu na Kaggle",
    "",
    "## Datasety",
    "",
    "| Dataset | Opis |",
    "|---|---|",
    "| [dd3d-v1-images](https://www.kaggle.com/datasets/jankombajn/dd3d-v1-images) | oryginalny zbiór v1 (ZZSN); kule, walce, sześciany; 3000 obrazów + opisy scen |",
    "| [dd3d-plans](https://www.kaggle.com/datasets/jankombajn/dd3d-plans) | plany testów przyczynowych (przesunięcie, wymazanie kierunku światła) |",
    *[
        f"| [dd3d-features-{n}-{i}](https://www.kaggle.com/datasets/jankombajn/dd3d-features-{n}-{i}) | ({n}-{i}) {PART[i]}; {d} |"
        for n, d in (
            ("v2r", f"losowa pozycja; {NEW}; aktywacje: SANA, SANA losowe wagi, SD 1.5, SDXL"),
            ("v1", "oryginalny zbiór v1 (ZZSN); aktywacje: SANA, SANA losowe wagi, SANA stary szum, SD 1.5, SDXL"),
        )
        for i in range(6)
    ],
    "",
    "## Wyjścia notatników",
    "",
    "| Notatnik | Wyjście | Opis | Stan |",
    "|---|---|---|---|",
]
for kind, name, n, desc in GROUPS:
    for i in range(max(n, 1)):
        slug = f"dd3d-{kind}-{name}" + (f"-{i}" if n else "")
        tag = f"{name}-{i}" if n else name
        which = f"{PART[i]}; " if n == 6 else ""
        st = state(Path(f"runs/kaggle/{kind}/{slug}"))
        url = f"https://www.kaggle.com/code/jankombajn/{slug}"
        rows.append(f"| [{slug}]({url}) | [output]({url}/output) | ({tag}) {which}{desc} | {st} |")
Path("docs/KAGGLE_DATA.md").write_text("\n".join(rows) + "\n")
