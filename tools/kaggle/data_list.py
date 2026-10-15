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
    (
        "misc",
        "renders",
        0,
        f"wszystkie nowe obrazy w tarach (v2, v2pos, v2ns, v2r, v2rns, v2rso); {NEW}; źródło datasetu dd3d-renders-tar",
    ),
    (
        "misc",
        "renders-png",
        0,
        f"wszystkie nowe obrazy rozpakowane; {NEW}; niepełne: limit 500 plików na wyjście",
    ),
    ("intervene", "test", 0, "test przyczynowy, próba techniczna; obiekt na środku; 4 obrazy"),
    ("analyze", "smoke", 0, "analiza, próba techniczna: sondy na bloku 10 i 3 (t = 400) dla v2, v2r, v2rso"),
    *[
        (
            "analyze",
            f"scan-{s}",
            0,
            f"analiza {s}: sondy na wszystkich blokach i krokach + piksele i latent; małe pliki CSV i PNG; kopia lokalnie w reports/experiments/{s}/scan",
        )
        for s in ("v1", "v2", "v2r", "v2rns", "v2rso")
    ],
    *[
        (
            "analyze",
            f"deep-{s}",
            0,
            f"analiza {s}, etap 2: wybrane sondy (porównania parami, kontrola z przetasowanymi etykietami) i mapy, gdzie sonda czyta; kopia w reports/experiments/{s}/best i maps",
        )
        for s in ("v1", "v2", "v2r", "v2rns", "v2rso")
    ],
    (
        "analyze",
        "transfer-pos",
        0,
        "transfer: sonda z v2 (obiekt na środku) testowana na v2r (losowa pozycja); kopia w reports/experiments/transfer/pos",
    ),
    (
        "analyze",
        "transfer-cue",
        0,
        "transfer: sonda z v2r testowana na v2rns (bez cienia) i v2rso (sam cień); kopia w reports/experiments/transfer/cue",
    ),
    (
        "analyze",
        "plans-v2",
        0,
        "plany testu przyczynowego na v2 (bloki 4, 10, 18; t = 400): przesunięcie wzdłuż sondy i wzorca, wymazanie LEACE",
    ),
    ("render", "v2rot", 1, f"pary do łatania aktywacji: 48 scen testowych v2 z azymutem +90°; {NEW}"),
    ("intervene", "test2", 0, "test przyczynowy, próba techniczna na 2 kartach: 2 losowe plany po 2 obrazy"),
    (
        "intervene",
        "causal",
        0,
        "test przyczynowy na v2: 60 obrazów, bloki 4, 10, 18; obrazy wyjściowe w tarach",
    ),
    (
        "intervene",
        "patch",
        0,
        "łatanie aktywacji między parami v2 / v2rot: 48 par, 10 bloków, 3 obszary, połówki kroków; obrazy w tarach",
    ),
    ("intervene", "patchtest", 0, "łatanie aktywacji, próba techniczna: 2 pary, blok 10"),
    (
        "intervene",
        "sweep",
        0,
        "przemiatanie kierunku wzorca (bloki 4 i 10, 11 kątów, 3 siły, 6 scen) do podglądu; obrazy w tarach",
    ),
    ("analyze", "smoke2", 0, "analiza, druga próba techniczna (rendery z tarów, siatki 1x1 i 2x2)"),
    (
        "analyze",
        "causal-v2",
        0,
        "analiza testu przyczynowego: obrót światła (sonda pikselowa i obszar cienia); kopia w reports/experiments/causal",
    ),
    (
        "analyze",
        "patch-v2",
        0,
        "analiza łatania: część obrotu światła dla bloku i obszaru; kopia w reports/experiments/patch",
    ),
    (
        "analyze",
        "transfer-cue-pooled",
        0,
        "transfer v2r -> v2rns, v2rso także dla średniej tokenów (1x1, 2x2); kopia w reports/experiments/transfer/cue_pooled",
    ),
    (
        "analyze",
        "sweep-plans",
        0,
        "plany przemiatania kierunku wzorca (pliki npz zostają na Kaggle, lokalnie tylko JSON)",
    ),
    (
        "analyze",
        "viewer-data",
        0,
        "dane podglądu: obrazy JPEG i odczyty kierunku z przemiatania i łatania (małe pliki JSON)",
    ),
]


def state(d: Path) -> str:
    if not d.exists():
        return "planowane"
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
    f"| [dd3d-renders-tar](https://www.kaggle.com/datasets/jankombajn/dd3d-renders-tar) | wszystkie nowe obrazy: v2, v2pos, v2ns (obiekt na środku), v2r, v2rns, v2rso (losowa pozycja); {NEW}; obrazy do przeglądania + etykiety |",
    *[
        f"| [dd3d-features-{n}-{i}](https://www.kaggle.com/datasets/jankombajn/dd3d-features-{n}-{i}) | ({n}-{i}) {PART[i]}; {d} |"
        for n, d in (
            ("v2r", f"losowa pozycja; {NEW}; aktywacje: SANA, SANA losowe wagi, SD 1.5, SDXL"),
            (
                "v1",
                "oryginalny zbiór v1 (ZZSN); aktywacje: SANA, SANA losowe wagi, SANA stary szum, SD 1.5, SDXL",
            ),
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
        slug = (f"dd3d-{name}" if kind == "misc" else f"dd3d-{kind}-{name}") + (f"-{i}" if n else "")
        tag = f"{name}-{i}" if n else name
        which = f"{PART[i]}; " if n == 6 else ""
        st = state(Path(f"runs/kaggle/{kind}/{slug}"))
        url = f"https://www.kaggle.com/code/jankombajn/{slug}"
        rows.append(f"| [{slug}]({url}) | [output]({url}/output) | ({tag}) {which}{desc} | {st} |")
Path("docs/KAGGLE_DATA.md").write_text("\n".join(rows) + "\n")
