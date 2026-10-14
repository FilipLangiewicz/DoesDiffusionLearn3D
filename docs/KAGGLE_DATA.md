# Dane projektu na Kaggle

Wszystkie duże dane leżą na Kaggle (konto `jankombajn`), a nie na laptopie. Są dwa rodzaje miejsc:

- **Dataset** – trwały zbiór plików, widoczny w *Datasets → Your Work*. Liczy się do limitu 200 GB (prywatne).
- **Wyjście notatnika** – pliki, które zapisał job, widoczne w *Code → Your Work → notatnik → Output*. Kaggle trzyma je razem z wersją notatnika i nie liczy ich do limitu datasetów. Znikają, jeśli notatnik zostanie uruchomiony ponownie (nowa wersja) albo usunięty.

Zasada: notatników z tabeli niżej **nie uruchamiać ponownie**. Najważniejsze wyjścia kopiujemy do datasetów jako zapas.

## Datasety (już istnieją)

| Dataset | Link | Zawartość | Po co |
|---|---|---|---|
| `dd3d-v1-images` | https://www.kaggle.com/datasets/jankombajn/dd3d-v1-images | 3000 renderów v1 (ZZSN) i ich pliki scen JSON, 1,1 GB | Wejście ekstrakcji v1; oryginalny dataset z ZZSN |
| `dd3d-plans` | https://www.kaggle.com/datasets/jankombajn/dd3d-plans | Plany testów przyczynowych (`plan_*.dat`: przesunięcia, macierz LEACE) | Wejście jobów `dd3d-intervene-*` |

## Datasety do utworzenia z wyjść notatników (bez laptopa)

| Dataset (proponowana nazwa) | Źródło: wyjścia notatników | Rozmiar | Dlaczego |
|---|---|---:|---|
| `dd3d-renders` | `dd3d-render-v2-0…5`, `v2pos-0`, `v2ns-0…5`, `v2r-0…5`, `v2rns-0…5`, `v2rso-0…5` | ok. 4 GB | Wszystkie nowe obrazy (v2 i warianty); bez nich nie ma eksperymentu |
| `dd3d-features-v2r` | `dd3d-extract-v2r-0…5` | ok. 58 GB | Cechy SANA, SANA z losowymi wagami, SD 1.5, SDXL dla testu rozstrzygającego (losowa pozycja) |
| `dd3d-features-v1` | `dd3d-extract-v1-0…5` | ok. 65 GB | Cechy v1 z poprawnym i starym szumem; odtworzenie i krytyka wyniku ZZSN |

Razem ok. 127 GB z 200 GB. Pozostałe wyjścia (v2, v2pos, warianty v2rns i v2rso, testy) zostają tylko jako wyjścia notatników.

**Jak utworzyć dataset z wyjścia notatnika:** otwórz notatnik → *Output* → przycisk tworzenia nowego datasetu z wyjścia (*New Dataset*). Jeśli okno pozwala dodać kilka notatników, dodaj wszystkie z wiersza tabeli; jeśli nie, utwórz jeden dataset na notatnik z dopiskiem części, np. `dd3d-features-v2r-0`. Dokładny wygląd przycisku mógł się zmienić, sprawdź w interfejsie.

## Wyjścia notatników

Link do wyjścia: `https://www.kaggle.com/code/jankombajn/<nazwa>/output`

| Notatniki | Zawartość wyjścia | Stan (14.10, 15:10) |
|---|---|---|
| `dd3d-render-v2-0…5` | `v2_<od>_<do>.tar`: obrazy v2 (obiekt w środku) i `labels.csv` | gotowe |
| `dd3d-render-v2pos-0` | 300 obrazów v2 z obiektem w losowym miejscu (tylko test) | gotowe |
| `dd3d-render-v2ns-0…5` | v2 bez cienia obiektu | gotowe |
| `dd3d-render-v2r-0…5` | v2r: to samo światło co v2, obiekt w losowym miejscu | gotowe |
| `dd3d-render-v2rns-0…5`, `v2rso-0…5` | v2r bez cienia / sam cień | gotowe |
| `dd3d-extract-v1-0…5` | `v1_taskNN.tar`: cechy v1 (5 przebiegów), ok. 11 GB na część | 0–3 gotowe, 4–5 liczą się |
| `dd3d-extract-v2r-0…5` | `v2r_taskNN.tar`: cechy v2r (4 przebiegi), ok. 9,6 GB na część | gotowe |
| `dd3d-extract-v2pos-0` | cechy v2pos | gotowe |
| `dd3d-extract-v2-0…5` | cechy v2 (SANA, SD 1.5) | w kolejce |
| `dd3d-extract-v2rns-0…5`, `v2rso-0…5` | cechy wariantów (SANA) | w kolejce |
| `dd3d-intervene-test` | próba techniczna testu przyczynowego | gotowe |

Pobieranie pojedynczego pliku na laptopa (tylko gdy naprawdę trzeba, pliki są duże): `python tools/kaggle/kfetch.py jankombajn/<nazwa> <folder> --pattern <regex>`.
