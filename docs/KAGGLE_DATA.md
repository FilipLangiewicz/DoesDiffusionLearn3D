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

**Jak utworzyć dataset z wyjść notatników (w przeglądarce, bez laptopa):**

1. Otwórz https://www.kaggle.com/datasets i kliknij **+ New Dataset**.
2. W oknie wybierz źródło **Notebook Output Files** (czasem jako zakładka *Code* / *Notebooks*). Nie wybieraj wysyłania plików z komputera.
3. Wyszukaj pierwszy notatnik z wiersza tabeli (np. `dd3d-extract-v2r-0`) i dodaj jego wyjście. Jeśli okno pozwala, dodaj kolejne notatniki z tego samego wiersza.
4. Wpisz tytuł z tabeli (np. `dd3d-features-v2r`), zostaw **Private** i kliknij **Create**.
5. Jeśli okno przyjmuje tylko jeden notatnik, utwórz osobny dataset na każdą część: `dd3d-features-v2r-0`, `dd3d-features-v2r-1` itd.
6. Utworzenie dużego datasetu (ok. 10 GB na część) trwa kilka–kilkanaście minut; Kaggle kopiuje pliki u siebie.

Linki do gotowych datasetów dopisz do tabeli „Datasety (już istnieją)”. `dd3d-features-v1` twórz dopiero, gdy części `v1-4` i `v1-5` będą gotowe.

## Wyjścia notatników

Zasada: tych notatników nie uruchamiać ponownie. Stan sprawdzisz po kliknięciu.

| Notatnik | Wyjście | Zawartość |
|---|---|---|
| [dd3d-render-v2-0](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-0/output) | obrazy v2 (obiekt w środku) i `labels.csv` |
| [dd3d-render-v2-1](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-1/output) | obrazy v2 (obiekt w środku) i `labels.csv` |
| [dd3d-render-v2-2](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-2/output) | obrazy v2 (obiekt w środku) i `labels.csv` |
| [dd3d-render-v2-3](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-3/output) | obrazy v2 (obiekt w środku) i `labels.csv` |
| [dd3d-render-v2-4](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-4/output) | obrazy v2 (obiekt w środku) i `labels.csv` |
| [dd3d-render-v2-5](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2-5/output) | obrazy v2 (obiekt w środku) i `labels.csv` |
| [dd3d-render-v2pos-0](https://www.kaggle.com/code/jankombajn/dd3d-render-v2pos-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2pos-0/output) | 300 obrazów z obiektem w losowym miejscu (test) |
| [dd3d-render-v2ns-0](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-0/output) | v2 bez cienia obiektu |
| [dd3d-render-v2ns-1](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-1/output) | v2 bez cienia obiektu |
| [dd3d-render-v2ns-2](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-2/output) | v2 bez cienia obiektu |
| [dd3d-render-v2ns-3](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-3/output) | v2 bez cienia obiektu |
| [dd3d-render-v2ns-4](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-4/output) | v2 bez cienia obiektu |
| [dd3d-render-v2ns-5](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2ns-5/output) | v2 bez cienia obiektu |
| [dd3d-render-v2r-0](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-0/output) | v2r: to samo światło co v2, losowa pozycja |
| [dd3d-render-v2r-1](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-1/output) | v2r: to samo światło co v2, losowa pozycja |
| [dd3d-render-v2r-2](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-2/output) | v2r: to samo światło co v2, losowa pozycja |
| [dd3d-render-v2r-3](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-3/output) | v2r: to samo światło co v2, losowa pozycja |
| [dd3d-render-v2r-4](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-4/output) | v2r: to samo światło co v2, losowa pozycja |
| [dd3d-render-v2r-5](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2r-5/output) | v2r: to samo światło co v2, losowa pozycja |
| [dd3d-render-v2rns-0](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-0/output) | v2r bez cienia obiektu |
| [dd3d-render-v2rns-1](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-1/output) | v2r bez cienia obiektu |
| [dd3d-render-v2rns-2](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-2/output) | v2r bez cienia obiektu |
| [dd3d-render-v2rns-3](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-3/output) | v2r bez cienia obiektu |
| [dd3d-render-v2rns-4](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-4/output) | v2r bez cienia obiektu |
| [dd3d-render-v2rns-5](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rns-5/output) | v2r bez cienia obiektu |
| [dd3d-render-v2rso-0](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-0/output) | v2r, sam cień (obiekt niewidoczny) |
| [dd3d-render-v2rso-1](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-1/output) | v2r, sam cień (obiekt niewidoczny) |
| [dd3d-render-v2rso-2](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-2/output) | v2r, sam cień (obiekt niewidoczny) |
| [dd3d-render-v2rso-3](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-3/output) | v2r, sam cień (obiekt niewidoczny) |
| [dd3d-render-v2rso-4](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-4/output) | v2r, sam cień (obiekt niewidoczny) |
| [dd3d-render-v2rso-5](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-render-v2rso-5/output) | v2r, sam cień (obiekt niewidoczny) |
| [dd3d-extract-v1-0](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-0/output) | cechy v1: SANA, losowe wagi, stary szum, SD 1.5, SDXL (ok. 11 GB) |
| [dd3d-extract-v1-1](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-1/output) | cechy v1: SANA, losowe wagi, stary szum, SD 1.5, SDXL (ok. 11 GB) |
| [dd3d-extract-v1-2](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-2/output) | cechy v1: SANA, losowe wagi, stary szum, SD 1.5, SDXL (ok. 11 GB) |
| [dd3d-extract-v1-3](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-3/output) | cechy v1: SANA, losowe wagi, stary szum, SD 1.5, SDXL (ok. 11 GB) |
| [dd3d-extract-v1-4](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-4/output) | cechy v1: SANA, losowe wagi, stary szum, SD 1.5, SDXL (ok. 11 GB) |
| [dd3d-extract-v1-5](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v1-5/output) | cechy v1: SANA, losowe wagi, stary szum, SD 1.5, SDXL (ok. 11 GB) |
| [dd3d-extract-v2r-0](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-0/output) | cechy v2r: SANA, losowe wagi, SD 1.5, SDXL (ok. 9,6 GB) |
| [dd3d-extract-v2r-1](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-1/output) | cechy v2r: SANA, losowe wagi, SD 1.5, SDXL (ok. 9,6 GB) |
| [dd3d-extract-v2r-2](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-2/output) | cechy v2r: SANA, losowe wagi, SD 1.5, SDXL (ok. 9,6 GB) |
| [dd3d-extract-v2r-3](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-3/output) | cechy v2r: SANA, losowe wagi, SD 1.5, SDXL (ok. 9,6 GB) |
| [dd3d-extract-v2r-4](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-4/output) | cechy v2r: SANA, losowe wagi, SD 1.5, SDXL (ok. 9,6 GB) |
| [dd3d-extract-v2r-5](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2r-5/output) | cechy v2r: SANA, losowe wagi, SD 1.5, SDXL (ok. 9,6 GB) |
| [dd3d-extract-v2pos-0](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2pos-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2pos-0/output) | cechy v2pos: SANA, losowe wagi, SD 1.5, SDXL |
| [dd3d-extract-v2-0](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-0/output) | cechy v2: SANA, SD 1.5 |
| [dd3d-extract-v2-1](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-1/output) | cechy v2: SANA, SD 1.5 |
| [dd3d-extract-v2-2](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-2/output) | cechy v2: SANA, SD 1.5 |
| [dd3d-extract-v2-3](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-3/output) | cechy v2: SANA, SD 1.5 |
| [dd3d-extract-v2-4](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-4/output) | cechy v2: SANA, SD 1.5 |
| [dd3d-extract-v2-5](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2-5/output) | cechy v2: SANA, SD 1.5 |
| [dd3d-extract-v2rns-0](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-0/output) | cechy v2rns: SANA |
| [dd3d-extract-v2rns-1](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-1/output) | cechy v2rns: SANA |
| [dd3d-extract-v2rns-2](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-2/output) | cechy v2rns: SANA |
| [dd3d-extract-v2rns-3](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-3/output) | cechy v2rns: SANA |
| [dd3d-extract-v2rns-4](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-4/output) | cechy v2rns: SANA |
| [dd3d-extract-v2rns-5](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rns-5/output) | cechy v2rns: SANA |
| [dd3d-extract-v2rso-0](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-0) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-0/output) | cechy v2rso: SANA |
| [dd3d-extract-v2rso-1](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-1) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-1/output) | cechy v2rso: SANA |
| [dd3d-extract-v2rso-2](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-2) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-2/output) | cechy v2rso: SANA |
| [dd3d-extract-v2rso-3](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-3) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-3/output) | cechy v2rso: SANA |
| [dd3d-extract-v2rso-4](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-4) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-4/output) | cechy v2rso: SANA |
| [dd3d-extract-v2rso-5](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-5) | [output](https://www.kaggle.com/code/jankombajn/dd3d-extract-v2rso-5/output) | cechy v2rso: SANA |
| [dd3d-intervene-test](https://www.kaggle.com/code/jankombajn/dd3d-intervene-test) | [output](https://www.kaggle.com/code/jankombajn/dd3d-intervene-test/output) | próba techniczna testu przyczynowego |

Pobieranie pojedynczego pliku na laptopa (tylko gdy naprawdę trzeba, pliki są duże): `python tools/kaggle/kfetch.py jankombajn/<nazwa> <folder> --pattern <regex>`.
