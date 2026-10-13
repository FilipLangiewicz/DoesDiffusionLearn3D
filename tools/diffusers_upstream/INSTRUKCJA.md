# Jak zgłosić błąd i poprawkę do diffusers (krok po kroku)

Repozytorium: https://github.com/huggingface/diffusers (licencja Apache-2.0).

Diffusers ma jedną ważną zasadę dla zmian robionych z pomocą AI: **najpierw issue, potem czekasz, aż opiekun biblioteki potwierdzi, dopiero wtedy PR**. PR otwarty bez tego mogą zamknąć bez przeglądu. Są więc dwa etapy, a między nimi zwykle kilka dni.

Pliki w tym folderze:

| Plik | Do czego |
|---|---|
| `ISSUE.md` | treść zgłoszenia błędu (etap 1) |
| `PR.md` | tytuł i opis pull requesta (etap 2) |
| `fix.patch` | sama poprawka (4 pliki schedulerów + testy) |
| `repro_sana.py` | odtworzenie błędu na prawdziwej SANA 1.6B (wymaga GPU) |
| `functional/` | skrypty testów na prawdziwych modelach (CIFAR i SANA 600M) |

---

## Etap 0. Zanim zaczniesz

1. Zapytaj prowadzącego SIWY:
   - czy PR do diffusers liczy się na ocenę 5,
   - czy wystarczy otwarcie PR, czy musi zostać przyjęty,
   - czy zaliczy się całemu zespołowi.
2. Przeczytaj `ISSUE.md` i `PR.md` i zmień wszystko, czego nie rozumiesz albo czego nie napisałbyś sam. Diffusers wymaga, żeby autor rozumiał i podpisywał się pod każdym słowem.
3. Zalecane: uruchom `repro_sana.py` na Athenie (sekcja na końcu) i wklej liczby dla SANA 1.6B do `ISSUE.md`, w miejsce komentarza „Add the SANA-1.6B numbers”. Wytyczne diffusers wolą odtworzenie na tym samym modelu, na którym znaleziono błąd.

## Etap 1. Issue

1. Zaloguj się na https://github.com.
2. Otwórz https://github.com/huggingface/diffusers/issues/new/choose i wybierz **Bug Report**.
3. Przepisz pola z `ISSUE.md`:
   - **Title**: linia pod `# Title`;
   - **Describe the bug**, **Reproduction**, **Logs**, **Who can help?**: odpowiednie sekcje;
   - **System Info**: wynik komendy `diffusers-cli env`. W naszym środowisku uruchamiasz ją tak: `.venv/bin/diffusers-cli env`.
4. Kliknij **Submit new issue** i zapisz numer issue, np. `#15020`.
5. Czekaj na odpowiedź opiekuna, np. „sounds good, feel free to open a PR”. Jeśli przez tydzień nikt nie odpowie, dopisz grzeczny komentarz z prośbą o opinię.

## Etap 2. Pull request (dopiero po potwierdzeniu)

1. Na stronie https://github.com/huggingface/diffusers kliknij **Fork**, żeby mieć własną kopię repozytorium.
2. W terminalu:

   ```bash
   git clone https://github.com/<TWOJ_LOGIN>/diffusers.git
   cd diffusers
   git remote add upstream https://github.com/huggingface/diffusers.git
   git fetch upstream
   git checkout -b fix-add-noise-sigma-order upstream/main
   ```

3. Nałóż poprawkę (ścieżkę dopasuj do swojego repo projektu):

   ```bash
   git apply ~/studia-szi/sem3/siwy/pro/repo/DoesDiffusionLearn3D/tools/diffusers_upstream/fix.patch
   git status   # 5 zmienionych plików
   ```

   Jeśli `git apply` zgłosi konflikt, to w międzyczasie ktoś zmienił te pliki. Wtedy napisz do mnie.

4. Środowisko i sprawdzenia:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[test,quality]" torchsde
   make style && make fix-copies          # nie powinny nic zmienić
   python -m pytest tests/schedulers/test_scheduler_dpm_multi.py tests/schedulers/test_scheduler_dpm_single.py \
     tests/schedulers/test_scheduler_unipc.py tests/schedulers/test_scheduler_deis.py
   ```

   Wszystko musi przejść. Jeśli liczby różnią się od tych w `PR.md`, popraw je w opisie.

5. Przeczytaj zmiany: `git diff`. W każdym z 4 schedulerów jest jedna zmiana w `__init__` (przypisanie `self.sigmas`), plus testy. Musisz umieć ją wyjaśnić recenzentowi. Wyjaśnienie jest w `docs/PODSUMOWANIE.md`, rozdział 9.

6. Commit i wysłanie do swojego forka:

   ```bash
   git add src/diffusers/schedulers tests/schedulers
   git commit -m "[schedulers] Fix the sigma order for add_noise() before set_timesteps()"
   git push origin fix-add-noise-sigma-order
   ```

7. Na GitHubie pojawi się przycisk **Compare & pull request**. Ustaw cel: `huggingface/diffusers`, gałąź `main`.
8. Tytuł i opis skopiuj z `PR.md`. W miejsce `#<ISSUE NUMBER>` wstaw numer issue z etapu 1. Zaznacz pola listy kontrolnej zgodnie z sekcją „Checklist of the template”.
9. Kliknij **Create pull request**. Potem odpowiadaj na uwagi recenzentów. Poprawki robisz kolejnymi commitami na tej samej gałęzi i `git push`.

## Ważne: pytanie o AI

Szablon PR diffusers pyta wprost: „Did you use an AI agent…?”. Poprawkę przygotował agent AI, więc **odpowiedź musi być „tak”**. Diffusers tego nie karze: przyjmują takie zmiany, ale wymagają uczciwego zaznaczenia i notatek z przeglądu (są w `PR.md`). Nieprawdziwa odpowiedź grozi zamknięciem PR, jeśli wyjdzie na jaw. Opis nie musi wymieniać nazwy narzędzia.

## Uruchomienie `repro_sana.py` na Athenie

Na węźle GPU, w środowisku z `diffusers` (np. tym z projektu: `pip install -e ".[model]"` w repo projektu):

```bash
srun -A <GRANT>-gpu-a100 -p plgrid-gpu-a100 --gres=gpu:1 -c 4 --mem=40G -t 00:20:00 \
  python tools/diffusers_upstream/repro_sana.py
```

Skrypt wypisze tabelę: współczynniki `a`, `b` oraz błąd odtworzenia z `add_noise` i z poprawnym szumem dla t = 200, 500 i 800. Dla t = 200 błąd z `add_noise` powinien być dużo większy niż z poprawnym szumem.
