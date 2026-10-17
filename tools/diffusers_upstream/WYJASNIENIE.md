# Błąd szumu w diffusers: wyjaśnienie

Ten plik wyjaśnia błąd, który naprawia `fix.patch`, i dowody, że poprawka działa. Kroki wysłania PR są w [`INSTRUKCJA.md`](INSTRUKCJA.md).

## Czym jest diffusers?

**diffusers** to biblioteka Hugging Face, przez którą wczytujecie SANA (`SanaPipeline`). Częścią pipeline'u jest **scheduler**, obiekt, który mówi, ile szumu ma być na danym kroku t. Funkcja `scheduler.add_noise(x0, szum, t)` miesza czysty obraz z szumem w proporcji właściwej dla kroku t.

## Co było nie tak?

SANA jest modelem przepływowym (`flow matching`). Na kroku t oczekuje wejścia

  x_t = (1 − t/1000)·x0 + (t/1000)·ε,  czyli dla t = 200: 0,8·x0 + 0,2·ε.

W ZZSN wywołaliście `add_noise` **zanim** scheduler dostał `set_timesteps`. W takim stanie scheduler `DPMSolverMultistepScheduler` ma tablicę poziomów szumu w **odwrotnej kolejności i w złym formacie**, więc dla t = 200 bierze wartość dla kroku 799. Model dostawał:

| t | co dostawał model (ZZSN) | co powinien dostać |
|---:|---|---|
| 200 | −24,5·x0 + 25,5·ε (norma 35× za duża, obraz odwrócony) | 0,8·x0 + 0,2·ε |
| 500 | −2,4·x0 + 3,4·ε | 0,5·x0 + 0,5·ε |
| 800 | 0,28·x0 + 0,72·ε (prawie dobrze) | 0,2·x0 + 0,8·ε |

Zapisane aktywacje z ZZSN się z tym zgadzają: w bloku 4 średnia norma tokenu wynosi ok. 11 100 dla t = 200 i ok. 720 dla t = 800. Oś „timestep” w wynikach ZZSN jest więc błędna, a t = 200 i t = 500 to wejścia, jakich model nigdy nie widział w treningu.

## Repozytorium i proponowane zmiany

- Repozytorium: https://github.com/huggingface/diffusers (licencja **Apache-2.0**, czyli open source; ok. 34 700 gwiazdek).
- Zmiana dotyczy 4 plików, w każdym tej samej linii w `__init__`, gdzie tworzona jest tablica `self.sigmas`:
  - `src/diffusers/schedulers/scheduling_dpmsolver_multistep.py`
  - `src/diffusers/schedulers/scheduling_dpmsolver_singlestep.py`
  - `src/diffusers/schedulers/scheduling_unipc_multistep.py`
  - `src/diffusers/schedulers/scheduling_deis_multistep.py`
- Treść zmiany:
  - **zwykłe (VP) schedulery:** tablica dostaje odwrotną kolejność (`.flip(0)`), żeby `sigmas[i]` pasowało do `timesteps[i]`, które idą od 999 do 0;
  - **tryb flow (`use_flow_sigmas`, np. SANA):** zamiast złego formatu VP jest `sigma = t / 1000`, czyli dokładnie to, co daje `set_timesteps`.
- Do tego 3 nowe testy w `tests/schedulers/test_scheduler_dpm_multi.py`.
- Razem 5 plików, +104 i −4 linie. Plik z różnicami: `tools/diffusers_upstream/fix.patch`.
- `add_noise`, `set_timesteps` i `step` pozostają bez zmian, więc zwykłe generowanie działa identycznie jak dotąd.

## Czy to na pewno błąd i czy poprawka działa?

**Wersja:** sprawdzone na najnowszym wydaniu z PyPI (**0.41.0**) i na najnowszym `main` (commit `916a579` z 13.10.2026). Od wcześniejszego klonu doszły tylko 2 commity, które nie dotykają schedulerów.

**Test na prawdziwych modelach.** Model dostaje obraz zaszumiony przez scheduler i ma odtworzyć czysty obraz. Mniejszy błąd jest lepszy. Wzorcem jest poprawny poziom szumu.

Mały model VP (`google/ddpm-cifar10-32`); wzorzec to `DDPMScheduler`, prosty scheduler, który robi ten krok poprawnie:

| t | wzorzec | 4 schedulery bez poprawki | 4 schedulery z poprawką |
|---:|---:|---:|---:|
| 100 | 0,045 | **0,590** | 0,045 |
| 300 | 0,115 | **0,294** | 0,115 |
| 500 | 0,191 | 0,190 | 0,191 |

**SANA 1.6B, wasz model** (Kaggle, GPU T4, float32, skrypt `repro_sana.py`); wzorzec to wzór (1 − t/1000)·x0 + (t/1000)·ε:

| t | wzorzec | diffusers bez poprawki | diffusers z poprawką |
|---:|---:|---:|---:|
| 200 | 0,180 | **36,6** | 0,180 |
| 500 | 0,379 | **4,62** | 0,379 |
| 800 | 0,624 | 0,601 | 0,624 |

**SANA 600M 512 px** (ta sama rodzina, float32, CPU laptopa):

| t | wzorzec | diffusers bez poprawki | diffusers z poprawką |
|---:|---:|---:|---:|
| 200 | 0,228 | **45,5** | 0,228 |
| 500 | 0,383 | **5,76** | 0,383 |
| 800 | 0,590 | 0,556 | 0,590 |

Z poprawką wyniki są identyczne ze wzorcem. Bez niej model przy małym t dostaje wejście, z którego nie da się odtworzyć obrazu.

**Testy biblioteki:** na najnowszym `main` z poprawką przechodzi 276 testów schedulerów. Bez poprawki failują tylko 3 nowe testy, a 270 istniejących przechodzi, więc niczego nie psuję.

**Ograniczenie:** dwa inne schedulery (SASolver i DPMSolverMultistepInverse w trybie flow) mają podobny problem i poprawka ich nie obejmuje. Jest to opisane w zgłoszeniu.

## Dlaczego nikt tego nie zauważył, skoro to popularna biblioteka?

- Każdy standardowy pipeline (generowanie, img2img, inpainting) najpierw wywołuje `set_timesteps`, a dopiero potem `add_noise`. W tej kolejności wszystko działa poprawnie.
- Błąd wychodzi tylko przy `add_noise` wywołanym od razu po wczytaniu schedulera. Robią tak skrypty treningowe i badania typu sondowanie, takie jak wasze w ZZSN.
- Oficjalne skrypty treningowe używają innych schedulerów (`DDPMScheduler`, `FlowMatchEulerDiscreteScheduler`).
- Tablica `sigmas` w `__init__` pojawiła się w PR #6006 (2023), żeby ta kolejność wywołań przestała kończyć się wyjątkiem. Wyjątek zniknął, ale nikt nie sprawdził, czy wartości są dobre.

## Czy ktoś to już zgłosił albo naprawiał?

Przeszukałem issue i PR w diffusers (otwarte i zamknięte) przez API GitHuba, kilkunastoma zapytaniami.

- Ten sam obszar kodu, ale inny problem:
  - #5158: zamknięte, zmieniło `add_noise`, żeby czytało `sigmas`;
  - #5981: zamknięte, wyjątek „brak atrybutu sigmas”;
  - #6006: scalone, dodało `sigmas` w `__init__` i spowodowało ten błąd.
- Zgłoszenia o odwróconej kolejności albo złym poziomie szumu nie znalazłem. Nie znalazłem też otwartego PR, który zmienia te linie.
- Inne otwarte PR przy tych schedulerach dotyczą innych rzeczy, np. #14356, #13450 i #14572.
- Zastrzeżenie: to przeszukanie wyszukiwarką GitHuba, a nie ręczne przeczytanie wszystkich ok. 15 000 zgłoszeń. Przed wysłaniem warto jeszcze raz sprawdzić na stronie.

Skrypty tych testów: `tools/diffusers_upstream/functional/`.

**Stan (13.10.2026):** zgłoszenie jest wysłane: https://github.com/huggingface/diffusers/issues/15021. Gotowy tekst PR: `PR.md`.
