# Notatki czytelnika dokumentacji (slepy odczyt)

Kolejnosc czytania:

1. README.md
2. data/examples/README.md
3. reports/experiments/rq1.md
4. docs/STAN_PRAC.md
5. docs/audit/labels_and_arrows.md
6. docs/PODSUMOWANIE.md
7. docs/design_proposal.md
8. docs/literature.md
9. docs/worklog.md
10. docs/usage.md
11. docs/KAGGLE_RUNBOOK.md
12. docs/KAGGLE_DATA.md

---

## 1. README.md

**1. Cel i odbiorca.** Wejscie do repozytorium: pytanie badawcze ("Does Diffusion Learn 3D?", czy model tekst->obraz ma w srodku 3D, testowane na kierunku swiatla), tabela wynikow, szybki start, struktura repo. Jasne od pierwszych zdan, dla kogo: ktos z ML. Pierwsze trzy zdania sa bardzo dobre.

**2. Co zrozumialem.** Renderuja proste obiekty (CLEVR, Blender) ze znanym kierunkiem swiatla. Podaja obrazy do SANA-1.6B. Na zamrozonych aktywacjach transformera trenuja sonde liniowa (PCA + Ridge). Sonda przewiduje azymut i elewacje swiatla. Uczy sie na kulach i walcach, a test jest na szescianach (generalizacja miedzy ksztaltami). Pierwszy wynik z czerwca 2026 okazal sie zle zinterpretowany (audyt). Odblask na podlodze dawal pikselom tyle samo informacji co modelowi. Blad szumu (`add_noise` przed `set_timesteps`) ukrywal informacje w modelu. Po poprawkach: SANA-1.6B ma kierunek swiatla "daleko poza pikselami" (MAE azymutu na szescianach 4.8 st. vs 12.8 st. dla pikseli na v1; 7.6 vs 20.1 na v2r). Dodatkowo: kod swiatla nie zalezy od pozycji obiektu, przesuniecie aktywacji wzdluz wzorca swiatla obraca cieniowanie o 60-65 st., ale nie rysuje nowego cienia rzucanego.

**3. Pojecia bez wyjasnienia.**
- "SANA-1.6B", "DC-AE latent" (README: "We add noise to the DC-AE latent of each image") - nigdzie nie rozwiniete; czytelnik znajacy dyfuzje sie domysli, ze to autoenkoder SANA, ale nie wiadomo.
- "Best pixel grid" (naglowek kolumny tabeli), "grid" w "The val split selects the block, the timestep and the grid of each source" - co to jest "grid" (siatka usredniajaca 8x8? rozdzielczosc?) nie jest powiedziane. Pozniej "(8x8 grid)".
- "v1", "v2r" ("v2r (one sun lamp, random object position)") - nazwy wersji danych; "r" nie wyjasnione (random position? ale to w nawiasie). Czy istnieje v2 (bez r)? Nie wiadomo.
- "flow noise", "the original noise rule", "a noise pitfall (`add_noise` before `set_timesteps`)" - bez wyjasnienia, co robi flow noise w SANA (model przeplywu) i na czym polegal blad. Pojedyncza linia, ale bez odsylacza.
- "The floor highlight" - jaki odblask? "cast shadow" vs "shading" - rozroznienie zakladane.
- "The code does not depend on the position" - slowo "code" w znaczeniu "reprezentacja swiatla w aktywacjach"; dla czytelnika to myli sie z "kodem programu". Pojawia sie tez w "The model uses the code to light the object".
- "A shift along the activation pattern of the light (2% of the token norm)" - jaki "pattern", jak liczony, "token norm"? Brak.
- "turns the shading by 60 to 65 degrees towards a target of 90" - co to jest target 90 st.?
- "block 10, t = 200" - indeks bloku transformera i timestep; skala t (0-1000?) niepodana.
- "make cues", "dd3d-v1-images", `dd3d ... audit-arrows` - bez objasnienia, co to "cues".
- "key lamp", "sun lamp" - lampa kluczowa vs slonce; roznica niewyjasniona.
- "Athena supercomputer (PLGrid)" - kontekst infrastrukturalny.
- "MAE" - w naglowku "Azimuth MAE" - zakladam: srednii blad bezwzgledny w stopniach (kolowy?). Nie powiedziane czy blad jest kolowy.

**4. Miejsca niejasne / zalozenie wiedzy autora.**
- "The val split selects the block, the timestep and the grid of each source." - kto jest "source" (kolumny tabeli), jak dobierany jest podzial train/val/test; czy test = szesciany to jedyny test?
- Tabela: puste komorki w wierszach 2-5 bez objasnienia (nie zmierzono? nie dotyczy?).
- "v2r without the cast shadow" / "with only the cast shadow" - jak sie usuwa cien (re-render? maska?), brak.
- "The first result was right for the wrong reasons." - pierwszy wynik "right" ale "does not show what it claimed" w statusie. Sformulowanie mylace: czy pierwszy wynik byl poprawny czy nie? (Status mowi "does not show what it claimed", pkt 1 mowi "right for wrong reasons".)
- Pkt 2: "A probe on the mean of all tokens, fitted on centred objects, gives 9.0 (17.6 on cubes) on objects at random positions" - to inna sonda niz w tabeli (7.6 na v2r) - latwo pomylic. Jest 9.0 vs 7.6 vs 17.6; nie wiadomo, co jest wartoscia "glowna".

**5. Sprzecznosci / niespojnosci wewnatrz README.**
- Tabela: bold dla SANA w wierszu v2r (7.6), ale SDXL ma 7.3 (nizej). Bold sugeruje, ze SANA wygrywa, a w tym wierszu SDXL jest lepszy lub rowny. Wniosek "SANA carries the light direction far beyond the pixels" jest ok wzgledem pikseli, ale dla SD1.5/SDXL tez podobnie - nie jest to unikalna wlasnosc SANA. Tekst tego nie mowi.
- Status: "the answer is now clearer" - ale jaki dokladnie jest wniosek? Zdanie "it uses this code to light the object" vs pkt 4 "does not draw a new cast shadow" - wnioski sa nuansowane, ale w jednym miejscu.
- Quick start: "make audit  # label and arrow audit of all 3000 scenes" a w Method "1000 images for each shape" - 3 ksztalty -> 3000; to dotyczy v1; dla v2r liczba obrazow niepodana tutaj.
- Sekcja Method opisuje tylko v1 ("random azimuth ... on a sphere around the scene centre", "one purple rubber object"), a tabela ma v2r z "one sun lamp, random object position" - opis metody nie obejmuje v2r ani interwencji (pkt 3-4) ani modeli SD1.5/SDXL.

**6. Nieaktualne.** Brak oczywistych, ale: "The old figures below ... first study (block 10, t = 200)" - stare rysunki, ktore wg statusu pokazuja wynik uznany za bledny; ich obecnosc w README (z podpisem "old") moze mylic. Nie widac w tekscie, ktory z nich to "good", a ktory "bad" (nazwy plikow readme_good / readme_bad sugeruja dobre/zle przypadki, ale podpis ich nie opisuje).

**7. Odeslania do nieistniejacych tu plikow.** `docs/article/main.tex` (artykul) - niedostepny, a status wskazuje go jako zrodlo opisu; strata: duza (pelny opis metody i wynikow). `tools/viewer/` (interaktywny viewer) - niedostepny, strata mala dla czytelnika dokumentacji. `docs/figures/readme_good.png`, `readme_bad.png` - sprawdze, czy sa w folderze. `Makefile`, kod `src/dd3d` - niedostepne celowo.

**8. Jezyk.** Angielski, zwiezly, czytelny, krotkie zdania; miejscami zbyt skrotowy ("Best pixel grid", "The code"). README jest po angielsku, a odsylacze do statusu "(Polish)" - uczciwie zaznaczone. Autorzy podani z imienia i nazwiska na koncu.

### 1b. Obrazy z README: docs/figures/readme_good.png i readme_bad.png (sa w folderze)

- Oba obrazy pokazuja szescian na szarej podlodze z dwiema strzalkami: zielona (etykieta "label (key lamp shadow)") i czerwona (predykcja). Tytul: "label az=56.4 el=41.3 | pred az=57.4 el=41.6, error az=1.0 el=0.3" (dobry) oraz "label az=109.4 el=59.2 | pred az=224.6 el=59.5, error az=115.2 el=0.3" (zly).
- Rozumiem idee (strzalka wskazuje kierunek cienia, wiec kierunek "z lampy" odwrocony). Nie rozumiem: czy "label" to azymut swiatla czy azymut cienia (legenda mowi "key lamp shadow"), jak 2D strzalka jest liczona z azymutu i elewacji ("exact camera" - w README, bez wyjasnienia).
- Uwaga: blad elewacji jest 0.3 st. w obu przypadkach, nawet gdy azymut myli sie o 115 st. To dziwne i niczym niewyjasnione. Moze elewacja jest latwa. Nie wiadomo tez, czy to przypadki wybrane losowo, czy najlepszy i najgorszy (nazwy plikow sugeruja wybor reczny).
- README mowi "The old figures below" - ale nie mowi, co tu jest dobre/zle i dlaczego pokazano stare wyniki uznane za bledne.

---

## 2. data/examples/README.md (+ overview.jpg)

**1. Cel i odbiorca.** Katalog wersji danych: tabela 8 wersji (v1, v2, v2ns, v2pos, v2r, v2rns, v2rso, v2rot) i arkusz z przykladami (po 3 obrazy na wersje). Dla czytelnika, ktory chce zrozumiec nazwy z tabeli README. Cel jest jasny z pierwszego zdania. Ten dokument powinien byc linkowany z samej tabeli w README (jest link, dobrze).

**2. Co zrozumialem.** Wszystkie wersje to sceny CLEVR z jednym fioletowym gumowym obiektem (kula, walec, szescian). Wersja v1 to dane oryginalne (z projektu "ZZSN"). Lampa powierzchniowa krazy wokol sceny. W tle jest jeszcze lampa sloneczna. Wersja v2 ma jedna lampe sloneczna (tylko kierunek). Ma tez plaska podloge i obiekt na srodku. Wersja v2ns to v2 bez cienia. Wersja v2r to v2 z obiektem na losowej pozycji. Wersja v2rns to v2r bez cienia. W wersji v2rso kamera nie widzi obiektu. Zostaje tylko cien. Wersja v2pos to maly zbior testowy (300). Ma losowa pozycje i wlasne sceny. Wersja v2rot to 48 par z azymutem swiatla o 90 st. wiekszym. Sluzy do testu "patch". Numery scen sa wspolne, wiec wiersze arkusza roznia sie jedna wlasciwoscia (wedlug tekstu).

**3. Pojecia bez wyjasnienia.**
- "ZZSN project" (tabela i obraz: "original data (ZZSN)") - nieznany skrot (domyslnie przedmiot/projekt wczesniejszy). Nigdzie w README i tu nie rozwiniety.
- "area lamp", "sun lamp of the base scene" - nie wiadomo, co to "base scene" (domyslnie CLEVR base_scene.blend).
- "ns", "so", "rot", "pos" w nazwach: "ns" = no shadow, "so" = shadow only, "r" = random, "rot" = rotated. Wywnioskowane z opisow, nie zdefiniowane wprost jako skroty.
- "Pairs for the patch test" - "patch test" nie wyjasniony tu (w README byl "A patch of the object tokens from a scene with another light").
- "same scene numbers" - co to jest numer sceny, jak jest powiazany z azymutem/elewacja (ten sam azymut w v1 i v2?). Z arkusza wynika, ze ten sam kierunek swiatla (cien po prawej na v2/v2r/v2ns), ale dla v1 nie wiadomo.

**4. Miejsca niejasne.**
- "Thus the rows of the sheet for v2, v2ns, v2r, v2rns, v2rso and v2rot differ from the v2 row in one property only." Zdanie jest nieprawdziwe wzgledem tabeli. Wersja v2 jest na liscie wierszy, ktore roznia sie od v2 (pomylka). Wersja v2rns roznie sie od v2 dwiema wlasciwosciami (pozycja i brak cienia). Wersja v2rso tez (pozycja i kamera). Fraza "one property" pasuje do pary rodzic-dziecko ("As v2r, but ..."), a nie do v2.
- v2rot: "As v2, but the light azimuth is the v2 azimuth + 90 deg". Ale skad wiem, ze elewacja i ksztalt/pozycja sa takie same? Domyslnie tak, bo "as v2".
- v2pos wyglada na obrazie jakby mial inne tlo (kremowe, jasniejsze, rozne odcienie miedzy kula a szescianem) - nie wyjasnione ("own scenes"). Nie wiadomo, czy to cecha danych, czy artefakt.
- v2rso na arkuszu: zamiast "samego cienia" widac czarna plame (czarny elipsoid / czarny szescian) - czyli obiekt jest widoczny jako czarna dziura (maska). Opis "the camera does not see the object" nie mowi, ze obiekt jest czarny, a nie niewidoczny. Moze to wazne (sonda moze uzyc czarnej plamy).
- Tabela: v1 ma "Labels and splits" w `data/master_labels.csv`, `data/splits/`, a v2pos i v2rot maja inny typ (v2rot: `configs/kaggle/patch_pairs.txt` - to nie etykiety, tylko lista par).
- v2rot ma 48 obrazow, na arkuszu widac 2 przyklady (kula i walec) - zgodne ze "three images of each version"? Nie: 2, nie 3 (arkusz). Tekst mowi "three images of each data set version" - dla v2rot tylko 2 widac.
- v2pos: 300 obrazow, "Test only" - czy to ten sam co v2r testowy? Jaka jest roznica miedzy v2pos i v2r? Obie "random position". Nie jest powiedziane, po co obie (z dokumentu: v2pos = zbior ewaluacyjny z innych scen, w v2r test na wspolnych).

**5. Sprzecznosci.** Z README: README mowi, ze v2r = "one sun lamp, random object position" - zgodne. README opisuje "v1 ... flow noise" - tu "v1 ... area lamp ... and the sun lamp of the base scene is on" - zgodne. README mowilo o 1000 obrazow na ksztalt (3000), tu 3000 - zgodne. Sprzecznosc wewnetrzna: patrz zdanie "differ from the v2 row in one property only" (powyzej). W README "key lamp has a random azimuth ... on a sphere around the scene centre" (v1) vs tu "area lamp turns around the scene centre" - zgodne, ale "key lamp" vs "area lamp" - dwa slowa na te sama lampe?

**6. Nieaktualne.** Nie widac. "The script `scripts/data/make_examples.py`" - ok.

**7. Brakujace pliki.** `scripts/data/make_examples.py`, `data/v2*/` (etykiety), `configs/kaggle/patch_pairs.txt`, `docs/KAGGLE_DATA.md` (jest na liscie do czytania). Strata: mala; liczy sie opis kolumn etykiet (czego brakuje - tu nie ma w ogole schematu kolumn CSV).

**8. Jezyk.** Krotki, przejrzysty. Tabela dobra. Arkusz (overview.jpg) bardzo pomaga: widac roznice miedzy wersjami od razu. Zdanie "The new versions use the same scene numbers" jest skrotowe.

---

## 3. reports/experiments/rq1.md

**1. Cel i odbiorca.** Plik to sama tabela (22 wiersze) bez tytulu, bez zdania wstepu i bez legendy. README nazywa ja "full table with 95% intervals". Z samego poczatku nie wynika nic poza tym, co mowia naglowki kolumn. Nazwa "rq1" nie jest nigdzie wyjasniona w tym pliku (domyslnie "research question 1").

**2. Co zrozumialem.** Tabela ma wersje danych (v1, v2, v2r, v2r bez cienia, v2r tylko cien). Ma tez zrodla cech (piksele, DC-AE, SANA z losowymi wagami, SANA, SD 1.5, SDXL). Dla kazdej pary podano trzy rzeczy. Pierwsza to wybrana konfiguracja (blok, t, siatka). Druga to blad MAE azymutu na tescie. Trzecia to blad na szescianach i roznica wzgledem pikseli, w nawiasach przedzialy. Wartosc "Cubes az MAE" zgadza sie z liczbami w README. Roznica do pikseli jest ujemna dla modeli (lepiej niz piksele). Dla SANA wynosi -8.0 na v1 i -12.5 na v2r.

**3. Pojecia bez wyjasnienia.**
- "Block, t, grid" - wartosci "7, 100, 4", "up2, 100, 8", "mid, 100, 4". Brak skali t, brak informacji, ze "up2" i "mid" to bloki U-Netu, brak wyjasnienia, ze "grid" to rozmiar usrednionej siatki tokenow.
- "16 px", "64 px", "8 px" w kolumnie "Block, t, grid" dla pikseli - ta sama kolumna ma tu inna jednostke (rozmiar siatki pikseli w px, nie w komorkach).
- "Test az MAE" vs "Cubes az MAE" - nie powiedziano, co to jest "test" (domyslnie: held-out kule i walce z tych samych scen?). Nie powiedziano, czy MAE jest kolowy ani w jakich jednostkach (domyslnie stopnie).
- "Cubes: diff. to pixels" - roznica MAE wzgledem wiersza pikseli; "--" w pierwszym wierszu.
- "v2r, no shadow" i "v2r, only shadow" - w data/examples/README te wersje nazywaja sie v2rns i v2rso. Dwie nazwy tej samej rzeczy.
- "original noise" - bez objasnienia (zob. README).
- Przedzialy w nawiasach kwadratowych - README mowi, ze to 95% bootstrap, ten plik nie.

**4. Miejsca niejasne.**
- Brak legendy: nie wiadomo, czy przedzial jest bootstrapem po obrazach, po scenach, czy po seedach. Nie wiadomo, ile jest probek testowych.
- Brakujace wiersze (v2: brak SANA z losowymi wagami i SDXL; v2r bez cienia: brak SD i SDXL) nie sa wyjasnione.
- Wiersz "SANA-1.6B, original noise | 18, 200, 4" ma blok 18 i t=200. README (podpis rysunkow) mowi o pierwszym badaniu "block 10, t = 200". To inny blok. Czytelnik nie wie, czy to ten sam eksperyment z ponownym doborem na val.
- v2 pixels: test 2.5 [2.2, 2.9], cubes 13.7. Ogromna luka (5x) miedzy testem a szescianami nie jest komentowana tutaj. Ten wynik sam w sobie jest interesujacy (piksele nie generalizuja miedzy ksztaltami).
- Piksele v2r bez cienia: "24.6 [20.7, 28.8]" i piksele v2r tylko cien: "24.6 [20.7, 29.0]". Prawie identyczne wartosci i przedzialy w dwoch roznych wierszach. Wyglada na przypadek albo kopie. Z samej tabeli nie da sie tego sprawdzic.
- Brak w tej tabeli wynikow, ktore README podaje jako "other results" (sonda na sredniej tokenow 9.0/17.6, piksele 71 st., interwencje). Plik nie jest wiec pelnym zrodlem wynikow z README.

**5. Sprzecznosci z README.** Wszystkie liczby kolumny "Cubes az MAE" (12.8, 8.9, 11.9, 4.8, 6.9, 6.0, 14.5, 20.1, 19.0, 15.9, 7.6, 11.1, 7.3, 23.3, 30.7, 9.9, 16.9, 17.8, 7.9) sa zgodne z tabela README. Sprawdzilem tez roznice: wszystkie zgadzaja sie z odejmowaniem z dokladnoscia do 0.1 (rozbieznosci +-0.1 z zaokraglen). Jedyna rzecz: SDXL (7.3, roznica -12.9) jest na v2r tak dobry jak SANA (7.6, -12.5), a przedzialy sie pokrywaja. README pogrubia tylko SANA. Nazewnictwo wersji rozni sie od data/examples/README (patrz wyzej).

**6. Nieaktualne.** Nie wiadomo. Brak daty i wersji.

**7. Brakujace pliki.** Brak odesłan. Niedostepne zrodlo liczb (skrypt analizy, CSV z wynikami) - plik nie mowi, skad pochodzi tabela. Strata: duza dla odtwarzalnosci, bo nie ma zadnej komendy ani sciezki.

**8. Jezyk.** Same cyfry i skroty. Dla kogos, kto zna projekt, wystarcza. Dla nowego czytelnika tabela jest nieczytelna bez README i bez legendy. Brak jednostek (st.).

---

## 4. docs/STAN_PRAC.md (po polsku, 16.10.2026)

Obrazy z tego dokumentu, ktore otworzylem, sa w folderze. Lista:

- data/examples/overview.jpg
- docs/article/figures/scan.png
- reports/experiments/causal_b05/b05t100/examples.png
- reports/experiments/causal_multi/all/examples.png
- docs/article/figures/patch.png

**1. Cel i odbiorca.** Dokument jest "krotkim przegladem dla prowadzacego". Odpowiada na pytania: co badamy, co wiemy, co pokazaly eksperymenty, co dalej. Z poczatku jasne, dla kogo jest. Nie jest jednak krotki: to 175 linii, 8 sekcji, 5 tabel i 4 diagramy. To glowne rozbudowane streszczenie projektu. Wrazenie: to najlepszy dokument do zrozumienia calosci.

**2. Co zrozumialem.** Pytanie ma dwie czesci. Czy aktywacje SANA-1.6B niosa kierunek swiatla, ktorego nie ma wprost w pikselach? Czy model tej informacji uzywa? Sonda liniowa (PCA 256 i Ridge) czyta sin, cos azymutu i elewacje. Blad azymutu jest kolowy. Obraz jest zaszumiony regula przeplywu, x_t = (1 - sigma) x0 + sigma eps. Audyt starego projektu ZZSN pokazal cztery rzeczy. Stary wynik nie bil pikseli (14,0 vs 14,1). Szum byl bledny (blad w diffusers). Etykieta byla pozycja lampy, nie swiatlem przy obiekcie. Sonde niosla plama swiatla na podlodze. Nowe dane (v2, v2r, v2rns, v2rso, v2rot) usuwaja te skroty. Na v2r SANA bije piksele o 12,5 st. Podobnie robia SD 1.5 i SDXL. Test przyczynowy: przesuniecie aktywacji wzdluz "wzorca" zmienia cieniowanie obiektu, a losowe przesuniecie nie. Duze przesuniecie nie rysuje nowego cienia. Podmiana tokenow obiektu miedzy scenami przenosi cieniowanie i okolo jednej trzeciej cienia.

**3. Pojecia bez wyjasnienia.**
- "wzorzec" / "+δ wzorzec" (sekcja 5, mermaid i tabela) - nie jest zdefiniowany, jak liczony jest wzorzec (domyslnie roznica srednich aktywacji?). To kluczowe pojecie testu przyczynowego, a nie ma definicji ani tu, ani w README.
- "najkrótsza zmiana odczytu sondy" - nie wiadomo, o jaka procedure chodzi (minimalna zmiana w normie L2, ktora zmienia odczyt sondy?). Na obrazie przyklady maja kolumne "probe_exact".
- "LEACE" - akronim bez objasnienia (metoda wymazywania pojec). Student XAI moze znac, ale nie ma ani opisu, ani cytatu w tym pliku.
- "sigma0 = 0,7", "20 krokow Eulera" - skad 0,7 i jak sie ma do t = 100?
- "Blok 5 z cechami z t = 100" - nie jasne, jak cechy z t = 100 wchodza do interwencji w generowaniu z sigma0 = 0,7.
- "ostry cien zostaje" - miara zdefiniowana slownie (czesc pikseli ostrego cienia, ktora dalej jest cieniem). Definicja jest. Brak jednak wzoru i progu "ostrosci".
- "kierunek ciemnej plamy" - geometria, "bez uczenia", ale nie powiedziano, jak sie liczy.
- "sonda pikselowa" w tescie przyczynowym - na jakich danych uczona? Nie wiadomo.
- "łatanie", "przeszczep", "podmiana" - trzy slowa na to samo (po angielsku "patch").
- "Haufe i in. (2014)" - brak pozycji w tym pliku (jest pewnie w literature.md).
- "ZZSN" - znowu bez rozwiniecia ("audyt starego projektu (ZZSN)").
- "issue #15021" - do jakiego repo (diffusers - z kontekstu). Dodatkowo brak linku.
- "TMLR" - czasopismo, znane w srodowisku.
- "v2rot", "pary do łatania" - powiazanie z sekcja 6 jest, ale 48 par to bardzo mala proba, nie skomentowana.

**4. Miejsca niejasne.**
- Diagram sekcji 1: "Render Blendera 512×512" prowadzi do "latent 32×32×32" i "1024 tokeny". Dla DC-AE o kompresji 32 obraz 512 px daje siatke 16×16 (256 tokenow). Dokument nie wspomina o skalowaniu do 1024 px. To moze byc pominiety krok. Nie jestem pewien, bo nie widze kodu.
- Tabela audytu: "Stary wynik nie bil pikseli | sześciany: SANA 14,0°, ta sama sonda na pikselach 14,1°". Liczba 14,5 w rq1/README (stary szum, ponowny dobor na val) jest inna niz 14,0. Dokument nie tlumaczy roznicy miedzy 14,0 (stary kod) a 14,5 (nowy kod, stary szum).
- Sekcja 5, tabela: "wzorzec ×3 | +121° / +121° / +120°" przy celu +90 st. Odczyt sondy przekracza cel o 30 st. Brak komentarza. Przy "wszystkie 20 blokow ×0,25" tez +120 st. i obraz pokazuje prostokaty. Dokument sam przyznaje, ze to artefakty. Znaczy to, ze wartosc "sonda pikselowa" przy duzych dawkach nie mierzy swiatla, tylko artefakty.
- Wynik "Wynik jest taki sam w każdym bloku, bo podmiana tokenu zastępuje cały jego stan." Wykres patch.png pokazuje plaskie linie dla blokow 0-18. Wyjasnienie jest jednym zdaniem i nie jest uzasadnione. Plaski wynik dla bloku 0 i 18 jest dla mnie zaskakujacy.
- "Cały cień przenosi dopiero podmiana wszystkich tokenów." Tabela mowi: wszystkie tokeny w bloku 10 daja 0,84, a pelny przebieg zrodla 1,00. Zdanie jest wiec lekko przesadzone.
- Kontrola losowa ("losowy wektor tej samej wielkości") jest slaba dla pytania o specyficznosc. Losowy kierunek w 2240 wymiarach jest prawie ortogonalny do wszystkiego. Lepsza kontrola to kierunek innej cechy (np. koloru). To uwaga merytoryczna czytelnika, a nie blad dokumentu.
- Wniosek 4 sekcji 4 "Kod nie zależy od pozycji" i tabela koncowa "Czy ten kod zależy od pozycji obiektu w kadrze? | Nie." Dane: sonda uczona na v2 daje 17,6 st. na szescianach z losowa pozycja, a sonda uczona na v2r daje 7,6 st. Piksele daja 71 st., wiec rzad wielkosci sie zgadza. Prosta odpowiedz "Nie" jest jednak zbyt mocna. Powinno byc "w duzej mierze nie".
- Wniosek 5 (v2rns i v2rso, "ok. 30-33°, piksele ok. 73-75°") - dotyczy przenoszenia sondy z pelnych scen. Tych liczb nie ma w README ani w rq1.
- Wniosek 4: "Średnia wszystkich tokenów (siatka 1×1) daje 5,9° na v2r." Nie jest powiedziane, czy to test czy szesciany. W rq1 najlepsza konfiguracja v2r to siatka 2, test 5,4.
- Rysunek scan.png (v1 i v2r): pozioma linia "pixels" lezy przy okolo 14,3 (v1) i okolo 25,8 (v2r). Tabela mowi 12,8 (v1) i 20,1 (v2r) dla "najlepszej siatki pikseli". Dla v2 (13,7) linia zgadza sie. Wiec w dwoch panelach linia pikseli nie jest najlepsza siatka, a podpis tego nie mowi. To wyglada na niespojnosc miedzy rysunkiem i tabela.
- W scan.png legenda zaslania czesc wykresu v2r (krzywe t=600 i t=800 wychodza poza os). Mala wada grafiki.
- examples.png (blok 5, t = 100): kolumny maja nazwy z kodu (base, probe_exact, pattern_x1, pattern_x2, pattern_x3, random_pattern, erase). Tabela w tekscie uzywa innych nazw. Brak mapowania. Dla czterech przykladow przesuniecie pattern_x1 wynosi okolo +59, +27, +25, +15 st., a w tabeli naglowkowej jest "+60 do +65". Tekst wyjasnia to tylko posrednio: dla bloku 5 dawka x1 daje +30 st.. README podaje tylko 60-65 st. Czytelnik README nie wie, ze liczba zalezy od bloku (4/10/18 vs 5) i dawki.
- "Wszystkie zbiory sa gotowe" - a w sekcji 8 nie ma zadnej listy rzeczy niegotowych oprocz przegladu artykulu. To wyglada dobrze, ale wymaga zaufania.

**5. Sprzecznosci z README, data/examples/README i rq1.**
- Wszystkie liczby tabeli w sekcji 4 zgadzaja sie z rq1. Porownalem komorka po komorce wszystkie kolumny i wersje. Przedzialy SANA tez sie zgadzaja ([4,6; 5,1], [7,0; 7,7], [7,3; 8,0], [9,4; 10,4], [7,6; 8,3]).
- Wniosek 1 (12,5 st., CI [11,1; 13,8]) zgadza sie z rq1 (-12.5 [-13.8, -11.1]).
- Liczby 9,0 / 17,6 / 71 st. i 14,5 / 4,8 / 12,8 zgadzaja sie z README.
- Nazwy wersji: tu v2rns i v2rso (jak w data/examples), a README i rq1 uzywaja opisu "no shadow", "only shadow". Nie sprzecznosc liczb, ale nazewnictwa.
- README (status): "The cast shadow follows only large changes." Sekcja 5 i README pkt 4: duze przesuniecie nie rysuje nowego cienia, ale kierunek ciemnej plamy przy x3 wynosi +109/+107/+92 st. Wiec geometria plamy idzie za celem przy x3, choc ostry cien nie jest narysowany. Dwa sformulowania ("follows" i "does not draw") brzmia jak sprzeczne, a jest to ta sama obserwacja widziana dwiema miarami.
- Kontrastuje README ("The first result was right for the wrong reasons") i tu "Stary wynik nie bil pikseli" ("Nie: piksele byly rownie dobre"). Tutaj stary wynik nie dowodzil niczego. W README "right" znaczy chyba "model rzeczywiscie ma informacje". Dwuznacznosc pozostaje.
- Sekcja 2 podaje "Etykieta to pozycja lampy, a nie swiatlo przy obiekcie (35 st. roznicy)". Rysunki w README maja legende "label (key lamp shadow)". Nie jest jasne, czy rysunki README pokazuja ta stara etykiete.
- Status w README mowi "draft" artykulu, tu "wszystkie wyniki sa wpisane". Zgodne.

**6. Nieaktualne lub tymczasowe.** "W tym tygodniu GPU: ok. 27 z 30 h." - informacja o limicie Kaggle, ktora szybko sie starzeje. "Nowe (15.10)" i "(16.10.2026)" - daty. Sekcja 8 "Pytania do prowadzacego: zespół 2 osoby zamiast 3, kontynuacja projektu z ZZSN, zasady oceny" - to jest pytanie administracyjne, ktore po rozstrzygnieciu bedzie nieaktualne. W dokumencie projektu badawczego to mylace.

**7. Brakujace pliki.** `docs/article/main.tex` i PDF (`docs/article/build/main.pdf`) - tu najwieksza strata: pelne definicje (wzorzec, miary, kontrole) sa tam. `tools/viewer/` (viewer). "lista kontrolna dla badan sondami" i "poprawka do biblioteki diffusers" - bez linku (diffusers_upstream jest w README). Obrazy w docs/article/figures i reports/experiments/... sa na szczescie dostepne.

**8. Jezyk.** Czytelny polski, dobre krotkie zdania. Mieszanka pojec angielskich ("hook", "patch" w nazwach plikow). Bledy dopasowania: kilka wyrazen nie jest przetlumaczonych jednolicie (lat anie, przeszczep, podmiana). Format dziesietny z przecinkiem (4,8) w tym pliku, a README i rq1 uzywaja kropki (4.8) - latwo przeoczyc, ale OK. Mermaid i tabele pomagaja.

---

## 5. docs/audit/labels_and_arrows.md (2026-10-13, po angielsku)

Obrazy: docs/figures/v2_sample.png (otwarty, jest w folderze). Probowalem otworzyc reports/audit/shadow_audit.png i reports/audit/lighting_study.png. Nie ma ich w folderze. Inne obrazy z dokumentu (data/audit_renders/conv_*.png, docs/audit/figures/, reports/viz/) tez sa poza folderem.

**1. Cel i odbiorca.** Raport techniczny z audytu starego projektu (ZZSN): jak dane definiuja azymut i elewacje, czy kod dobrze liczy blad, czy strzalki na rysunkach sa poprawne. Tytul i trzy pytania na poczatku sa jasne. Odbiorca to recenzent lub autor, ktory zna Blendera. Czytelnik bez wiedzy o Blenderze (TRACK_TO, AREA lamp, hide_render) nadrabia z trudem.

**2. Co zrozumialem.** Autorzy skopiowali konfiguracje renderu CLEVR do testowalnego kodu. Zweryfikowali kopie (blad kamery 0,56 px, render zgodny z obrazem, IoU 0,98). Etykieta to polozenie lampy wokol poczatku ukladu, ale obiekt stoi do 2,8 jednostki dalej. Rzeczywisty kierunek swiatla przy obiekcie roznie sie srednio o 35 st. (mediana 27 st., maks. 180 st.). Plama swiatla na podlodze wskazuje wprost etykiete, wiec jest skrotem niezaleznym od ksztaltu. To tlumaczy, czemu stara sonda "generalizowala" na szesciany. Strzalki na starych rysunkach mialy sie zle: zla kamera, rzut ortogonalny, 18-19% strzalek poza obrazem. Strzalki pokazywaly cien, nie swiatlo. Test opcji oswietlenia pokazal, ze lampa SUN i plaska podloga usuwaja plame na podlodze. Dlatego powstalo v2.

**3. Pojecia bez wyjasnienia.**
- "ZZSN" (w pierwszym akapicie: "the ZZSN probe outputs", "the ZZSN report", "Figure 4.1 of the ZZSN report") - dalej bez rozwiniecia. Raport ZZSN nie jest czescia repo.
- "`data/final`" jako nazwa zbioru (Date/Scope) - README i data/examples/README nazywaja ten zbior "v1" (`data/master_labels.csv`). Mapowanie data/final = v1 nie jest podane.
- Nazwa sceny "SmoothCube_v2_new_000004" - "v2_new" w nazwie sceny z danych (domyslnie v1). To koliduje z nowym znaczeniem "v2".
- "IoU", "MD5", "TRACK_TO", "AREA", "SUN" - dla specjalistow.
- "Floor highlight", "Shadow depth", "Object contrast", "Shadow direction error" (tabela sekcji 8) - miary bez definicji. Podano tylko, ze to roznice luminancji w skali 0-255.
- "conv_az*.png", "conventions.json", "shadows.json" - nazwy plikow konwencji, nie opisane.
- "Pixel cues" (sekcja 5) - odesle do pliku, ktorego nie ma. Liczb 9,4 / 44,3 st. (z STAN_PRAC) tu nie ma.
- "elevation" - nie jest wyjasnione, co w sekcji 8 znaczy "a probe can read the elevation from the mean brightness". Czy to jest powod, dla ktorego wyniki dotycza tylko azymutu?

**4. Miejsca niejasne.**
- Zdanie "The README now says 'shadow direction'" opisuje historie zmiany, a nie stan. Czytelnik bez historii nie wie, co to "now". Z tym, ze README rzeczywiscie mowi "arrows of the cast shadow" (zgodne).
- Sekcja 6 daje 7 rekomendacji dla v2, ale dokument nie mowi, ktore z nich wdrozono. Zdanie na koncu ("make_v2_spec.py writes the render spec ... with these rules") sugeruje "wszystkie". Dla punktu 4 (jitter 0 albo zapis kamery) i 6 (rotacja w radianach) nie ma potwierdzenia.
- Sekcja 5 podaje, ze sonda ZZSN ma blad 14,0 st. dla etykiety i 35,5 st. dla kierunku swiatla przy obiekcie. Ten drugi wynik nie wystepuje w README ani w STAN_PRAC. Dodatkowo nie wiadomo, czy "35,5" pokazuje, ze stara sonda nie odczytuje kierunku, czy ze etykieta jest zla.
- Rozpietosc elewacji "8,4 do 88,8 st." przy obiekcie vs deklarowane "30-75 st." w README (to dotyczy etykiety, wiec bez sprzecznosci). README Method nie wspomina jednak, ze etykieta v1 nie jest kierunkiem swiatla przy obiekcie.
- Audyt nie wspomina bledu szumu (add_noise przed set_timesteps). A STAN_PRAC (sekcja 2) pokazuje go jako ustalenie audytu, a README nazywa go czescia audytu ("An audit found..."). Wiec "audyt" ma dwa znaczenia: ten plik (etykiety i strzalki) i szerszy (plus szum, plus kontrole pikseli). Szerszy opis jest tylko w STAN_PRAC (i pewnie w PODSUMOWANIE i worklog).
- Rysunki zakladane przez tekst (strzalki, test oswietlenia) nie sa dostepne w folderze. Dla L3 (kluczowe ustalenie) czytelnik nie widzi plamy.
- Tabela sekcji 8: "Shadow direction error" jest niezerowy nawet dla SUN i plaskiej podlogi (7,1 i 5,3 st.). Zrodlo bledu nie jest wyjasnione (pomiar? kamera?).
- v2_sample.png: 6 obrazow bez podpisu. Tla maja rozna jasnosc (zalezy od elewacji, co zgadza sie z tekstem), ale rysunek nie ma legendy ani opisu.

**5. Sprzecznosci z poprzednimi dokumentami.**
- "Mean 35.0 deg" zgadza sie z STAN_PRAC ("średnio 35°"). "14.0 deg" zgadza sie z STAN_PRAC (14,0). Rekomendacje dla v2 zgadzaja sie z opisem v2 w data/examples/README (jedna lampa sloneczna, plaska podloga).
- STAN_PRAC mowi "Etykieta to pozycja lampy" i mierzy "różnica średnio 35° w azymucie" - spojne.
- Elewacja: sekcja 8 pisze, ze przy SUN jasnosc podlogi zalezy tylko od elewacji (skrot dla elewacji). Wszystkie wyniki w rq1 i README to blad azymutu. Wyniki dla elewacji nie pojawily sie w zadnym przeczytanym dotad pliku.
- Blad strzalek: "5.2 deg on average" itp. Nic w README. Dobrze.
- README: "make audit # label and arrow audit of all 3000 scenes" odpowiada sekcji 7 tego pliku (`arrow_audit.py`, ok. 2 min). Brakuje w README `make` dla pozostalych krokow (render, shadow_audit).

**6. Nieaktualne.** Dokument jest dziennikiem audytu z 13.10. Pozniej (15.10) odkryto blad szumu, ktory zmienil interpretacje ("stary wynik nie bil pikseli" pozostaje, ale wniosek o sondzie sie zmienil). Plik nie ma uwagi "patrz STAN_PRAC po nowsze wyniki". Sekcja 5 ("The ZZSN probe follows the label") jest aktualna jako opis starego wyniku, ale nie zawiera pelnego wniosku.

**7. Brakujace pliki.** `reports/experiments/pixel_cues/results.md` (liczby plamy: strata srednia), `data/audit_renders/conv_*.png`, `docs/audit/figures/`, `reports/audit/shadow_audit.png`, `reports/audit/lighting_study.png` (obrazy), `reports/audit/arrow_audit.csv`, `reports/viz/`, kod skryptow audytu i `legacy/...`. Strata: sredia, bo tekst sam jest dokladny. Rysunki sa potrzebne, zeby uwierzyc w L3 i w konwencje azymutu.

**8. Jezyk.** Angielski prosty, krotkie zdania, ponumerowane ustalenia (L1-L7, A1-A6) - bardzo dobrze sie czyta i latwo cytowac. Gesty od nazw plikow i polecen. Mala niekonsekwencja: "dataset" i "dataset v2" (v2 to w tym pliku rekomendacja, w innych dokumentach gotowy zbior).

---

## 6. docs/PODSUMOWANIE.md (po polsku, "stan na 16.10.2026")

Obrazy z tego dokumentu otworzylem. Wszystkie sa w folderze docs/figures/:

- cues.png
- conventions.png
- lighting_study.png
- shadow_audit.png
- readme_good.png
- readme_bad.png
- v2_grid.png
- probe_maps_pixels.png
- v2_variants.png
- v2r_corners_rejected.png (w tekscie tylko nazwa, bez osadzenia)

Wczesniej w sekcji 5 audytu probowalem otworzyc reports/audit/shadow_audit.png i lighting_study.png. Nie ma ich, ale ich kopie sa w docs/figures/.

**1. Cel i odbiorca.** Dokument zbiera "wszystko, co zostalo sprawdzone i zrobione". Jest podzielony na 15 rozdzialow, kazdy odpowiada na jedno pytanie. Z pierwszego akapitu nie wynika, dla kogo jest. Glos tekstu pokazuje, ze dla zespolu autorow ("wasz model", "planowaliscie", "dopasowalem"). Prowadzacy dostaje STAN_PRAC. To powinno byc powiedziane wprost na poczatku.

**2. Co zrozumialem.** To dziennik calego przebiegu pracy od audytu (12-13.10) do wynikow z 15-16.10. Audyt pokazal, ze stary wynik ZZSN nie dowodzi wiedzy modelu o swietle. Piksele daja ten sam blad (14,1 st.), a sonde niesie plama swiatla na podlodze. Etykieta jest pozycja lampy, a nie swiatlem przy obiekcie (35 st. roznicy). W ekstrakcji byl blad szumu, ktory wynika z pulapki w bibliotece diffusers. Autorzy zglosili go jako issue i przygotowali poprawke. Zbudowano nowe dane (v2, v2r, wersje bez cienia i sam cien). Dokument opisuje tez sprzet (laptop za slaby, klaster wyczerpany, wszystko na Kaggle) i otwarte decyzje.

**3. Pojecia bez wyjasnienia.**
- "SIWY (XAI)" (Audyt, pkt 7) - pierwszy raz w tym dokumencie nazwa przedmiotu. W README jej nie ma. Kontekst kursu jest wiec widoczny dopiero tutaj.
- "RQ3" (sekcja 10: "Warianty do pytania „która wskazówka” (RQ3)") - pytania RQ1, RQ2 nigdzie tu nie zdefiniowano. Plik rq1.md nosi nazwe RQ1.
- "wzorzec" w sekcji 9 oznacza wartosc odniesienia (baseline). W STAN_PRAC "wzorzec" to kierunek w przestrzeni aktywacji. To dwa rozne znaczenia tego samego slowa w jednym projekcie.
- "wskazowka" (cue) - uzywane jako "cieniowanie" i "cien" bez definicji. Dopiero sekcja 10 tlumaczy, ze v2rns i v2rso daja jedna wskazowke.
- "etykieta globalna" vs "lokalna" (tabela sekcji 3) - krotkie wyjasnienie w nawiasie ("swiatlo przy obiekcie"). Wystarcza.
- "LEACE", "sterowanie" (sekcja 12), "PCA 256", "float16/float32/bfloat16" - znane specjalistom.
- "scheduler", "add_noise", "set_timesteps", "flow matching", "VP schedulery" - sekcja 9 wyjasnia je dobrze (to najlepiej wyjasniona czesc).
- "r ≈ 0,00" (sekcja 10) - korelacja pozycji z kierunkiem, ale nie powiedziano, jaka korelacja.
- "konto jankombajn", "kqueue" - szczegoly operacyjne.

**4. Miejsca niejasne.**
- Intro twierdzi: "Szczegoly techniczne (po angielsku) sa w plikach podanych na koncu rozdzialow, ale nie trzeba ich czytac". Czytelnik, ktory chce odtworzyc wyniki, musi je jednak czytac. Zdanie jest zbyt optymistyczne.
- Sekcja 2: trzy rozne liczby dla "piksele na szesciankach" w v1: 14,1 (64x64, sekcja 2), 14,3 (cala klatka, sekcja 3) i 12,8 (najlepsza siatka, README i rq1). Dodatkowo test: 11,5 (sekcja 2) i 11,8 (sekcja 3). Nie wyjasniono, ze to rozne protokoly (stary i nowy). Czytelnik widzi niespojnosc.
- Sekcja 5: "W 22 z 36 przypadkow o bledzie powyzej 45 st. predykcja jest blizej prawdziwego swiatla przy obiekcie niz etykiety". Liczba 36 nie ma kontekstu (z ilu?).
- Sekcja 9: "PR czeka na odpowiedz opiekunow" i "Gotowy tekst: tools/diffusers_upstream/PR.md". Nie wiadomo, czy PR zostal otwarty, czy tylko przygotowany. Pytanie "czy wystawiony PR daje ocene 5" sugeruje, ze jest wystawiony. Dwuznacznosc.
- Sekcja 9, arytmetyka: "przechodzi 276 testow ... Bez poprawki failuja tylko 3 nowe testy, a 270 istniejacych przechodzi". Jesli 276 zawiera 3 nowe, to istniejacych jest 273, a nie 270. Albo literowka, albo 3 testy sa pominiete.
- Sekcja 13: "Z poprawnym szumem norma wynosi 718". Sekcja 9 mowi, ze stare aktywacje dla t = 800 maja norme ok. 720. Obie liczby sa prawie rowne. Nie jest powiedziane, dla jakiego t jest 718. Czytelnik moze podejrzewac blad.
- Sekcja 10, tabela v2: wiersz "test na obiektach w losowych miejscach (300 obrazow) 69,0 st." Inne dokumenty (README, STAN_PRAC, sekcja 0 tego pliku) podaja dla pikseli 71 st. Dwie liczby (69,0 i 71) dla podobnego pomiaru. Nie ma wyjasnienia (inna siatka pikseli?).
- Sekcja 10, wniosek 1: "Na takim zbiorze zaden model nie pokaze przewagi nad pikselami". rq1 pokazuje inaczej dla v2: sześciany SANA 7,3 st. wobec 13,7 st. pikseli (roznica -6,3 [-6,9; -5,8]). Przewaga jest, tylko na szesciankach, a nie na tescie. Zdanie jest nieaktualne albo zbyt silne.
- Sekcja 10, wykres probe_maps_pixels.png: dla v2 tytul mowi "pixels 16x16, cubes az MAE 40.2 deg", a rq1 dla v2 daje 13,7 (siatka 64 px). Rzecz jest nieomowiona. Czytelnik moze pomyslec, ze to blad.
- Sekcja 12 (Teza artykulu): "sondy liniowe na modelu dyfuzyjnym latwo daja falszywy dowod wiedzy 3D ... Wynik negatywny tez jest tu wartosciowy". Sekcje 0 i README (16.10) mowia, ze po poprawkach wynik jest pozytywny. Teza nie zostala zaktualizowana.
- Sekcja 11 i 12 (stan prac): "Szkic artykulu ... brakuje wynikow z GPU", "[pending]" na czerwono, "Testy przyczynowe ... proba na GPU w kolejce", "Design proposal ... szkic do przejrzenia", "Dziennik ... pierwszy wpis". STAN_PRAC (16.10) mowi, ze wszystkie wyniki sa wpisane. Dokument ma date 16.10, a sekcje 11-14 pochodza z 13.10.
- Sekcja 15 "Mapa plikow" nie zawiera README, STAN_PRAC, rq1.md, usage/Kaggle, worklog ani data/examples. Mapa jest niepelna.
- Rysunek v2r_corners_rejected.png: cztery kafelki bez podpisow. Widac szesciany przy krawedziach kafelkow (dwa "stykaja sie"). Bez podpisu nie wiem, co jest "odrzucone" i dlaczego. Dokument wymienia plik tylko w nawiasie, bez opisu rysunku.
- Sekcja 6 pisze "wylaczenie" lampy "Area" i "wylaczenie nie dzialalo". Znaczy "wylaczenie lampy nie usuwa plamy". Skrot mysli.

**5. Sprzecznosci z poprzednimi dokumentami.**
- README mowi, ze instrukcje Kaggle sa w docs/KAGGLE_RUNBOOK.md. STAN_PRAC tez. PODSUMOWANIE (sekcja 13) odsyla do "docs/usage.md, rozdzial 6". Sprawdze w usage.md.
- STAN_PRAC: "Pelne szczegoly sa w docs/PODSUMOWANIE.md". PODSUMOWANIE (sekcja 0, pkt 7): "Szczegoly i tabele: docs/STAN_PRAC.md". Wzajemne odsylanie. Szczegoly interwencji i podmiany tokenow sa tylko w STAN_PRAC. PODSUMOWANIE ma je w jednym zdaniu.
- Komendy: README "make venv install, make test, make lint, make audit, make cues". PODSUMOWANIE (sekcja 11) "make install, make test, make audit, make cues" - bez "venv" i "lint".
- Liczby zgodne ze STAN_PRAC i README: 4,8 / 12,8 / 14,5; 7,6 / 20,1 / 19,0 / 15,9 / 11,1 / 7,3; 9,9 / 7,9 / 23,3 / 16,9; 9,0 / 17,6; 35 st.; 14,0 / 14,1; 9,4 / 44,3; wzory szumu -24,5 / 25,5.
- Liczby zgodne z audytem: 0,56 px, 0,14, IoU 98%, 5,2 st., 16%, 34 st., 18-19%, 4,7 / 7,6 st., tabela oswietlenia, 81%.
- Nazwa kontenerow: sekcja 1 mowi "enkoder SANA", STAN_PRAC i README "DC-AE". To ten sam enkoder, ale dwie nazwy.
- Sekcja 0, pkt 7 (audyt): "Projekt swietnie pasuje do SIWY" - ocena projektu w dokumencie projektu.
- Sekcja 1 wymienia szum dla t = 200, 500, 800 i bloki 4, 10, 18 w ZZSN. rq1 podaje inne konfiguracje wybrane na walidacji (np. v1: blok 7, t = 100). Nie sprzecznosc, ale nie jest powiedziane, ze nowa siatka t jest inna (100-800).
- Elewacja: tu (sekcje 2, 10) pojawiaja sie bledy elewacji (6,9; 6,3; 3,4). Wyjasnienie, czemu wyniki glowne dotycza tylko azymutu, jest tu (sekcja 10, pkt 2: jasnosc podlogi zdradza elewacje). README i rq1 tego nie mowia.

**6. Nieaktualne.**
- Sekcje 11, 12, 14 (patrz wyzej) i fraza "Bieżąca praca jest na galezi feat/kaggle-pipeline" - dokument z 16.10 opisuje stan z 13.10. Niewiadomo, czy galez jest nadal biezaca.
- "34 testy", "34 700 gwiazdek", "commit 916a579 z 13.10", "ok. 15 000 zgloszen" - liczby ulotne.
- Sekcja 13 (laptop, usypianie WSL, GitHub Credential Manager) - rady osobiste, bez znaczenia badawczego.
- Sekcja 12: teza "wynik negatywny" - stara.

**7. Brakujace pliki.** `docs/article/venues.md`, `docs/article/main.tex`, `reports/experiments/pixel_cues/results.md`, `tools/diffusers_upstream/*` (PR.md, fix.patch, functional/, repro_sana.py), `runs/kaggle/logs/queue.log`, `tools/viewer/`. Strata duza w jednym punkcie: nie widze samej poprawki diffusers ani tekstu zgloszenia. Dla zrozumienia projektu strata mala, bo sekcja 9 wystarcza jako opis.

**8. Jezyk.** Dobrze sie czyta, pelne zdania, dobre tabele. Problemy: (a) glos miesza "ja" (dopasowalem, sprawdzilem) i "wy" (wasz model, planowaliscie), a autorow jest dwoje. Dokument wyglada jak notatka asystenta dla zespolu, a nie dokument projektu. (b) Zdania administracyjne ("Czy PR daje 5 z przedmiotu?") w dokumencie badawczym. (c) Dlugosc: sekcja 9 (diffusers) ma ok. 100 linii i jest dluzsza niz sekcje o wynikach badan. (d) Ton: "Ciekawostka", "Nie do konca", "swietnie pasuje".

---

## 7. docs/design_proposal.md (2026-10-13, po angielsku)

**1. Cel i odbiorca.** Formalna propozycja projektu dla przedmiotu: streszczenie, tlo, pytania badawcze RQ1-RQ4 z hipotezami, plan eksperymentow, funkcjonalnosc (CLI), stos, plan obliczen, harmonogram na 18 tygodni, role, ryzyka, bibliografia. Odbiorca: koordynator przedmiotu. Z poczatku to jasne ("Design proposal", data, galaz). Nie ma jednak zdania, ze to wymog kursu. Jest dopiero pozniej w PODSUMOWANIE ("wymog SIWY").

**2. Co zrozumialem.** Projekt kontynuuje letni projekt ZZSN (w tym pliku nazwany "the old project"). Audyt tygodnia 1 pokazal osiem ustalen F1-F8. Nowy projekt ma zadac pytanie jeszcze raz. Ma poprawna ekstrakcje, bazy odniesienia (piksele, latent DC-AE, SANA z losowymi wagami), nowy zbior v2 i testy przyczynowe (LEACE i przesuniecie aktywacji). Pytania: RQ1 czy SANA ma wiecej niz piksele, RQ2 co sonda czyta (etykieta globalna czy lokalna), RQ3 ktora wskazowka niesie informacje, RQ4 czy model uzywa kierunku. Plan zaklada tez artykul (TMLR) i zgloszenie poprawki do diffusers.

**3. Pojecia bez wyjasnienia.**
- "F1 ... F8" - numeracja ustalen. Harmonogram (tydzien 1) mowi "F1 to F10". F9 i F10 nie istnieja w tabeli.
- "E0" - eksperyment zero (powtorzenie starego). Zdefiniowany w 4.1. W innych dokumentach sie nie pojawia.
- "model randomization check", "selectivity", "control task" - definicje w 4.2 sa krotkie, ale sa.
- "steer", "erase", "dose response" - jezyk interpretowalnosci, bez wyjasnienia.
- "Member A / B / C" - role bez imion. Nie wiadomo, kto jest A i B.
- "venues.md", "SLURM", "PLGrid" - bez wyjasnienia (PLGrid pojawilo sie tez w README).
- "ZZSN" - tu go nie ma wcale (tylko "old project"). W innych plikach jest wszedzie.

**4. Miejsca niejasne.**
- Brak informacji, ktore czesci planu sa wykonane. Dokument ma date 13.10, a harmonogram ma tygodnie do lutego 2027. Czytelnik nie wie, ze tygodnie 2-5 sa w duzej mierze zrobione do 16.10.
- Sekcja 4.4 (RQ3) planuje cztery warianty: z cieniem, bez cienia, "with the floor highlight" i "cue-conflict renders". W rzeczywistosci (data/examples) powstaly v2, v2ns, v2r, v2rns, v2rso, v2pos, v2rot. Nie ma wariantu z plama na podlodze ani prawdziwych "cue-conflict". Zmiana planu nie jest omowiona nigdzie.
- Sekcja 4.1: "4x4 average pool" jest stala. Pozniejsze dokumenty wybieraja siatke (1, 2, 4, 8) na walidacji. Nie jest oznaczone jako zmiana.
- Sekcja 7: "The features of one data set take about 50 GB". Z danych w 2.3 (287 KB na obraz, blok i krok) wychodzi 20 blokow x 5 krokow x 3000 obrazow x 287 KB, czyli ok. 86 GB. Rachunek sie nie zgadza, chyba ze zapisuje sie mniej konfiguracji.
- Sekcja 5.1: `dd3d extract` "runs as a SLURM array job". Sekcja 7 mowi, ze klaster jest wyczerpany i wszystko idzie na Kaggle. Wewnetrzna niespojnosc.
- Sekcja 9: role A, B, C. Nie wiadomo, jak sie maja do dwoch autorow z README.
- Sekcja 5.3 obiecuje logi TensorBoard i wyniki z hashem commita. Zadne inne dokumenty tego nie wspominaja.

**5. Sprzecznosci z poprzednimi dokumentami.**
- Sekcja 2.3: "An upstream issue and a patch ... are ready ... We did not send them yet." Harmonogram tydzien 6: "Send the upstream issue and patch." PODSUMOWANIE (rozdzial 9, tez z 13.10) i STAN_PRAC mowia, ze issue #15021 jest juz wyslane. Sprzecznosc statusu.
- Sekcja 2.2 F1: liczby 14,09 / 6,25 (sześciany), 11,45 / 5,60 (test), SANA 13,01 / 6,88 - zgodne z PODSUMOWANIE (14,1; 6,3; 11,5; 13,0).
- F2 zgodne z STAN_PRAC i PODSUMOWANIE (wzory szumu). F3 (35, 16, 35,5) zgodne z audytem. F4 (334,9, 62,7) zgodne.
- Lista CLI w 5.1 (extract, merge, probe, baseline, audit-arrows, viz, render-spec, erase, steer) vs README ("extract, merge, probe, viz, audit-arrows"). README podaje mniej komend. Sprawdze w usage.md.
- Hipoteza H4 ("An activation shift moves the shadow") jest czesciowo obalona przez STAN_PRAC (przesuniecie nie rysuje nowego cienia). Plik nie jest oznaczony jako historyczny.
- Hipoteza H1b ("SANA better than pixels ... on a data set without shortcut cues") jest potwierdzona (v2r). Brak odnotowania.
- Tydzien 1 w harmonogramie: "Start of the extraction of v1 and v2". STAN_PRAC pokazuje ekstrakcje v1, v2, v2r, v2rns, v2rso, SD, SDXL, losowe wagi - wszystko zrobione w tygodniu 1-2.
- "Last planned GPU job on 2026-11-13" vs STAN_PRAC (16.10): "W tym tygodniu GPU: ok. 27 z 30 h" i wszystkie wyniki wpisane. Plan jest wyprzedzony o ponad miesiac.
- PODSUMOWANIE (sekcja 11): "design_proposal - szkic do przejrzenia". Zgodne ze statusem "wersja z 13.10".

**6. Nieaktualne.** Calosc czesci "Planned ..." i "Schedule" jest czesciowo wykonana i nie ma kolumny statusu. "State after week 1" (2.3) jest przestarzale (np. "We did not send them yet"). Sekcja 7 mowi o szacunkach, a rzeczywiste czasy mozna znalezc w docs/KAGGLE_*. To jest dokument historyczny (zatwierdzony plan), ale nie jest tak oznaczony.

**7. Brakujace pliki.** `docs/article/venues.md`, `docs/literature.md` (do przeczytania). Odeslan do kodu brak. Strata: mala.

**8. Jezyk.** Prosty angielski, krotkie zdania, tabele. Bardzo czytelny. Struktura jak wymog formalny. Bibliografia 16 pozycji z linkami (sprawdzone 13.10). W STAN_PRAC jest cytat Haufe i in. (2014), ktorego tu nie ma (zapewne w literature.md).

---

## 8. docs/literature.md (2026-10-13, po angielsku)

**1. Cel i odbiorca.** Tabela literatury: 8 sekcji, razem 31 prac (policzylem 3+5+3+7+5+4+2+2). Kazda ma kolumny: venue, co robi, komentarz dla projektu, kod, wytrenowane modele, metryki, zasoby autorow. Dla koordynatora (wymog SIWY) i dla autorow projektu. Cel jasny z pierwszego akapitu. Liczba 31 zgadza sie z PODSUMOWANIE.

**2. Co zrozumialem.** To przeglad prac w osmiu grupach:

- sondy (Alain, Hewitt, Belinkov)
- pojecia i wymazywanie (TCAV, INLP, Amnesic, LEACE)
- testy poprawnosci (Adebayo, Geirhos)
- wiedza 3D w modelach (Chen, Zhan, El Banani i inni)
- sterowanie dyfuzja
- estymacja swiatla (IC-Light, DiLightNet, DiffusionLight)
- model SANA i dane CLEVR
- przeplyw (Flow Matching, Rectified Flow)

Kazda pozycja ma komentarz, jak jest uzyta w projekcie (np. Hewitt daje "selectivity", Elazar daje plan testu przyczynowego, Chen jest najblizsza praca). Sekcja 9 podaje, ze autorzy nie znalezli pracy, ktora sonduje kierunek swiatla w modelu dyfuzyjnym i porownuje z kontrola pikselowa.

**3. Pojecia bez wyjasnienia.** "squib", "h-space", "INLP", "SAE", "RIEBench", "DIFT", "PCK", "TPR-GAP", "si-RMSE" - zargon specjalistow, ale zwykle tlumaczony w zdaniu obok. "Selectivity" jest zdefiniowane. "Cue-conflict renders" - brak definicji, odsyla do planu (Geirhos).

**4. Miejsca niejasne.**
- "The BibTeX entries are in `article/references.bib`" - sciezka jest wzgledna (domyslnie docs/article/). Pliku nie ma.
- Sekcja 9 konczy sie zdaniem "To read next: ... Relighting as a Probe ...". To zapowiedz pracy niedokonczonej. Nie wiadomo, czy zostala przeczytana.
- Komentarze mowia w czasie przyszlym ("We train the probe on shuffled light labels", "We can use shadow and shade as separate cues in the cue-conflict renders"). Dla czytelnika po 16.10 nie jest jasne, co z tego zrobiono (kontrola z przetasowanymi etykietami - tak, 83-93 st.; cue-conflict - nie).
- Zdanie w sekcji 9 ("We found no such work") to twierdzenie o nowosci. Metoda wyszukiwania ("We searched the web") nie jest opisana (frazy, daty, bazy).

**5. Sprzecznosci.**
- STAN_PRAC (sekcja 5, wniosek 4) powoluje sie na "Haufe i in. (2014)". Ta praca nie wystepuje w tabeli 31 prac ani w bibliografii z design_proposal (16 pozycji). Moze jest w references.bib, ale nie w tej tabeli.
- Design proposal lista 16 pozycji, a wszystkie sa w literature.md (sprawdzilem po nazwiskach). Zgodne.
- Sekcja 7 (SANA): "The noise injection must follow the flow rule of the scheduler" - zgodne z reszta projektu.
- Brak sprzecznosci liczb.

**6. Nieaktualne.** Plany w czasie przyszlym (patrz wyzej). Pozycja "To read next" z 2026 jest ulotna. Sekcja 5 mowi "We use them for the planned control test", a test jest zrobiony.

**7. Brakujace pliki.** `article/references.bib` (BibTeX). Strata: niewielka.

**8. Jezyk.** Prosty angielski, rzeczowy i uczciwy: "not found", "not stated", "not applicable" sa zdefiniowane na poczatku i uzywane konsekwentnie. Tabela ma 8 kolumn i jest szeroka. W Markdown na waskim ekranie bedzie trudna do czytania. Dla czytelnika, ktory chce znac tylko 3-4 kluczowe prace, brakuje krotkiego wyboru ("przeczytaj najpierw Chen, Zhan, Hewitt").

---

## 9. docs/worklog.md (tydzien 1, 2026-10-12 do 2026-10-18)

**1. Cel i odbiorca.** Tygodniowy dziennik zespolu (Done, Findings, Next steps, Blockers). Jeden wpis (tydzien 1) i pusty szablon "Week N". Dla zespolu i dla prowadzacego (wymog SIWY wg PODSUMOWANIE). Cel jest jasny z pierwszego akapitu. Opis mowi, ze wpis pisze sie "at the end of the week", a w praktyce to dziennik z dopiskami "Update 2026-10-13/14/15".

**2. Co zrozumialem.** To skondensowana chronologia calej pracy do 15.10. Zawiera audyt, pakiet dd3d i przejscie z klastra na Kaggle. Opisuje renderowanie v2 i wariantow oraz ekstrakcje cech. Wymienia skan sond i test przyczynowy na 60 obrazach. Wymienia tez test podmiany (48 par) i drugi pomiar swiatla (kierunek ciemnej plamy). Ustalenia F1-F18 podaja liczby, ktore zgadzaja sie z README i STAN_PRAC. Blokery: brak GPU na klastrze i pytanie o liczebnosc zespolu (2 zamiast 3). Z tego dokumentu najlatwiej zobaczyc chronologie.

**3. Pojecia bez wyjasnienia.**
- "activation pattern" (F15), "pixel probe reading", "sigma0 0.85, 0.95", "patch test in 10 blocks and 3 regions" - bez definicji.
- "dd3d.store", "10 GB tar files", "templates/analyze.py" - nazwy z kodu.
- "287 KB for each image" - dla bloku i kroku (z kontekstu).
- "F1 ... F18" - numeracja laczy ustalenia audytu (F1-F10) z wynikami (F11-F18). To jedno pole "Findings" z dwoma rodzajami tresci.
- "VP schedules" - bez wyjasnienia (PODSUMOWANIE wyjasnia).

**4. Miejsca niejasne.**
- F13: "A probe on the mean of all tokens, fitted on v2, gives 9.0 / 17.6 deg on v2r. The pixel probe gives 71 deg." Tu jest "on v2r". PODSUMOWANIE (sekcja 10) podaje dla pikseli 69,0 st. na "obiektach w losowych miejscach (300 obrazow)", czyli v2pos. To moze tlumaczyc roznice 69,0 vs 71 (inny zbior: v2pos vs v2r). README, STAN_PRAC i data/examples nie mowia, na ktorym zbiorze liczy sie 9,0 / 17,6 / 71. Wniosek: liczba zalezy od zbioru, a nie jest to napisane.
- F16: "Only a patch of all tokens moves the shadow fully (84%)." 84% to nie "fully". Dokument sam to widzi (pelny przebieg zrodla daje 100%). Zdanie jest niedokladne. STAN_PRAC ma to samo ("Cały cień przenosi dopiero podmiana wszystkich tokenów").
- F9: "With the flow rule, the norm is 718." Tu jest jasne, ze to blok 4 i t = 200. To rozwiazuje moje pytanie z PODSUMOWANIE (718 vs 720 przy t = 800 jest przypadkiem, bo to rozne pomiary).
- F11-F18 powtarzaja tresc README i STAN_PRAC. Dziennik pelni wiec role trzeciego streszczenia wynikow.
- Sekcja "Next steps" nie ma zadnych zadan badawczych, tylko administracyjne (pytanie do koordynatora, przeglad jezyka, wybor czasopisma). To wynika ze stanu prac ("wszystkie wyniki sa wpisane").

**5. Sprzecznosci.**
- "We prepared an upstream issue and a patch ... We did not send them yet." Stare. PODSUMOWANIE (13.10) i STAN_PRAC (16.10) mowia, ze issue #15021 jest wyslane. Worklog ma wpisy do 15.10, wiec tekst nie byl zaktualizowany. Ta sama sprzecznosc co w design_proposal.
- Opis w dzienniku: tekst "Update 2026-10-13: ... the GPU allocation of the cluster is exhausted" zgodny z PODSUMOWANIE i design_proposal.
- Sekcja Done mowi "all renders (v2, v2pos, v2ns, v2r, v2rns, v2rso)" po 14.10. v2rot jest wyrenderowane osobno 15.10. Zgodne z data/examples (48 par).
- STAN_PRAC (sekcja 6) mowi o podmianie "każdy blok 0–18", a dziennik i patch.png maja 10 blokow (0, 2, ..., 18). To tylko co drugi blok. Opis w STAN_PRAC jest nieprecyzyjny.
- Liczby F12, F14, F15 zgodne z README, STAN_PRAC i rq1.

**6. Nieaktualne.** "We did not send them yet" (patrz wyzej). Szablon "Week N (YYYY-MM-DD...)" na koncu to puste miejsce, wiec latwo przeoczyc, ze to szablon. Brak wpisu dla tygodnia 2 (jeszcze nie ma, bo to 16-17.10).

**7. Brakujace pliki.** `docs/article/venues.md`, `tools/kaggle/templates/analyze.py`, `tools/viewer/`. Strata: mala.

**8. Jezyk.** Krotki angielski, punktory, czytelny. Dobry jako chronologia. Zawartosc F11-F18 jest gesta liczbowo. Dla nowego czytelnika bez STAN_PRAC niektore zdania (F15, F16, F18) sa trudne, bo zakladaja znajomosc pojec "pattern", "patch".

---

## 10. docs/usage.md (po angielsku)

**1. Cel i odbiorca.** Instrukcja uzytkowania: wymagania, instalacja, dane, komendy (audyt, sondy pikselowe, ekstrakcja, sonda na cechach, rysunki, test CPU), rendery Blendera, joby Kaggle, zasady dewelopmentu. Dla nowego czlonka zespolu. Cel jasny od pierwszego zdania. To pierwszy dokument, ktory daje komendy do skopiowania.

**2. Co zrozumialem.** Instalacja to `make venv`, `make install`, `make test` (tak samo jak w README). Dane v1 maja scene JSON (poza gitem), etykiety i splity (w gicie), a obrazy i cechy sa tylko na Kaggle. Komendy CLI: `dd3d audit-arrows`, `dd3d probe pixels`, `dd3d extract`, `dd3d merge`, `dd3d probe <katalog cech>`, `dd3d viz`. Ekstrakcja cech wymaga GPU 16 GB. Rendery wymagaja Blendera 2.78c. Sekcja 6 opisuje joby Kaggle (render, ekstrakcja) i ich limity (5 jobow CPU, brak bfloat16 na T4, 12 h na job).

**3. Pojecia bez wyjasnienia.**
- "`.[model,track]`" (sekcja 2) - nazwy dodatkow (extras) pakietu. Nie wiadomo, co instaluje `make install`, a co trzeba dodac. Jesli `make install` nie bierze `[model]`, to `dd3d extract` i test CPU nie dzialaja po instalacji lokalnej.
- "The queue downloads only the logs" (sekcja 6, krok 3) - "the queue" nie zostalo wprowadzone w tym pliku (jest `kqueue.py` tylko w PODSUMOWANIE i design_proposal).
- "analysis job" (krok 3, sekcja 6) - odeslane do runbook.
- "shard", "task-id / num-tasks", "--ref <commit>", "--kernel <owner>/<slug>" - zargon Kaggle, wyjasniony w runbook (dalej).
- "prompt cache" (4.6) - co to za pamiec i po co (domyslnie zapis embeddingow tekstu).
- "`git clone <repository>`" - brak adresu.
- "pool grid" w 4.3 ("The configuration file sets ... the pool grid") - nie powiedziano, ze zapisuje sie 8x8 i dalej agreguje.

**4. Miejsca niejasne i luki.**
- Sekcja 3: "Do not copy [images and features] to the laptop". Tymczasem 4.2 (`dd3d probe pixels`, `make cues`) i README ("make cues ... needs the v1 images") wymagaja obrazow lokalnie. Instrukcja jest wewnetrznie sprzeczna: komendy z 4.2, 4.3 i 4.4 nie maja gdzie dzialac na laptopie. Brakuje wprost zdania: "uruchom to jako job Kaggle (runbook)".
- `make audit` (README) i `data/final/scenes/<shape>/` jako "In git: no". Audyt opiera sie na plikach JSON scen. Nie wiem, czy `make audit` dziala na swiezym klonie. W README `make audit` nie ma uwagi o zaleznosci od danych (w przeciwienstwie do `make cues`).
- Luka glowna: brak instrukcji dla najwazniejszych wynikow nowych (skan sond na wielu blokach i krokach, test przyczynowy, podmiana tokenow, viewer, generowanie tabel typu rq1.md). Sekcja 4 kończy sie na pojedynczej sondzie (blok 10, t = 200). Do odtworzenia wyniku glownego trzeba czytac runbook i KAGGLE_DATA.
- Brak opisu ekstrakcji SD 1.5 i SDXL (PODSUMOWANIE: `extract_unet.py`) oraz interwencji (`intervene.py`).
- Sekcja 1 i 4.3: "a GPU with 16 GB (an A100 on the Athena cluster)" i "SLURM jobs in scripts/slurm". Klaster jest wyczerpany (design_proposal, PODSUMOWANIE). Sekcja 6 mowi o T4 na Kaggle. Wymagania i komendy z sekcji 1-4 sa nieaktualne wzgledem sekcji 6.
- 4.6: "CPU smoke test of SANA-1.6B ... needs about 4 GB free RAM and 17 GB of disk". PODSUMOWANIE (sekcja 13): "SANA na tym laptopie sie nie da ... w float32 model ma 6,4 GB i nie miesci sie w pamieci". Test dymny ma dzialac (z offloadem na dysk), ale nie wiadomo, czy to jest ta sama rzecz. Dwuznacznosc.
- Dwie sciezki do audytu strzalek: `dd3d audit-arrows` (usage) i `python scripts/audit/arrow_audit.py` (audit) oraz `make audit` (README). Nie wiadomo, czy to to samo.
- Brak sekcji "Jak dodac nowy eksperyment". Sekcja 7 (reguly) mowi tylko o testach, konfiguracji i commitach.
- Brak przewidywanych wynikow ("po tym kroku powinienes zobaczyc..."), brak czasow dla wiekszosci krokow (jest dla audytu w audit.md, dla renderu 25 s).

**5. Sprzecznosci z innymi dokumentami.**
- Instrukcja Kaggle: README i STAN_PRAC odsylaja do KAGGLE_RUNBOOK.md. PODSUMOWANIE i worklog odsylaja do usage.md, sekcja 6. Obie sekcje istnieja, ale dublowanie moze sie rozjechac.
- Komendy CLI: usage zawiera `dd3d probe pixels`, `dd3d probe <dir>`, `dd3d extract`, `dd3d merge`, `dd3d viz`, `dd3d audit-arrows`. To zgadza sie z README (extract, merge, probe, viz, audit-arrows) i NIE zgadza sie z design_proposal (baseline, render-spec, erase, steer). Design jest wiec nieaktualny wobec kodu. W README nie ma `probe pixels` jako osobnej komendy, ale to podkomenda.
- Uklad cech: usage 4.3 opisuje `features.npy` + `ids.txt` w `data/features/...`. Worklog mowi o "feature chunks" i 10 GB tarach na Kaggle. To dwa formaty. Czy oba obowiazuja, nie jest powiedziane.
- Sekcja 3: `data/final/` to v1 (to rozwiazuje pytanie z audytu: "data/final" = v1). Ale zadna inna czesc dokumentacji tego nie mowi wprost.
- README: "`make venv install`" jako jedna linia. Usage: dwie linie. Zgodne.
- `make format` (usage 7) nie ma w README.

**6. Nieaktualne.** Sekcja 1 (A100 na Athena), sekcja 2 (cluster), 4.3 (SLURM). Sekcja 6 i 3 sa aktualne.

**7. Brakujace pliki.** `scripts/slurm/README.md`, `configs/extract*.yaml`, `scripts/cpu/*.py`, `tools/kaggle/*.py`, `Makefile`. Strata: duza dla (a): nie widze Makefile ani konfiguracji. Czytam tylko to, co dokument mowi.

**8. Jezyk.** Prosty angielski, imperatyw, krotkie kroki. Dobry styl instrukcji. Naglowki ponumerowane (sekcja 6 "Jobs on Kaggle" jest cytowana w innych dokumentach po numerze, wiec numeracja nie moze sie zmienic). Przy konkretnych komendach dobrze jest podane "writes ..." (co powstaje).

---

## 11. docs/KAGGLE_RUNBOOK.md (po angielsku)

**1. Cel i odbiorca.** Instrukcja operacyjna dla osoby, ktora uruchamia lub zmienia joby na Kaggle: reguly, limity, narzedzia, plik kolejki, analiza, logi, znane problemy. Wprost: "Read it before you start or change a Kaggle job". Cel jasny. Odbiorca to czlonek zespolu z dostepem do konta autora, a nie obcy czytelnik.

**2. Co zrozumialem.** Cale obliczenia (rendery, ekstrakcja GPU, analiza CPU) ida na Kaggle jako "script jobs". Kazdy job klonuje repo z GitHuba na danym commicie. Narzedzia (`make_jobs.py`, `kqueue.py`, `kfetch.py`, `sync_reports.py`, `data_list.py`, `collect.py`) generuja foldery jobow, wysylaja je do kolejki, pobieraja male wyniki i kopiuja je do `reports/experiments/`. Cechy leza na Kaggle (tary z chunkami .npy), analiza tez dziala tam, a na laptop wraca tylko `results`, logi i male pliki. Etapy analizy: `scan-<set>`, `deep-<set>`, `transfer-pos`, `transfer-cue`, `plans-v2`, `causal-v2`, `patch-v2`.

**3. Pojecia bez wyjasnienia.**
- "deep-<set>", "transfer-pos", "transfer-cue", "plans-v2" - nazwy etapow. Nie wiadomo, co robi "deep" (domyslnie: baseline i wybor konfiguracji?) ani ktory etap daje jakie liczby z README (np. 9,0 / 17,6 to pewnie transfer-pos, ale to moj domysl).
- "render packs", "`.small`", "`.pattern`", "`.complete` vs `.done`" - markery bez rozroznienia (roznica miedzy `.complete` a `.done` nie jest opisana).
- "dd3d-renders", "dd3d-renders-tar", "dd3d-plans", "dd3d-v1-images" - nazwy zbiorow/jobow. Wyjasnia je dopiero KAGGLE_DATA (dalej).
- "Highmem", "T4x2", "task T of N writes chunks 2T and 2T+1 of 2N" - detale technike.
- "`/home/flang/tools/kag`" - sciezka na konkretnej maszynie autora. Nowy czlonek jej nie ma.
- "WSL", "Drive C:" - zalozenie srodowiska autora.

**4. Miejsca niejasne.**
- Brak sekwencji "od zera do tabeli rq1": kroki (1) wyrenderuj zbiory, (2) wyekstrahuj cechy, (3) odpal `analyze scan-<set>`, (4) `sync_reports.py`, nie sa zestawione w jednym miejscu. Sa tylko rozproszone kawalki (usage 6, ten plik "Start a new job" i "Analysis jobs"). Brak konkretnej komendy dla analizy (`make_jobs.py analyze --spec ...`), pokazanej jest tylko ekstrakcja.
- "Start a new job", krok 2: komenda ma "..." na srodku. Nie da sie jej skopiowac.
- Skad bierze sie `reports/experiments/rq1.md`? Narzedzia nie mowia o jego generowaniu. Wiadomo tylko, ze `sync_reports.py` kopiuje male wyniki. Nie wiem, czy tabela jest generowana, czy pisana recznie.
- Reguly mowia "Do not unpack renders or features on the laptop", a tabela narzedzi ma `collect.py` ("Unpacks downloaded tar files and merges feature chunks (local use only)"). Zdanie "local use only" jest niejasne.
- Dwa formaty cech: Kaggle ma chunki `chunk_<task>.npy`, a usage 4.3 `features.npy` po `dd3d merge`. Tutaj widac, ze `dd3d.store` czyta chunki na miejscu. Polaczenie nie jest wyjasnione.
- "At most 5 CPU jobs and 2 GPU jobs ... Accelerator NvidiaTeslaT4Highmem = two T4": nie wiadomo, czy job na dwoch T4 liczy sie jako jeden czy dwa joby GPU.
- Konto `jankombajn`: czy zbiory `dd3d-*` sa publiczne? Nic na ten temat. Jesli nie, zadna inna osoba nie odtworzy analizy bez dostepu do konta autora albo bez wyrenderowania wszystkiego od nowa (kilkanascie godzin CPU).

**5. Sprzecznosci.**
- Zgodne z usage.md (limity: 5 CPU, 2 GPU, 30 h, 12 h na job, brak bfloat16, Blender 7 s) i z PODSUMOWANIE (kolejka, `queue.log`).
- README i STAN_PRAC wskazuja ten plik jako instrukcje. PODSUMOWANIE i worklog wskazuja usage.md sekcja 6. Obie czesci istnieja i sie nie sprzeczaja, ale dubluja.
- `kqueue.py` pisze `docs/KAGGLE_DATA.md`, a `data_list.py` tez go pisze ("Add new data sets to its list"). To plik generowany. Reczna edycja bylaby nadpisana. Naglowek pliku KAGGLE_DATA powinien o tym mowic (sprawdze).
- Pozostale liczby zgodne.

**6. Nieaktualne.** Brak. Plik jest bardzo "zywy", wiec moze sie szybko zmieniac. Brak daty i wersji.

**7. Brakujace pliki.** `tools/kaggle/*.py`, `templates/analyze.py`, `configs/kaggle/*.json` (specyfikacje etapow). Strata: bardzo duza dla celu (b) "odtworzyc wynik", bo to w nich jest realna definicja analizy.

**8. Jezyk.** Zwiezly angielski, imperatywy, tabele. Dla nowego czytelnika zbyt skrotowy (zalozenie, ze wie, czym jest "job", "shard", "kernel"). Kilka zdan ma charakter notatek z doswiadczen ("A job with it waited 47 minutes for its inputs") - uzyteczne, ale pisane dla siebie.

---

## 12. docs/KAGGLE_DATA.md (po polsku, 120 linii)

**1. Cel i odbiorca.** Lista zbiorow danych Kaggle (16 datasetow) i wyjsc notatnikow (okolo 90 jobow: render, extract, analyze, intervene) ze stanem "gotowe". Dla zespolu, ktory chce znalezc, co lezy na Kaggle. Z samego poczatku jasne tylko ze tytulu. Brak jednego zdania wstepu. Plik jest generowany automatycznie (wg runbook: `kqueue.py` i `data_list.py` go pisza), ale plik tego nie mowi.

**2. Co zrozumialem.** Obrazy i cechy sa podzielone na 6 odcinkow na wersje (kule 0-499 i 500-999, walce, szesciany). Dla kazdej wersji jest job renderu, job ekstrakcji i job analizy (`scan`, `deep`). Tabela mowi, z jakich modeli sa cechy. Dla v1 i v2r to SANA, SANA z losowymi wagami, SD 1.5, SDXL (v1 dodatkowo "SANA stary szum"). Dla v2 tylko SANA i SD 1.5. Dla v2rns i v2rso tylko SANA. Opisy etapow tlumacza nazwy z runbook:

- "deep": wybrane sondy, porownania parami, kontrola z przetasowanymi etykietami, mapy czytania.
- "transfer-pos": sonda z v2 testowana na v2r.
- "transfer-cue": sonda z v2r testowana na v2rns i v2rso.
- "plans-v2": plany testu przyczynowego dla blokow 4, 10, 18 przy t = 400.

Plik podaje tez, gdzie kopie wynikow leza w `reports/experiments/...`.

**3. Pojecia bez wyjasnienia.**
- "Datasety" vs "Wyjscia notatnikow" - roznica miedzy zbiorem a wyjsciem joba jest w runbook, nie tu.
- "stan: gotowe" przy `dd3d-renders-png`, ktory jednoczesnie jest "niepelne: limit 500 plikow" - sprzeczne slowa w jednym wierszu.
- "plany ... (npz zostaja na Kaggle, lokalnie tylko JSON)" - brak objasnienia, czym sa plany.
- "sweep" (przemiatanie kierunku wzorca) - nowe pojecie, tylko tu.

**4. Miejsca niejasne.**
- Brak daty, rozmiaru, commita kodu i licencji dla kazdego wiersza. Nie wiadomo, ktora wersja kodu wyprodukowala dany zbior.
- Brak informacji o dostepie: czy linki `kaggle.com/datasets/jankombajn/...` sa publiczne, czy prywatne. Dla obcego czytelnika to krytyczne i nieznane.
- Plan testu przyczynowego jest dla t = 400 ("plans-v2: bloki 4, 10, 18; t = 400"). README i STAN_PRAC podkreslaja t = 100 jako najlepszy krok, a o t = 400 dla glownego testu przyczynowego nie mowia. To ukryty parametr.
- Brak zestawienia "do odtworzenia tabeli rq1 potrzebujesz: ...". Trzeba to skladac z 120 wierszy.
- Dla v2ns istnieja rendery (6 jobow), ale nie ma jobow ekstrakcji ani analizy. Zbior jest wiec wyrenderowany, a nigdzie niewykorzystany. Dokumenty nie tlumacza, po co.
- Lista nie zawiera jobow dla katalogow z wynikow STAN_PRAC: `reports/experiments/causal_b05/` (blok 5, t = 100) i `reports/experiments/causal_multi/` (wszystkie 20 blokow). W KAGGLE_DATA sa `causal`, `causal-sigma`, `patch`, `sweep`, ale nie te dwa. A STAN_PRAC mowi: "lista wszystkich jobow i danych jest w docs/KAGGLE_DATA.md". Lista jest wiec niepelna albo nieaktualna.
- `dd3d-renders-tar` i `dd3d-renders` nie wymieniaja v2rot (a v2rot ma osobny job `dd3d-render-v2rot-0`). Opis "wszystkie nowe obrazy" jest nieprecyzyjny.

**5. Sprzecznosci.**
- Zgodne z rq1 i README: puste komorki w tabelach wynikaja z braku takich ekstrakcji. Przyklady: brak SDXL dla v2, brak losowych wag dla v2rns i v2rso. To rozwiazuje moje pytanie z rq1. Wyjasnienie jest jednak tylko w tym pliku.
- Zgodne z STAN_PRAC: 60 obrazow (causal), 48 par, 10 blokow, 3 obszary (patch), sigma0 = 0,85 i 0,95 (causal-sigma), bloki 4, 10, 18.
- Zgodne z README: `dd3d-v1-images` istnieje.
- Niezgodne z README ("Kaggle data set dd3d-v1-images ... make cues") nie ma. Niezgodne z usage.md ("Do not copy images to laptop"): brak, to inny aspekt.
- STAN_PRAC "lista wszystkich jobow" - patrz wyzej (niepelna).

**6. Nieaktualne.** Brak dat, wiec nie da sie ocenic. Plik jest odswiezany automatycznie, ale nie zawiera najnowszych jobow (causal_b05, causal_multi).

**7. Brakujace pliki.** Wszystkie linki prowadza do Kaggle (poza folderem). Tu strata jest pelna: nie moge sprawdzic niczego.

**8. Jezyk.** Po polsku, w dokumencie, ktory jest wskazywany z anglojezycznego README i z anglojezycznego runbook. Tabela jest dluga i powtarzalna (6 wierszy na wersje) i trudna do czytania. Brakuje grupowania i podsumowania ("Najwazniejsze zbiory: ..."). Dla autorow wystarczy, dla nowego czytelnika to ciag wierszy.

---

# CZESC KONCOWA (po przeczytaniu calosci)

## A. Czym jest ten projekt, tak jak to zrozumialem

**Pytanie.** Czy model tekst-obraz SANA-1.6B (transformer dyfuzyjny) ma w aktywacjach kierunek swiatla sceny 3D. Czy ma go wyraznie bardziej niz da sie odczytac wprost z pikseli. Czy model tego kierunku uzywa przy generowaniu obrazu.

**Metoda.**

- Sceny CLEVR w Blenderze 2.78c. Jeden fioletowy obiekt (kula, walec, szescian). Losowy azymut (0-360 st.) i elewacja (30-75 st.) swiatla.
- Obraz jest kodowany przez DC-AE i zaszumiany regula przeplywu. Zapisuje sie aktywacje 20 blokow transformera dla krokow t = 100-800.
- Sonda liniowa (usrednianie na siatce, PCA 256, Ridge) przewiduje [sin az, cos az, el]. Blad azymutu jest kolowy.
- Sonda uczy sie na kulach i walcach, a test jest na szescianach.
- Kontrole: piksele, latent DC-AE, SANA z losowymi wagami, SD 1.5, SDXL, przetasowane etykiety. Przedzialy to 95% bootstrap.

**Historia.** Wczesniejszy projekt (ZZSN, czerwiec 2026) twierdzil, ze SANA koduje kierunek swiatla (14,0 st. na szescianach). Audyt z 12-13.10 to obalil z kilku powodow:

- Ta sama sonda na pikselach daje 14,1 st.
- Etykieta jest pozycja lampy, a nie swiatlem przy obiekcie (srednio 35 st. roznicy).
- Plama swiatla na podlodze jest skrotem.
- W kazdej scenie swieci dodatkowa lampa.
- Ekstrakcja miala blad szumu (`add_noise` przed `set_timesteps` w diffusers). Przy t = 200 model dostawal -24,5 x0 + 25,5 eps.
- Strzalki na rysunkach byly zle, a siatka alfa w Ridge nic nie zmieniala.

**Nowe dane:**

- v2: jedna lampa SUN, plaska podloga, obiekt w srodku.
- v2r: to samo, ale obiekt w losowym miejscu.
- v2ns i v2rns: bez cienia.
- v2rso: sam cien.
- v2pos: test z losowa pozycja.
- v2rot: 48 par z azymutem +90 st.

**Wyniki.**

- Z poprawnym szumem na v1: SANA 4,8 st. na szescianach, najlepsze piksele 12,8 st. Ze starym szumem 14,5 st.
- Na v2r (bez skrotow): SANA 7,6 st., piksele 20,1, latent 19,0, losowe wagi 15,9. SDXL 7,3 st., SD 1.5 11,1 st.
- SANA czyta kierunek ze samego cieniowania (9,9 st.) i z samego cienia (7,9 st.).
- Sonda na sredniej tokenow, uczona na obiektach w srodku, daje 9,0 st. (17,6 st. na szescianach) na losowych pozycjach. Piksele daja 71 st.
- Test przyczynowy: przesuniecie aktywacji wzdluz "wzorca" swiatla (2% normy tokenu) obraca cieniowanie na wyjsciu o 60-65 st. w strone celu +90 st. Losowe przesuniecie nic nie robi.
- To przesuniecie nie rysuje nowego cienia rzucanego. Przy duzej dawce usuwa czesc starego i przyciemnia podloge.
- Podmiana tokenow obiektu miedzy scenami przenosi cieniowanie i okolo jednej trzeciej cienia.

**Wniosek autorow.** Stary wynik nie dowodzil niczego. Po poprawkach SANA (podobnie SD 1.5 i SDXL) ma kierunek swiatla w cechach wyraznie ponad pikselami. Model uzywa go do cieniowania obiektu. Cien rzucany powstaje tylko po czesci z informacji przy obiekcie. Do tego dochodzi poprawka do biblioteki diffusers (issue #15021), szkic artykulu w formacie TMLR i narzedzia Kaggle.

## B. Czy po lekturze umialbym...

**(a) Zainstalowac i uruchomic testy: tak, prawie.** `make venv install` i `make test` (README, usage 2) wystarcza na Linuxie lub WSL2 z Pythonem 3.10+. Czego brakuje:

- adres repozytorium (`git clone <repository>`),
- informacja, co `make install` instaluje (czy dodatek `[model]` i `[track]`),
- oczekiwany wynik (liczba testow i czas jest tylko w PODSUMOWANIE: 34 testy, ok. 10 s),
- wymagania co do wersji `make` i narzedzi.

**(b) Odtworzyc glowny wynik: nie.** Powody:

- Nie ma jednej sciezki "od zera do tabeli rq1". Kroki sa rozproszone (usage 6, runbook, KAGGLE_DATA).
- Wszystko wymaga konta Kaggle z dostepem do zbiorow `jankombajn/dd3d-*`. Nie wiadomo, czy sa publiczne.
- Instrukcje sa sprzeczne. usage 3: "Do not copy them to the laptop". README: `make cues` "needs the v1 images". Komendy 4.2-4.4 nie maja gdzie dzialac lokalnie.
- Komenda dla etapu analizy (`make_jobs.py analyze --spec ...`) nie jest pokazana. Specyfikacje `configs/kaggle/*.json` i szablony sa poza dokumentacja.
- Nie wiadomo, skad powstaje `rq1.md` i jak sa wybierane blok, t i siatka.
- Brakuje seedow, wersji bibliotek i commitow danych.
- Sciezka `/home/flang/tools/kag` i konto autora sa specyficzne dla jednej maszyny.
- Mozna najwyzej: uruchomic testy, audyt strzalek (jesli JSON scen sa w repo) i obejrzec gotowe wyniki.

**(c) Dodac nowy eksperyment: czesciowo.** Mam zasady (test jednostkowy, konfiguracja w `configs/`, commity konwencjonalne) i opis jobow Kaggle (runbook). Brakuje:

- opisu ukladu pakietu `src/dd3d` (jest tylko lista modulow),
- opisu interfejsu szablonu `templates/analyze.py` i formatu specyfikacji,
- instrukcji dodania nowego wariantu danych (`make_v2_spec.py`, etykiety, splity, wpis w `data_list.py`),
- instrukcji dla interwencji i podmiany tokenow,
- slownika danych (kolumny CSV, definicja splitow).

## C. Struktura dokumentacji

**Logika.** Uklad jest rozsadny: wejscie (README), dane (data/examples), wyniki (rq1), status (STAN_PRAC), narracja (PODSUMOWANIE), audyt, plan, literatura, dziennik, instrukcje. Sam podzial na pliki mi odpowiada. Problemem jest to, ze nie ma jednego zrodla prawdy i trzy dokumenty sie powtarzaja.

**Powtorzenia.**

- Wyniki glowne: README, STAN_PRAC (sekcja 4), PODSUMOWANIE (sekcja 0), worklog (F11-F18) i rq1. Pieciokrotnie.
- Ustalenia audytu: audyt, STAN_PRAC (sekcja 2), PODSUMOWANIE (rozdzialy 0-8), design_proposal (F1-F8) i worklog (F1-F8).
- Blad szumu: README, STAN_PRAC, PODSUMOWANIE (rozdzial 9), design_proposal (F2), worklog (F2).
- Instrukcja Kaggle: usage (sekcja 6) i runbook.
- Pytania administracyjne: STAN_PRAC, PODSUMOWANIE (rozdzial 14), design_proposal, worklog.

**Dokumenty, ktore sie wykluczaja (stan na 16.10).**

- design_proposal i worklog: "We did not send them yet" vs PODSUMOWANIE i STAN_PRAC: issue #15021 wyslane.
- PODSUMOWANIE (rozdzial 12): teza artykulu "wynik negatywny" i brak wynikow GPU vs README i STAN_PRAC: wynik pozytywny, wszystkie wyniki wpisane.
- PODSUMOWANIE (rozdzial 10): "zaden model nie pokaze przewagi nad pikselami" na v2 vs rq1: SANA bije piksele na szescianach v2 o 6,3 st.
- design_proposal (CLI: baseline, render-spec, erase, steer) vs README i usage (extract, merge, probe, viz, audit-arrows).
- STAN_PRAC: "Pelne szczegoly sa w PODSUMOWANIE" vs PODSUMOWANIE: "Szczegoly i tabele: STAN_PRAC".

**Brakuje.**

1. Slownika pojec (SANA, DC-AE, flow noise, blok, t, siatka, wzorzec, sonda, cieniowanie vs cien, wskazowka, ZZSN, SIWY, RQ1-RQ4).
2. Slownika danych (kolumny CSV, splity 1600/200/200/1000, nazwy plikow, mapowanie `data/final` = v1).
3. Indeksu wynikow: kazda liczba z README z odsylaczem do pliku lub joba.
4. Opisu metody w repo (jest tylko w artykule, ktorego nie mam). README Method opisuje tylko v1.
5. Przewodnika odtworzenia.
6. Opisu architektury `src/dd3d`.
7. Sekcji "Ograniczenia" (jedna klasa obiektow, sceny syntetyczne, tylko azymut, 60 obrazow i 48 par w tescie przyczynowym, slaba kontrola losowa).
8. Oznaczen "historyczne" na planach (design_proposal, literatura, czesc usage).

**Zrodlo prawdy dla wynikow.** Nie jest jasne. Dla tabeli sond kandydatem jest `reports/experiments/rq1.md` (README tak mowi), ale plik nie ma legendy i nie zawiera wynikow transferu ani interwencji. Dla interwencji i podmiany tokenow jest to STAN_PRAC (liczby) i pliki `reports/experiments/causal*/patch/` (ktorych nie widze). Dla audytu jest to `docs/audit/labels_and_arrows.md` i PODSUMOWANIE. Artykul (`main.tex`) jest wskazany jako kanoniczny, ale go nie ma w folderze. Liczby w README, rq1, STAN_PRAC i worklog sa jednak zgodne (sprawdzilem komorka po komorce). Problem nie jest w bledach liczb, tylko w wielu kopiach.

## D. 10 najwazniejszych problemow (od najpowazniejszego)

1. **Brak sciezki odtworzenia wyniku glownego i sprzeczne instrukcje danych.**
   - usage.md (sekcja 3): "The images and the features are on Kaggle only ... Do not copy them to the laptop."
   - README: "make cues # pixel probes on image regions (needs the v1 images of the Kaggle data set dd3d-v1-images)".
   - KAGGLE_RUNBOOK: "Use `/home/flang/tools/kag` for all Kaggle CLI calls" i "Account `jankombajn`".
   - Analiza zalezy od konta jednej osoby, a komendy nie maja miejsca uruchomienia.
2. **Nieaktualne statusy w dokumentach, ktore wygladaja na biezace.**
   - PODSUMOWANIE ma date 16.10, ale rozdzial 11 mowi: "Szkic artykulu ... szkic, brakuje wynikow z GPU". STAN_PRAC (16.10): "wszystkie wyniki sa wpisane".
   - worklog i design_proposal: "We did not send them yet" vs PODSUMOWANIE: "zgloszenie jest wyslane: .../issues/15021".
   - PODSUMOWANIE (rozdzial 12): "Wynik negatywny tez jest tu wartosciowy" vs README: "SANA-1.6B carries the light direction far beyond the pixels".
3. **Naglowkowe twierdzenia sa mocniejsze lub bardziej wybiorcze niz dane w dokumentacji.**
   - README pogrubia SANA w wierszu v2r (7,6 st.), a SDXL ma 7,3 st. STAN_PRAC przyznaje, ze efekt jest wspolny dla modeli dyfuzyjnych.
   - README: "turns the shading by 60° to 65°". STAN_PRAC: dla bloku 5 przy dawce x1 wychodzi +30 st. Rysunek examples.png pokazuje +59, +27, +25, +15 st.
   - STAN_PRAC: "Czy ten kod zależy od pozycji obiektu w kadrze? | Nie." przy 17,6 st. na szescianach (vs 7,6 st. dla sondy uczonej na v2r).
   - STAN_PRAC: "Cały cień przenosi dopiero podmiana wszystkich tokenów". Tabela: 0,84 (blok 10).
   - Test przyczynowy ma tylko losowa kontrole tej samej normy. Plany uzywaja t = 400 (KAGGLE_DATA), o czym README i STAN_PRAC nie mowia.
4. **Kluczowe pojecia nie sa zdefiniowane w czytanych dokumentach.**
   - STAN_PRAC: "wzorzec ×1 (zmiana 2% normy tokenu)". Nigdzie nie ma definicji wzorca.
   - README: "A shift along the activation pattern of the light".
   - PODSUMOWANIE (rozdzial 9) uzywa "wzorzec" jako wartosc odniesienia. To drugie znaczenie tego slowa.
   - Brak definicji: "code", "grid", "flow noise", "cue", ZZSN, SIWY, RQ1-RQ4, skala t.
5. **rq1.md to goła tabela bez legendy.**
   - Brak jednostek, znaczenia przedzialow, definicji "Test", wyjasnienia "Block, t, grid" (te same kolumny dla pikseli maja "16 px").
   - Nazwy "v2r, no shadow" i "v2r, only shadow" (zamiast v2rns i v2rso).
   - Dwa wiersze pikseli maja ten sam test "24.6" i prawie te same przedzialy ("24.6 [20.7, 28.8]" i "24.6 [20.7, 29.0]").
   - README i rq1 podaja tylko azymut. Powod (elevation zdradza jasnosc podlogi) jest tylko w PODSUMOWANIE (rozdzial 10).
6. **Rozne liczby bazowe bez wyjasnienia.**
   - Piksele na szescianach v1: 12,8 (README, rq1), 14,1 (PODSUMOWANIE, 64x64), 14,3 (cues, linia w scan.png).
   - scan.png: linia pikseli dla v2r jest przy ok. 25,8, a tabela mowi 20,1.
   - Piksele na losowych pozycjach: 71 (README, worklog, na v2r) vs 69,0 (PODSUMOWANIE, v2pos). Roznica wynika chyba ze zbioru, ale nikt tego nie pisze.
   - probe_maps_pixels.png: v2 "cubes az MAE 40.2 deg" vs rq1 13,7 (inna siatka).
7. **Zbyt wiele kopii tych samych tresci.** Wyniki glowne w pieciu miejscach, audyt w pieciu. STAN_PRAC i PODSUMOWANIE wzajemnie odsylaja po szczegoly. Dryf jest juz widoczny (problem 2).
8. **Brak slownika danych i niejasne nazwy.**
   - "v1" = `data/final` = `master_labels.csv`. Sceny v1 maja nazwy "SmoothCube_v2_new_000004" (kolizja z "v2").
   - data/examples/README: "differ from the v2 row in one property only". To nieprawda dla v2rns i v2rso (dwie wlasciwosci wzgledem v2).
   - v2ns jest wyrenderowane, ale nie ma jego ekstrakcji ani wyniku.
   - v2rso ma czarna plame na miejscu obiektu (wyjasnienie jest tylko w PODSUMOWANIE).
   - Splity 1600/200/200/1000 tylko w design_proposal i PODSUMOWANIE.
9. **Plany nie sa oznaczone jako historyczne.**
   - design_proposal 5.1: `dd3d baseline`, `render-spec`, `erase`, `steer`. README i usage tych komend nie maja.
   - design_proposal 4.4: warianty "with the floor highlight" i "cue-conflict renders". W danych ich nie ma.
   - usage sekcja 1 i 4.3: "an A100 on the Athena cluster", "SLURM jobs". Klaster jest wyczerpany.
   - design_proposal (sekcja 7): "about 50 GB" cech, a rachunek z 287 KB daje ok. 86 GB.
10. **Mieszanie odbiorcow, glosow i jezykow oraz niepelna lista jobow.**
    - PODSUMOWANIE: "wasz model", "planowaliscie" (glos "wy"), "dopasowalem" (glos "ja"). Pytania w rodzaju "Czy PR daje 5 z przedmiotu?" w dokumencie badawczym.
    - STAN_PRAC nazywa sie "krotkim przegladem", a ma 175 linii. Zawiera "W tym tygodniu GPU: ok. 27 z 30 h".
    - KAGGLE_DATA po polsku (generowany, bez wstepu), a wskazuja na niego angielskie README i runbook. Lista nie zawiera jobow dla `causal_b05` i `causal_multi`, a STAN_PRAC mowi, ze jest "lista wszystkich jobow".
    - Rady osobiste (usypianie WSL, Credential Manager) w PODSUMOWANIE.

## E. Co jest zrobione dobrze

- Pierwsze trzy zdania README. Pytanie jest jasne, a model, wlasciwosc i test (uczenie na kulach i walcach, test na szescianach) sa zrozumiale od razu.
- Uczciwy baner statusu w README: pierwszy wynik "does not show what it claimed". Wyniki negatywne i niedomkniete (brak nowego cienia) sa pokazane.
- Audyt (labels_and_arrows.md): ponumerowane ustalenia L1-L7 i A1-A6, testy dokladnosci kopii (0,56 px, IoU 0,98), konkretne rekomendacje i sekcja "How to repeat the audit".
- STAN_PRAC: diagramy mermaid (pipeline, rodowod danych, test przyczynowy), tabele z kontrolami, obrazy przykladowe, wnioski podane osobno od liczb.
- data/examples: arkusz overview.jpg i tabela wersji. Najlepsza pomoc dla nowego czytelnika.
- Kontrole metodologiczne: piksele, latent, losowe wagi, przetasowane etykiety, SD 1.5 i SDXL, przedzialy bootstrap, blad kolowy, wybor hiperparametrow tylko na walidacji.
- Spojnosc liczb: porownalem rq1, README, STAN_PRAC, worklog i PODSUMOWANIE komorka po komorce. Liczby glowne sie zgadzaja (roznice +-0,1 to zaokraglenia).
- PODSUMOWANIE, rozdzial 9: wyjasnienie bledu diffusers z tabelami, testami na trzech modelach, ograniczeniami i zastrzezeniem co do przeszukania zgloszen.
- literature.md: konwencja "not found / not stated / not applicable" i data weryfikacji. design_proposal: hipotezy, kryteria sukcesu, ryzyka, harmonogram.
- Runbook: sekcje "Rules" i "Known problems" to praktyczna wiedza.
- Wykresy scan.png i patch.png sa czytelne i zgodne z tekstem. examples.png uczciwie pokazuje artefakty (prostokaty).

## F. Lista plikow

Wszystkie otwarte pliki leza w folderze `<izolowana kopia dokumentacji>/` (podane niżej wzgledem niego). Pliki zapisane: `<notatki>/docs_reader.md`.

Otwarte (istnialy):

- README.md
- data/examples/README.md
- data/examples/overview.jpg
- reports/experiments/rq1.md
- docs/STAN_PRAC.md
- docs/audit/labels_and_arrows.md
- docs/PODSUMOWANIE.md
- docs/design_proposal.md
- docs/literature.md
- docs/worklog.md
- docs/usage.md
- docs/KAGGLE_RUNBOOK.md
- docs/KAGGLE_DATA.md
- docs/figures/readme_good.png
- docs/figures/readme_bad.png
- docs/figures/v2_sample.png
- docs/figures/cues.png
- docs/figures/conventions.png
- docs/figures/lighting_study.png
- docs/figures/shadow_audit.png
- docs/figures/v2_grid.png
- docs/figures/probe_maps_pixels.png
- docs/figures/v2_variants.png
- docs/figures/v2r_corners_rejected.png
- docs/article/figures/scan.png
- docs/article/figures/patch.png
- reports/experiments/causal_b05/b05t100/examples.png
- reports/experiments/causal_multi/all/examples.png

Proba otwarcia (plik nie istnial w folderze):

- reports/audit/shadow_audit.png
- reports/audit/lighting_study.png

Nie otwieralem (poza listą zadania lub poza folderem): docs/article/venues.md, docs/article/main.tex, docs/article/references.bib, kod, Makefile.

