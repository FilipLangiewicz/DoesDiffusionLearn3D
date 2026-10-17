# Audyt walidacyjny projektu (17.10.2026)

Gałąź `audit/validation`. Ten audyt nie zmienia żadnego istniejącego pliku. Dodaje tylko ten raport, notatki dwóch niezależnych czytelników, cztery skrypty kontrolne i ich wyniki z Kaggle.

## Jak powstał ten raport

1. **Ślepy czytelnik artykułu.** Osobny agent dostał tylko PDF artykułu, bez kodu i dokumentacji. Udawał kogoś pomiędzy studentem magisterki z AI a prowadzącym SIWY. Zapisywał po każdej porcji tekstu, co rozumie, czego brakuje i co mu się nie zgadza. Notatki: [`czytelnik_artykulu.md`](czytelnik_artykulu.md).
2. **Ślepy czytelnik dokumentacji.** Drugi agent dostał tylko README i pliki `docs/*.md` (z obrazami), bez kodu i artykułu. Notatki: [`czytelnik_dokumentacji.md`](czytelnik_dokumentacji.md).
3. **Sprawdzenie liczb.** Każdą liczbę z artykułu, która ma źródło w `reports/`, porównałem z plikiem wyników.
4. **Przegląd kodu**, który daje te liczby: ekstrakcja, szum, sonda, metryki, interwencje, miary cienia, plany i analizy.
5. **Testy kontrolne na Kaggle (CPU)** w czterech miejscach, gdzie z kodu wynikało podejrzenie. Skrypty: `scripts/audit/validation/`. Wyniki: [`kaggle/`](kaggle/).
6. **Regulamin SIWY**: oba PDF-y z `docs/`, punkt po punkcie.

Skala ważności:

- **KRYTYCZNE**: zmienia wniosek artykułu albo grozi niezaliczeniem projektu.
- **WAŻNE**: recenzent to zauważy, a tekst trzeba poprawić, ale wniosek zostaje.
- **DROBNE**: liczba, słowo, podpis, porządek.

---

## 0. Werdykt w skrócie

**Co się broni (sprawdzone):**

- Audyt projektu ZZSN: sonda na pikselach jest tak samo dobra jak sonda na SANA, etykieta to pozycja lampy, plama światła na podłodze jest skrótem 2D, a szum był wstrzyknięty błędnie. Wszystko to ma pokrycie w kodzie i w plikach wyników.
- Poprawiony pipeline: wynik 4,8° wobec 14,5° na v1 jest uczciwie wyliczony. Ten sam protokół, wybór na walidacji.
- Główna tabela RQ1 (`rq1.tex` = `reports/experiments/rq1.md`): wszystkie liczby zgadzają się z plikami skanu. Na v2r SANA daje 7,6° wobec 20,1° dla najlepszej siatki pikseli, z kontrolami (latent, losowe wagi, przetasowane etykiety). Wyciek z testu nie występuje: PCA, skaler i alfa są dopasowane tylko na train/val.
- Kod ekstrakcji, reguły szumu, metryk kołowych, SDEdit i łatania robi to, co opisuje artykuł (szczegóły w sekcji 6).
- Liczby w tabelach testu przyczynowego (Tabela 5 i 6) zgadzają się z `reports/experiments/causal*/results.csv`.

**Co się nie broni albo wymaga zmiany (szczegóły niżej):**

1. **Łatanie tokenów obiektu (K1, potwierdzone na Kaggle).** „Około jednej trzeciej cienia” to w większości kopia pikseli w samych podmienionych komórkach. Dalej od obiektu cień źródła się nie pojawia (5% wobec 3% bez podmiany). Zdanie „object tokens carry part of the cast shadow” trzeba wycofać ze streszczenia, z wyników, z zakończenia i z README.
2. **Wymazanie LEACE (K3, potwierdzone na Kaggle).** Eraser nie usuwa informacji, którą czyta sonda na siatce (4,6° przed i 4,7° po). Wynik „no effect” nie jest więc testem użycia kodu, a opis erasera w artykule jest nieprawdziwy.
3. **Rama „3D czy 2D” i „ponad wejście” (K4).** Dowody pokazują przewagę nad *liniową* sondą na pikselach, a nie wiedzę 3D. Brakuje bazy z modelu niegeneratywnego.
4. **Regulamin (K5).** Jeden autor wszystkich commitów w SIWY i zespół 2-osobowy. To największe ryzyko formalne.
5. **Kilkanaście rozbieżności liczb i opisów** między artykułem a plikami wyników (sekcja 3, W1–W12), m.in. opis sondy („standard scores”, których w kodzie nie ma), Tabela 1 bez źródła w repo, kalibracja miar cienia.
6. **Dokumentacja:** za dużo źródeł prawdy, nieaktualne pliki, brak ścieżki odtworzenia (sekcja 5).

**Co sprawdziłem i okazało się w porządku:** przesunięcie wzdłuż wzorca naprawdę obraca cieniowanie samego obiektu (K2). Mierzy to sonda na wycinku obiektu, bez podłogi i cienia. Do poprawy jest tylko sposób podawania liczby „60–65°”.

**Ocena gotowości:** artykuł nie jest jeszcze gotowy do zgłoszenia. Część poprawek to sam tekst. Kilka wymaga krótkich analiz na CPU (bez nowego GPU). Jedna rekomendacja (baza z modelu niegeneratywnego) wymaga ok. 1 h GPU.

---

## 1. Ustalenia krytyczne

### K1. Łatanie tokenów obiektu nie przenosi cienia rzucanego. „Jedna trzecia cienia” to kopia pikseli w podmienionych komórkach

**Co mówi artykuł.** Streszczenie: „A patch of the object tokens from a scene with another light moves the shading and about a third of the cast shadow”. Wyniki: „It also moves part of the cast shadow: 33% to 38% of the crisp source shadow appears”. Zakończenie: „The tokens at the object carry part of it.” To samo jest w README, STAN_PRAC i worklogu (F16).

**Podejrzenie z kodu.** Obszar „object” to komórki siatki 16×16 (po 32 px), które dotykają bryły obiektu (`make_patch_plan.py`, `probe_maps.regions`). Takie komórki obejmują też podłogę tuż przy obiekcie, a tam zaczyna się cień. Podmiana tych tokenów może więc po prostu skopiować początek cienia źródła. Analiza (`analyze_patching.py`) liczy udział cienia źródła w *całym* obrazie, razem z tymi komórkami.

**Sprawdzenie (Kaggle CPU, `patch_copy_check.py`, 48 par, prawdziwe obrazy wyjściowe).** Ostry cień wyjścia źródłowego podzieliłem na część w komórkach obiektu i część poza nimi. Druga wersja sprawdzenia poszerza komórki o 16 px, czyli jeden token, bo dekoder rozmywa zmianę na sąsiednie piksele.

| Warunek (blok 10, o ile nie podano inaczej) | Cień źródła w całym obrazie (jak w artykule) | W komórkach obiektu | Poza komórkami | Poza komórkami + 16 px | Obrót ciemnej plamy poza komórkami (część 90°) |
|---|---:|---:|---:|---:|---:|
| bez podmiany (odniesienie) | 0,12 | 0,20 | 0,04 | 0,03 | 0,00 |
| tokeny obiektu | 0,38 | **0,63** | 0,21 | **0,05** | 0,03 |
| tokeny obiektu, blok 18 | 0,35 | 0,71 | 0,09 | **0,03** | 0,02 |
| wszystkie tokeny | 0,84 | 0,83 | 0,82 | 0,81 | 0,86 |
| pełny przebieg źródła | 1,00 | 1,00 | 1,00 | 1,00 | 0,93 |

Średnio **42% ostrego cienia źródła leży w samych komórkach obiektu**, a 70% w komórkach poszerzonych o jeden token. Podmiana tokenów obiektu odtwarza cień źródła prawie tylko tam, gdzie podmienia tokeny. Dalej niż jeden token od komórek pojawia się 5% cienia źródła (w bloku 18: 3%), a bez żadnej podmiany 3%. Kierunek ciemnej plamy poza komórkami się nie obraca (0,03 wobec 0,93 dla źródła). Rysunek [`kaggle/patch_copy/examples.png`](kaggle/patch_copy/examples.png) pokazuje to wprost: po podmianie stary cień zostaje na miejscu, a zielony kontur cienia źródła poza czerwoną ramką jest pusty.

**Wniosek.** Ten problem jest realny, nie przesadzam. Zdanie „object tokens carry part of the cast shadow” nie ma pokrycia w danych. Dane mówią coś prostszego i spójnego z resztą artykułu: tokeny obiektu niosą cieniowanie obiektu, a cień rzucany *daleko od obiektu* od nich nie zależy. Każdy obszar zmienia obraz tylko u siebie i w najbliższym sąsiedztwie. Cały cień przenosi dopiero podmiana wszystkich tokenów. To samo dotyczy zdania o pierwszej i drugiej połowie kroków (28% wobec 16%): poza poszerzonymi komórkami jest to 5% wobec 3%.

**Co zrobić.** Poprawić streszczenie, wyniki, dyskusję, zakończenie, README, worklog F16 i Rysunek 8 (prawy panel). Do analizy łatania dodać miarę „poza obszarem podmiany” (skrypt audytu można przenieść do `scripts/experiments/`). Ta poprawka wzmacnia główną tezę artykułu (kod przy obiekcie steruje cieniowaniem, a nie cieniem) zamiast ją osłabiać.

### K2. Cieniowanie naprawdę się obraca, ale „60–65°” to odczyt sondy całego obrazu, nie miara cieniowania

**Co mówi artykuł.** Streszczenie: „A small shift along the activation pattern of the light turns the shading of the object by 60° to 65°”. Liczba pochodzi z sondy pikselowej 64×64 na całym obrazie. Artykuł uzasadnia, że ta sonda „mostly reads the shading”, liczbą 59% wagi na komórkach obiektu. Ta liczba pochodzi jednak z rozkładu wag **innej sondy** (8×8, `reports/experiments/v2/maps/regions.md`), a nie z sondy 64×64, która mierzy wyjścia.

**Sprawdzenie (Kaggle CPU, `shading_measure.py`).** Tę samą sondę dopasowałem do wycinka samego obiektu (podłoga szara, bez cienia). Na renderach v2 wycinek daje 3,2° (kule i walce). Potem odczytałem nią wszystkie wyjścia testu przyczynowego i łatania. Ta miara jest szumna na obrazach z SDEdit (błąd wobec etykiety na wyjściu bez zmiany wynosi 35°). Prawdziwa zmiana światła o 90° (przebieg źródła w łataniu) daje w niej obrót 40–61° (mediana 65–67°), a nie 90°. Dlatego podaję ją obok tej kalibracji.

| Warunek | Sonda całego obrazu (artykuł) | Wycinek obiektu: średnia (mediana) |
|---|---:|---:|
| wzorzec ×1, bloki 4 / 10 / 18 | +65 / +60 / +62 | +52 (58) / +50 (63) / +51 (53) |
| wzorzec ×3, bloki 4 / 10 / 18 | +121 / +121 / +120 | +82 (111) / +71 (102) / +71 (97) |
| losowy wektor (norma ×3) | ok. 0 | od −7 do −1 |
| wymazanie LEACE | 0 | od −6 do +2 |
| prawdziwe światło +90° (przebieg źródła) | +88 / +93 | +61 (65) / +40 (67) |
| wagi sondy w skali normy tokenu, blok 18 | **+84** | **+11 (30)** |

**Wniosek.** Ta wątpliwość się **nie potwierdziła** co do istoty. Przesunięcie wzdłuż wzorca rzeczywiście obraca cieniowanie samego obiektu, a losowy wektor nie. W skali tej miary wzorzec ×1 zmienia cieniowanie mniej więcej tak samo jak prawdziwe światło obrócone o 90°. Do poprawy jest tylko sposób podania: „60–65°” to odczyt sondy całego obrazu, a nie kąt cieniowania. Artykuł powinien podać miarę cieniowania (wycinek obiektu) z kalibracją na przebiegu źródła albo na renderach v2rot. Lista kontrolna artykułu (punkt 8: „one of them specific to the cue in question”) sama tego wymaga, a dla cieniowania takiej miary nie było.

Wynik w bloku 18 jest ciekawym przykładem tezy artykułu: sonda całego obrazu czyta +84°, a obiekt obraca się tylko o ok. 11–30°. Duży krok wzdłuż wag sondy zmienia więc inne części obrazu (podłogę). To wzmacnia zdanie z dyskusji, że miara całego obrazu może wprowadzać w błąd. Artykuł tego wiersza w ogóle nie omawia (zob. W7).

### K3. Wymazanie LEACE nie usunęło informacji, którą czyta sonda. Wynik „no effect” nic nie mówi o użyciu kodu

**Co mówi artykuł.** „The LEACE eraser ... is fitted on the cell features of the train split. It removes all linear information about the azimuth from every token of the block.” Wynik: „The LEACE eraser in one block has no effect on any measure”. Wyjaśnienie: zaszumione wejście nadal pokazuje cień, więc późniejsze bloki odczytują światło ponownie.

**Podejrzenie z kodu.** `make_intervention_plan.py` dopasowuje *jeden* eraser do wektorów komórek, a każda komórka każdego obrazu jest osobną próbką. LEACE usuwa wtedy tylko tę część kodu, która jest taka sama na każdej pozycji. Sonda na siatce ma osobne wagi dla każdej komórki, więc może dalej czytać kod zależny od pozycji (np. „po której stronie obiektu jest jasno”).

**Sprawdzenie (Kaggle CPU, `erasure_check.py`, te same cechy i te same erasery co w planach).** Eraser jest afiniczny, więc wymazanie każdego tokena daje te same średnie komórek co wymazanie średnich. Dopasowałem nowe sondy do cech po wymazaniu.

| Blok, t | Sonda na siatce: przed / po | Sonda na średniej wszystkich tokenów (1×1): przed / po |
|---|---:|---:|
| 4, t = 400 (8×8) | 4,6° / **4,7°** | 8,5° / 46,8° |
| 10, t = 400 (8×8) | 4,4° / **4,5°** | 6,2° / 55,7° |
| 18, t = 400 (8×8) | 4,5° / **4,3°** | 8,2° / 55,5° |
| 5, t = 100 (4×4) | 2,6° / **2,9°** | 4,1° / 88,3° |

(Kule i walce, split test. Sześciany dają ten sam obraz. W pliku dla bloku 5 etykieta wiersza mówi „8x8”, ale siatka to 4×4, zgodnie z planem tego bloku.)

**Wniosek.** Ten problem jest realny. Eraser usunął wspólny kod (sonda 1×1 spada prawie do losowej), ale sonda na siatce czyta azymut po wymazaniu **tak samo dobrze jak przed**. W blokach, w których artykuł wymazywał, informacja liniowa o świetle została. Zdanie „removes all linear information about the azimuth from every token” jest fałszywe. Wynik „no effect” ma najprostsze wyjaśnienie: kod zależny od pozycji został na miejscu. Wyjaśnienie z artykułu (wejście z szumem) może też działać, ale nie jest potrzebne.

**Co zrobić.** Do wyboru:

- (a) tylko tekst: opisać eraser jako usunięcie wspólnego kodu, podać wynik tej kontroli i wycofać interpretację;
- (b) poprawić eksperyment: eraser zależny od pozycji (np. osobny LEACE na każdą komórkę w przestrzeni PCA albo rzutowanie na przestrzeń zerową sondy na siatce, jak INLP), sprawdzić tym samym skryptem, że sonda na siatce spada do poziomu losowego, i powtórzyć wymazanie na GPU (ok. 0,5 h).

Wariant (b) daje prawdziwy test użycia kodu (punkt 7 listy kontrolnej artykułu).

### K4. Pytanie z wstępu (model 3D czy statystyki 2D) zostaje bez odpowiedzi, a „ponad wejście” znaczy tylko „ponad liniową sondę na pikselach”

**Co mówi artykuł.** Wstęp pyta: „Do they build an internal model of the 3D scene, or do they only reproduce 2D image statistics?”. Zakończenie: „SANA and two other diffusion models carry the light direction in a form that the input does not offer”. README: „carries the light direction far beyond the pixels”.

**Dlaczego to luka.** Rozumowanie przyczynowo-skutkowe wygląda tak:

1. Kierunek światła da się odczytać z jasnej i ciemnej strony obiektu oraz z kierunku cienia. Obie wskazówki są wzorcami 2D. Wycinek samego obiektu daje na pikselach 4,4° (sekcja 6 artykułu), więc wejście *zawiera* tę informację.
2. Liniowa sonda na surowych pikselach nie umie jej znaleźć w dowolnym miejscu kadru. Każda nieliniowa, splotowa sieć wyuczona na zdjęciach prawdopodobnie umie. Losowa SANA już bije piksele (15,9° wobec 20,1°).
3. Najlepszy krok to zawsze t = 100, czyli prawie czysty obraz. Wtedy model działa głównie jako ekstraktor cech obrazu, a nie jako generator.
4. Wniosek: wynik RQ1 pokazuje, że **wytrenowane sieci dyfuzyjne mają liniowo odczytywalny i mało zależny od pozycji kod kierunku światła**. Nie pokazuje, że to „wiedza 3D”, ani że to cecha *modeli generatywnych*. Ten sam wynik dałby zwykły model rozpoznający obrazy, który ma tylko statystyki 2D.

Najmocniejszy argument w sprawie „3D” artykuł ma w części przyczynowej: przesunięcie kodu zmienia cieniowanie, ale nie rysuje nowego cienia rzucanego. To przemawia raczej *przeciw* spójnemu modelowi światła 3D. Artykuł tego nie mówi wprost.

**Co zrobić.**

- W tekście: wstęp i zakończenie muszą jasno powiedzieć, czego sonda nie rozstrzyga (2D czy 3D). Słowa „beyond the input” i „in a form that the input does not offer” trzeba zmienić na „beyond a linear read-out of the input”.
- Eksperyment (zalecany, ok. 1 h GPU): ta sama sonda na cechach modelu niegeneratywnego, np. DINOv2 albo CLIP ViT (oba z Hugging Face). Jeśli dadzą podobny wynik, zdanie „cecha modeli dyfuzyjnych” (STAN_PRAC) upada, ale artykuł zyskuje uczciwą odpowiedź. Jeśli dadzą wyraźnie gorszy, to jest to mocny, nowy wynik.

### K5. Regulamin: wkład członków zespołu i liczba osób

**Fakty z repozytorium.** Od 2026-10-01 jest 146 commitów i wszystkie mają jednego autora. Drugi członek zespołu ma 0 commitów w czasie SIWY (jego 14 commitów pochodzi z projektu ZZSN). Zespół ma 2 osoby, a regulamin wymaga 3. Worklog i design proposal odkładają pytanie do prowadzącego na tydzień 2.

**Regulamin.** „Brak wkładu albo istotnie mniejszy wkład członka zespołu do projektu skutkuje oceną −3 za projekt” (regulamin przedmiotu, § 3 pkt 11). „Za przejawy braku wkładu uznawane są w szczególności brak kontrybucji kodu do projektu (w postaci commitów) i brak pisemnych opracowań” (regulamin projektów, § 1 pkt 11).

**Ryzyko.** To jest największe ryzyko formalne projektu, większe niż każda usterka merytoryczna. Historia gita wprost pokazuje jedną osobę.

**Co zrobić.** To decyzja zespołu, nie poprawka techniczna. Zapytać prowadzącego jak najszybciej (rozmiar zespołu, kontynuacja ZZSN). Ustalić podział dalszej pracy tak, żeby wkład drugiej osoby był widoczny w commitach (np. testy, rozdział artykułu, film, prezentacja). W worklogu prowadzić wpisy z podziałem na osoby. Nie przepisywać historii gita.

---

## 2. Ważne: logika i spójność artykułu

### W1. „Four problems” i „each problem can explain the result without 3D knowledge”

Streszczenie wylicza cztery problemy: (1) piksele są tak samo dobre, (2) błąd szumu, (3) etykieta, (4) odblask. Punkt 1 to wynik kontroli, a nie usterka pipeline'u. Sekcja 4 ma pięć podsekcji i dalsze usterki: odwrócony kierunek azymutu w starym raporcie, dodatkowa lampa, zła kamera na rysunkach, alfa bez wpływu oraz **wybór konfiguracji na danych testowych**. Ta ostatnia jest poważna, a dostała pół zdania.

Wkład 1 mówi, że każdy problem „can explain the result without 3D knowledge”. Błąd szumu działał odwrotnie: ukrył informację (14,5° zamiast 4,8°). Ślepy czytelnik złapał to od razu (problem nr 3 na jego liście).

**Co zrobić.** Podzielić usterki na dwie grupy: „co sztucznie poprawiało wynik pikseli lub skrót” (etykieta, odblask, dodatkowa lampa, wybór na teście) oraz „co psuło odczyt z modelu” (szum, alfa). Policzyć je raz i tak samo w streszczeniu, we wkładach i w sekcji 4.

### W2. Opis sondy nie zgadza się z kodem: nie ma „standard scores”

Artykuł (sekcja 5) i README: „PCA, standard scores and a Ridge regression”. W kodzie `ProbeConfig.whiten = False` (`src/dd3d/probe.py:28`), a `StandardScaler(with_std=False)` tylko centruje wyniki PCA. Żaden skrypt wyników nie włącza `whiten`. Siatka alfa jest za to skalowana do wariancji PCA (`alpha_grid`), i to jest właściwa poprawka błędu ZZSN. Zdanie z sekcji 4.5 („because the PCA scores were not scaled”) też trzeba więc przepisać: przyczyną była siatka alfa daleko poniżej wartości własnych.

Docstring `scripts/experiments/pixel_cues.py` też mówi „standard scores”. Plik `reports/experiments/pixel_cues/results_whitened.md` to inna wersja, której artykuł nie używa.

### W3. Tabela 1 ma liczby, których nie ma w plikach wyników

Tabela 1: piksele 64×64 dają 14,1° / 6,3°, a piksele 32×32 dają 13,8° / 6,5°. Tekst sekcji 5: „11,5° against 13,0°” na teście.

Pliki wyników z obecnego kodu (`reports/experiments/v1/scan/scan.csv`, `v1/best/compare.md`):

| Siatka | Test | Sześciany |
|---|---:|---:|
| 64×64 | 11,84 | 14,32 |
| 32×32 | 11,45 | 14,09 |

Liczby 11,45 i 14,09 z worklogu („F1: raw 64x64 pixels”) to dokładnie dzisiejsze 32×32. Najpewniej w przeglądzie z 12.10 rozdzielczość była opisana źle albo liczona innym kodem. Wartości 13,8 nie ma w żadnym pliku. Tabela 3 (14,3) i Tabela 1 (14,1) podają dwie różne liczby dla tej samej kontroli. Artykuł mówi: „Every number in this paper comes from a script in the repository”. Dla Tabeli 1 to nie jest prawda.

**Co zrobić.** Przepisać Tabelę 1 i zdanie z sekcji 5 na liczby z `v1/scan/scan.csv`. Wniosek się nie zmienia, bo piksele nadal są równe SANA.

### W4. Warianty „bez cienia” i „sam cień” nie rozdzielają wskazówek do końca

`tools/blender/render_spec.py` ustawia `cycles_visibility.shadow = False` (v2rns) albo `cycles_visibility.camera = False` (v2rso). W obu wariantach obiekt dalej **odbija fioletowe światło na podłogę**. Artykuł sam pokazuje, że ta wskazówka jest silna: na v2 bez cienia sama podłoga daje 4,6° (sekcja 6, „Weak cues remain”).

Zdanie „So the model encodes the light from the shading of the object and from the cast shadow, each one alone” jest więc za mocne. v2rns to cieniowanie obiektu plus odbite światło na podłodze. v2rso to cień plus odbite światło niewidocznego obiektu.

**Co zrobić.** Zmienić opis na „without the cast shadow” i „without the visible object”. Dopisać wprost, że w obu wariantach zostaje światło odbite. Opcjonalnie wyrenderować wariant z `cycles_visibility.diffuse = False` dla obiektu (CPU, render ok. 6 h na zbiór, i 1 GPU h ekstrakcji).

### W5. „The representation does not depend on the position” jest za mocne

Dowody: sonda 1×1 (średnia wszystkich tokenów) dopasowana na v2 daje na v2r 9,0° (kule i walce) i 17,6° (sześciany). Ta sama sonda dopasowana w rozkładzie (na v2r) daje 5,9° i 13,2°. Dla porównania, najlepsze piksele w rozkładzie dają 20,1° na sześcianach. Streszczenie podaje tylko 9,0°. STAN_PRAC odpowiada na pytanie o zależność od pozycji jednym słowem: „Nie.”

**Co zrobić.** Zamienić na „largely independent of the position”. W streszczeniu podać obie liczby (9,0° i 17,6°) albo dopisać „on spheres and cylinders”. Brakuje też kontroli 1×1 dla losowych wag i latentu (zauważył to ślepy czytelnik).

### W6. Wniosek z przesunięcia we wszystkich blokach miesza projekt interwencji z kodem modelu

Artykuł: „So the linear code of the light changes the brightness of large image regions, and the model renders this change as patches.” Ale wektor δ jest stały w każdej komórce siatki 4×4. Prostokąty leżą dokładnie na siatce przesunięcia. Najprostsze wyjaśnienie przyczynowe brzmi więc: to artefakt kształtu interwencji. Nie jest to cecha kodu światła. Ten sam zarzut (interwencja stała w komórce nie może narysować ostrego kształtu) dotyczy w słabszej formie wszystkich wniosków typu „shift does not draw a new crisp shadow”. Zauważył to też ślepy czytelnik.

**Co zrobić.** Napisać wprost, że prostokąty wynikają z siatki interwencji. Wniosek o braku nowego cienia ograniczyć do „shifts that are constant in each grid cell”.

### W7. Niezgłoszony wiersz: wagi sondy w bloku 18 dają +84° w sondzie całego obrazu

Tabela 5: `probe_r1` w bloku 18 daje +84° [+80, +89], a losowy wektor tej samej normy +14° [−3, +32]. Zmiana pikseli wynosi 40 i 49 poziomów szarości (`causal/b18/results.csv`), więc obraz *nie* jest zniszczony tak, jak w blokach 4 i 10 (148 i 132). Podpis Tabeli 5 mówi, że te wiersze nie mają znaczenia tylko w blokach 4 i 10, a tekst bloku 18 nie omawia. Kolumna „pixels” jest średnią z bloków, więc czytelnik nie widzi tej różnicy.

Sprawdzenie z K2 pokazuje, że cieniowanie obiektu obraca się tu tylko o ok. 11° (mediana 30°). Zdanie z dyskusji („The probe weights are a poor steering handle”) się więc broni, ale artykuł musi pokazać, dlaczego +84° nie przeczy tej tezie.

**Co zrobić.** Podać zmianę pikseli dla każdego bloku. W jednym zdaniu omówić blok 18 jako przykład miary całego obrazu, którą zmiana podłogi wprowadza w błąd.

### W8. Kontrola losowa ma normę kroku ×3, a nie ×1, i to zmienia odczyt miary „kept share”

Plany (`reports/experiments/plans/*/plan.json`) dają `random_pattern` normę równą `pattern_x3`. Wynika to też z docstringu `make_intervention_plan.py`. Artykuł mówi „a random vector with the same norm in each cell”, a streszczenie „a random shift of the same size does nothing”. Wynik jest dzięki temu *mocniejszy* (losowy wektor 3× większy nic nie robi), ale trzeba to napisać.

Druga strona tej samej liczby: losowy wektor zostawia 82–88% ostrego cienia, a wzorzec ×1 tylko 68–80%. Mały krok wzdłuż wzorca *usuwa* więc trochę cienia ponad poziom szumu miary. Zdanie „The crisp shadow stays” trzeba złagodzić do „most of the crisp shadow stays”.

### W9. Krok t = 400 dla planów bez uzasadnienia, a Ograniczenia są nieaktualne

Plany bloków 4, 10, 18 używają cech z t = 400, a sondy czytają najlepiej przy t = 100. Tekst nie mówi dlaczego (ślepy czytelnik pytał o to dwa razy). Ograniczenia mówią „The shifts act in one block at a time and use directions from the features at t = 400”, a artykuł ma już test bloku 5 z t = 100 i test wszystkich bloków naraz. Ograniczenia mówią też „60 images”, a test wszystkich bloków ma 48.

### W10. Rysunki pokazują inne bazy niż tabela

- Rysunek 4 (skan) ma linię pikseli 64×64 (na v2r: 25,8° na sześcianach), a Tabela 4 porównuje z najlepszą siatką (20,1°). Na rysunku SANA wygląda lepiej niż w tabeli.
- Rysunek 3 ma w tytule „v2: pixels 16x16, cubes az MAE 40.2 deg”, a tekst podaje dla v2 13,7°. Rysunek nie ma skali kolorów.
- Rysunek 7: podpisy nad obrazami są nieczytelne w druku (ok. 3 pt).
- Tabela 5 używa nazw z kodu (`probe_exact`, `probe_r1`, `random_probe`), a tekst innych słów. Nie ma też kolumny „kept”, choć tekst cytuje te liczby. Pusta komórka (`--` w bloku 10) nie ma wyjaśnienia: miara cienia nie dała tam żadnego wyniku (`shadow_n = 0`).

### W11. Audytowane badanie nie jest nigdzie cytowane

Czytelnik nie może sprawdzić żadnego zarzutu wobec „the original study” (rysunki, kierunek azymutu). Przy podwójnie ślepej recenzji pracę własną cytuje się w trzeciej osobie albo dołącza jako anonimowy materiał dodatkowy. Trzeba wybrać jeden z tych sposobów i napisać to w tekście.

### W12. Długość

Tekst główny kończy się na stronie 15 (ok. 14,5 strony). `docs/article/venues.md` zaleca do 12 stron dla zwykłego trybu recenzji TMLR.

---

## 3. Rozbieżności liczb (artykuł wobec plików wyników)

| Miejsce w artykule | Artykuł | Pliki wyników | Uwagi |
|---|---|---|---|
| Sekcja 9, „The best layer” | „reads best in block 5 at t = 100 (test MAE 2.6° on v2)” | Na v2 walidacja wybiera blok 4, siatkę 8×8 (2,2°). Blok 5 (4×4) ma 2,61° i nie jest najlepszy. Blok 5 jest najlepszy na **v2r**. | `v2/scan/scan.csv`, `rq1.tex` |
| Sekcja 9, blok 5 | „At ×2 the shading reaches the target in all 60 images” | `toward_target_share = 0.95`, czyli 57 z 60 w granicy ±45° | `causal_b05/b05t100/results.csv`. Do tego to odczyt sondy całego obrazu, nie cieniowania (zob. K2). |
| Sekcja 9, miary | „with two measures” | W wypunktowaniu są trzy miary | — |
| Sekcja 8, kontrola | „shuffled labels give 83° to 93° for every source” | Średnie po 3 tasowaniach: od 81,6° do 97,7°; pojedyncze przebiegi od 74° do 103° | `*/best/control.csv`. Wniosek bez zmian. |
| Sekcja 7, v2r | „the position does not correlate with the light (\|r\| < 0.01)” | \|r\| do 0,02 (y z cos az) | `data/v2r/labels.csv`. Wniosek bez zmian. |
| Streszczenie i sekcja 4.4 | „35 times the expected norm” | √(24,5² + 25,5²) ≈ 35,4 to norma względem szumu o normie 1. Wobec oczekiwanego wejścia (0,8·x0 + 0,2·ε, norma ok. 0,82) to ok. 43 razy. | Lepiej podać same współczynniki albo napisać „about 40 times”. |
| Sekcja 4.4 | „11,100 at t = 200 and 720 at t = 800” obok „718” (flow, t = 200) | Liczby poprawne, ale zbieżność 720 i 718 myli czytelnika | Dopisać, że 720 to stara reguła przy t = 800. |
| Tabela 1 i sekcja 5 | 14,1 / 13,8 / 11,5 / 13,0 | 14,32 (64) / 14,09 (32) / 11,84 / 11,45 | Zob. W3 |
| STAN_PRAC | „format TMLR, 16 stron” | PDF ma 17 stron | — |
| Sekcja 9, kalibracja miary ciemnej plamy | „median error 6.4° (10.3° on cubes)” | 6,38° na teście (zgodne), **8,68° na sześcianach**; brak odczytu w 0,5% i 4,6% obrazów | Brak pliku wyników w repo. Przeliczone w `kaggle/measures/`. |
| Sekcja 9, kalibracja „kept share” | „1.00 for the same image and 0.00 to 0.01 when the light turns by 90°” | 1,00 (zgodne); dla par v2 / v2rot **mediana 0,07, średnia 0,09** | Dolny poziom miary to ok. 0,09, a nie 0. Oba cienie zaczynają się przy podstawie obiektu. |
| Sekcja 9, sonda pikselowa | „On v2 it gives 59% of its weight to the object cells” | 59% to sonda 8×8 (`v2/maps/regions.md`). Wyjścia mierzy sonda 64×64. | Zob. K2 |

Wszystkie pozostałe liczby z Tabel 4, 5 i 6, z akapitów o transferze, mapach i łataniu sprawdziłem i są zgodne z plikami (do zaokrąglenia).

---

## 4. Czytelność artykułu (wyniki ślepego czytelnika)

Czytelnik zrozumiał główną linię: stary wynik nie dowodzi niczego, nowe dane i kontrole, kod światła istnieje i steruje cieniowaniem, ale nie cieniem. Jego streszczenie wniosków (część A notatek) jest wierne. To dobry znak. Najważniejsze bariery dla kogoś bez dostępu do kodu:

1. **Słowa z wieloma znaczeniami albo bez definicji:** „code” (raz reprezentacja, raz kod programu, w jednym streszczeniu), „activation pattern” (definicja dopiero na str. 9), „natural size”, „dose”, „handle”, „turns the shading”, „key lamp”, „the first variant” (v2ns nie ma nazwy w artykule).
2. **Nazwy zbiorów i warunków:** v1, v2, v2pos (w tekście tylko „300 images”), v2ns, v2r, v2rns, v2rso, v2rot. Brakuje jednej tabeli zbiorów na początku sekcji 6.
3. **Miary wyjścia:** jedna miara ma cztery nazwy („shadow measure”, „dark-region measure”, „dark region”, „shadow”). Brakuje progów (20 i 40 poziomów szarości; w artykule jest tylko 40).
4. **Odtwarzalność bez kodu:** brakuje rozdzielczości i bloków SD 1.5 / SDXL, sposobu losowania wag, skalowania 512 → 1024, liczby prób bootstrap (2000), siatki alfa, wyboru 60 obrazów, budowy masek obszarów w łataniu, wzoru na azymut cienia lampy słonecznej (Dodatek A ma tylko lampę punktową v1).
5. **Język:** krótkie zdania pomagają w opisie danych. W dyskusji brakuje jednak spójników przyczynowych („because”, „therefore”), więc wnioski czyta się jak listę faktów. Kilka zdań jest tak skróconych, że traci się sens (np. „the wrong noise hid the information” bez wyjaśnienia mechanizmu w streszczeniu).

Pełna lista z numerami stron i cytatami: [`czytelnik_artykulu.md`](czytelnik_artykulu.md), część D.

**Propozycja struktury:** (a) na końcu wstępu jedno zdanie o tym, co znaczy „code” w tym tekście; (b) tabela zbiorów danych na początku sekcji 6; (c) tabela miar wyjścia z progami na początku sekcji 9; (d) nazwy warunków w tabelach takie same jak w tekście; (e) dodatek „Experimental details” z parametrami z punktu 4.

---

## 5. Dokumentacja

Ślepy czytelnik dokumentacji zrozumiał projekt, ale nie umiałby odtworzyć głównego wyniku. Najważniejsze problemy (potwierdzam każdy w plikach):

1. **Za dużo źródeł prawdy.** Główne wyniki są w pięciu miejscach (README, STAN_PRAC, PODSUMOWANIE §0, worklog F11–F18, rq1.md). Ustalenia audytu też w pięciu. STAN_PRAC odsyła po szczegóły do PODSUMOWANIE, a PODSUMOWANIE do STAN_PRAC.
2. **Nieaktualne treści, które wyglądają na bieżące:**
   - PODSUMOWANIE §10: „Na takim zbiorze żaden model nie pokaże przewagi nad pikselami” (v2). Tymczasem `rq1` daje SANA 7,3° wobec 13,7° na sześcianach v2.
   - PODSUMOWANIE §11: artykuł „szkic, brakuje wyników z GPU”. §12: „Wynik negatywny też jest tu wartościowy”.
   - worklog i design proposal: „We did not send them yet” (issue do diffusers), a PODSUMOWANIE podaje numer zgłoszenia.
   - design proposal: polecenia CLI `baseline`, `render-spec`, `erase`, `steer`, których nie ma. Plan TensorBoard, którego eksperymenty nie używają. Harmonogram na 18 tygodni, a prawie wszystko zrobiono w tygodniu 1.
   - `usage.md`: ścieżka Athena / SLURM / `data/features` lokalnie, sprzeczna z zasadą „dane tylko na Kaggle”. Opisane polecenia nie są tymi, które dały liczby z artykułu (te dały skrypty `scripts/experiments/*` w zadaniach Kaggle).
3. **Brak ścieżki odtworzenia.** Nie ma dokumentu „od zera do Tabeli 4”: który job, który spec, który skrypt i który plik wyniku.
4. **PODSUMOWANIE jest pisane głosem kogoś spoza zespołu** („dopasowałem”, „wasz model”, „planowaliście”). Ma też notatki osobiste (usypianie WSL, Credential Manager). Prowadzący czytający ten plik uzna, że autorem jest ktoś z zewnątrz.
5. **`rq1.md` nie ma legendy** (jednostki, co znaczy „Test”, co znaczy „Block, t, grid”, dlaczego tylko azymut).
6. **KAGGLE_DATA.md** jest po polsku i nie ma wstępu. Brakuje w nim jobów `causal_b05` i `causal_multi`, choć STAN_PRAC mówi o „liście wszystkich jobów”.
7. **worklog** ma tylko tydzień 1 i pusty szablon „Week N” na końcu. Nie zapisuje odchyleń od harmonogramu, a regulamin tego wymaga.
8. `data/examples/README.md`: „the rows ... differ from the v2 row in one property only”. To nieprawda dla v2rns i v2rso, które różnią się od v2 dwiema cechami (pozycja i cień albo pozycja i widoczność obiektu).

### Propozycja: jedno źródło prawdy

Zgadzam się z Twoją oceną: STAN_PRAC i PODSUMOWANIE to pliki chwilowe. Proponuję taki podział:

| Plik | Rola | Co z nim zrobić |
|---|---|---|
| `README.md` | Wejście: pytanie, 5 zdań wyników, tabela główna, linki | Zostaje. Każda liczba z linkiem do pliku w `reports/`. |
| `docs/RESULTS.md` (nowy) | **Jedyne źródło prawdy** o wynikach: tabela zbiorów, tabele RQ1–RQ4 z legendą, test przyczynowy, łatanie, ograniczenia. Każda liczba ma ścieżkę do pliku wyniku i nazwę joba. | Zbudować z STAN_PRAC (struktura, mermaid) i z `rq1.md`. Najlepiej generować tabele skryptem z `reports/`, jak `rq1_table.py`. |
| `docs/REPRODUCE.md` (nowy) albo nowa sekcja `usage.md` | Ścieżka od danych do każdej tabeli i rysunku | Zastąpić stare sekcje Athena/SLURM. |
| `docs/audit/labels_and_arrows.md` | Zamrożony audyt ZZSN | Zostaje, z dopiskiem „stan na 13.10”. |
| `docs/design_proposal.md` | Zamrożony plan | Zostaje bez zmian treści (to dokument z etapu), z banerem „plan z 13.10; stan: RESULTS.md, odchylenia: worklog”. |
| `docs/worklog.md` | Historia tygodni, odchylenia od planu, podział pracy na osoby | Dopisać odchylenia. Usunąć szablon. |
| `docs/STAN_PRAC.md`, `docs/PODSUMOWANIE.md` | Pliki chwilowe | Usunąć (historia zostaje w gicie). Przedtem przenieść do RESULTS.md to, czego nie ma gdzie indziej, np. wyjaśnienie błędu diffusers (§9 PODSUMOWANIE) do `tools/diffusers_upstream/`. |
| `docs/KAGGLE_RUNBOOK.md`, `docs/KAGGLE_DATA.md` | Operacje | Zostają. Ujednolicić język i dodać brakujące joby. |

Do rozstrzygnięcia przez Ciebie: język `RESULTS.md`. Regulamin nie narzuca języka. Reszta repozytorium jest po angielsku.

---

## 6. Kod i testy

**Poprawne (sprawdziłem logikę, nie tylko styl):**

- `dd3d.noise` i `dd3d.extract`: reguła przepływu `(1 − t/1000)x0 + (t/1000)ε` z krokiem t podanym do transformera w tej samej skali. Maska uwagi pustego promptu, ziarno na obraz, pooling w kolejności wierszy.
- `dd3d.probe`: PCA, skaler i siatka alfa są dopasowane tylko na train, a alfa jest wybierana na val. `probe_direction` i `spatial_contributions` są dokładne dla modelu liniowego.
- `dd3d.metrics`: błąd kołowy i bootstrap są poprawne (2000 prób, percentyle).
- `dd3d.select`: blok, t i siatkę wybiera tylko val.
- `dd3d.intervene`: krok Eulera `x + (σ' − σ)·v` ma dobry znak dla `flow_prediction`. Łatanie zapisuje źródło i podmienia tokeny w tym samym kroku. Kontrola pozytywna (wszystkie tokeny) jest dobrze uzasadniona: SANA nie ma połączeń między blokami poza strumieniem tokenów.

**Problemy:**

1. `ProbeConfig.whiten` i opis „standard scores” (zob. W2).
2. LEACE jest dopasowany na komórkach ze wszystkich pozycji naraz (`make_intervention_plan.py:105–116`), więc usuwa tylko wspólny kod. Sprawdzenie na Kaggle potwierdza, że sonda na siatce czyta azymut po wymazaniu tak samo jak przed (zob. K3).
3. `analyze_patching.py` liczy udział cienia źródła w całym obrazie, razem z podmienionymi komórkami. Przez to miara myli kopię pikseli z przeniesieniem cienia (zob. K1).
4. **Brak testów kodu, na którym stoją główne liczby:** `probe.py` (wybór alfa tylko na val, dokładność `spatial_contributions`), `cues.py` (`shadow_azimuth`, `shadow_kept`, `object_mask`), `select.py`. Testy integralności danych sprawdzają tylko podziały v1, a nie v2*. Regulamin wymaga testów „adekwatnych do tematu”. Obecne 42 testy przechodzą, ale nie pokrywają metod pomiaru. Dwa błędy z K1 i K3 dałoby się złapać prostym testem tych miar na sztucznych danych.
5. Stałe ścieżki maszyny: `tools/kaggle/kjob.py:22` (`/home/flang/tools/kag`), `tools/diffusers_upstream/NOTES.md`, runbook. Nikt inny nie uruchomi kolejki bez zmian w kodzie.
6. Progi miar cienia (20 i 40 poziomów szarości, minimum 30 i 50 pikseli) są stałymi w kodzie, bez testu wrażliwości. Poziom „bez efektu” dla `shadow_kept` wynosi 0,82–0,88, a nie 1,0, więc miara jest dość szumna przy brzegu cienia.

Lint (`ruff check`, `ruff format --check`) i testy (`pytest`, 42 testy) przechodzą.

---

## 7. Regulamin SIWY: lista kontrolna

| Wymaganie | Stan | Uwagi |
|---|---|---|
| Zespół 3-osobowy | **NIE** | 2 osoby. Pytanie do prowadzącego wciąż otwarte. |
| Udokumentowany wkład każdego członka (commity, opracowania) | **RYZYKO** | Zob. K5 |
| Design proposal: harmonogram tygodniowy z datami, bibliografia, eksperymenty, funkcjonalność, stack, zasoby obliczeniowe | TAK | `docs/design_proposal.md` jest kompletny, ale plan i stan się rozjechały (zob. sekcja 5). |
| Analiza literatury w tabeli (linki, komentarz, kod, modele, metryki, zasoby) | TAK | `docs/literature.md` jest wzorowy. |
| Bieżące dokumentowanie pracy i odchyleń od harmonogramu („weekly standup”) | CZĘŚCIOWO | Worklog ma jeden tydzień i nie opisuje odchyleń. |
| Prototyp, środowisko eksperymentalne | TAK | Do pokazania na etapie prototypu |
| Dokumentacja i instrukcja użytkowania | CZĘŚCIOWO | `usage.md` jest nieaktualny wobec Kaggle (zob. sekcja 5) |
| Testy adekwatne do projektu | CZĘŚCIOWO | Zob. sekcja 6, pkt 3 |
| Film 3–5 min | NIE (jeszcze) | W planie na tydzień 15 |
| Artykuł naukowy | TAK (szkic) | Zob. sekcje 1–4. „Proces zgłaszania powinien zostać rozpoczęty w trakcie semestru” daje ocenę 5. |
| Autoformatter i linter | TAK | ruff, pre-commit |
| Środowisko wirtualne | TAK | `.venv`, `pyproject.toml` |
| Skryptowe budowanie, testy, uruchamianie | TAK | `Makefile`, `dd3d` (typer), joby Kaggle |
| PEP8 | TAK | ruff, linia 110 |
| Conventional commits | PRAWIE | 6 commitów od 1.10 nie ma prefiksu (np. „Remove the course from the README”) |
| Niemutowalność danych | TAK | Etykiety i podziały w gicie, obrazy na Kaggle |
| Struktura projektu | TAK | `src/` layout, `configs/`, `scripts/`, `reports/` |
| Śledzenie eksperymentów (TensorBoard, W&B) | **CZĘŚCIOWO** | TensorBoard jest tylko opcją w `dd3d probe`. Eksperymenty z artykułu go nie używają. Śledzeniem są w praktyce specyfikacje jobów z commitem i pliki w `reports/`. Warto to nazwać wprost w dokumentacji albo dodać lekkie logowanie (np. TensorBoard w `feature_scan.py`). |
| Konfiguracja oddzielona od kodu | TAK | `configs/`, `configs/kaggle/*.json`. Domyślne wartości sondy i progi miar są w kodzie. |
| Eksport kodu z notatników | TAK | Jedyny notatnik jest w `legacy/` |
| Frameworki odpowiednie (Hugging Face) | TAK | diffusers, transformers |
| Kontynuacja projektu z innego przedmiotu | **DO USTALENIA** | Zgłosić prowadzącemu, co jest nowe (design proposal §1 to dobrze opisuje) |

---

## 8. Porządek w repozytorium

1. **Ślad narzędzia AI w kodzie:** `tools/viewer/template.html:2` ma znacznik `<meta name="claude:viewport-fit-sides" ...>`. Do usunięcia.
2. **Głos autora z zewnątrz w PODSUMOWANIE** (zob. sekcja 5, pkt 4).
3. **Anonimowość materiału dodatkowego dla TMLR.** Logi w `reports/experiments/logs/` i szablony w `tools/kaggle/templates/` mają adres GitHub z nazwiskiem. `README.md` ma sekcję „Authors”. `docs/` ma PDF-y ZZSN. Do zgłoszenia trzeba zrobić anonimową kopię (np. anonymous.4open.science) albo wyczyszczone archiwum.
4. Szablon „Week N” na końcu worklogu.
5. `legacy/` i `probe_exp/` (wyniki ZZSN) są opisane w README i potrzebne do audytu. Mogą zostać.
6. Plik `reports/experiments/pixel_cues/results_whitened.md` nie jest używany przez artykuł ani dokumentację. Do usunięcia albo opisania.

---

## 9. Plan naprawy (kolejność)

| # | Zadanie | Typ | Wysiłek |
|---|---|---|---|
| 1 | Zapytać prowadzącego o zespół i kontynuację ZZSN. Ustalić widoczny wkład drugiej osoby. | organizacja | od razu |
| 2 | Łatanie (K1): poprawić wnioski w streszczeniu, wynikach, zakończeniu, README i worklogu. Do analizy dodać miarę „poza obszarem podmiany”. Poprawić Rysunek 8. | tekst + CPU | 2–3 h |
| 3 | Wymazanie (K3): wariant (a) w tekście albo (b) eraser zależny od pozycji, sprawdzenie i powtórka na GPU | tekst, albo CPU + GPU ok. 0,5 h | 1 h albo 1 dzień |
| 4 | Cieniowanie (K2): dodać miarę na wycinku obiektu z kalibracją na przebiegu źródła; poprawić „60–65°”, blok 18 (W7), „all 60 images” | CPU + tekst | 2 h |
| 5 | Poprawić rozbieżności liczb (sekcja 3) i Tabelę 1 (W3) | tekst | 1 h |
| 6 | Poprawić opis sondy (W2), listę usterek (W1), zakres wniosków (W4, W5, W6, W7, W8) i Ograniczenia (W9) | tekst | 2–3 h |
| 7 | Przeramować wstęp i zakończenie wokół pytania 2D / 3D (K4) | tekst | 1–2 h |
| 8 | Opcjonalnie: baza DINOv2 lub CLIP na v2r (K4) | GPU ok. 1 h + CPU | 0,5 dnia |
| 9 | Tabela zbiorów, tabela miar, dodatek ze szczegółami, nazwy warunków, rysunki (sekcja 4, W10) | tekst i rysunki | 0,5 dnia |
| 10 | Skrócić tekst główny do 12 stron (część do dodatku) | tekst | 0,5 dnia |
| 11 | Jedno źródło prawdy w dokumentacji (sekcja 5), usunąć STAN_PRAC i PODSUMOWANIE, nowy `REPRODUCE` | dokumentacja | 0,5 dnia |
| 12 | Testy `probe.py`, `cues.py`, `select.py`, integralność podziałów v2* | kod | 2–3 h |
| 13 | Porządek: znacznik w `template.html`, szablon worklogu, nieużywany plik wyników, ścieżki maszyny | kod | 30 min |
| 14 | Anonimowa kopia repozytorium do zgłoszenia | organizacja | 1 h |

---

## 10. Co zostało dodane na tej gałęzi

- `docs/audit/validation_2026-10-17/RAPORT.md` (ten plik), `czytelnik_artykulu.md`, `czytelnik_dokumentacji.md`.
- `scripts/audit/validation/patch_copy_check.py`, `shading_measure.py`, `erasure_check.py`, `measure_check.py`: skrypty kontrolne. Każdy przeszedł test na sztucznych danych przed uruchomieniem na Kaggle.
- `configs/kaggle/audit_validation.json`, `audit_erasure.json`, `audit_measures.json`: specyfikacje jobów `dd3d-analyze-audit-validation`, `dd3d-analyze-audit-erasure` i `dd3d-analyze-audit-measures` (CPU, konto z runbooka, bez limitu GPU).
- `docs/audit/validation_2026-10-17/kaggle/`: małe pliki wyników tych jobów i ich logi (`log_*.txt`).
