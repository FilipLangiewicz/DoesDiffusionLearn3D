# Notatki czytelnika: "What Does a Diffusion Transformer Know about the Light Direction? An Audit of a Probe Study, with Controls and Interventions"

## Uwaga o procedurze (ważne dla oceny tych notatek)

Narzędzie Read nie pozwoliło czytać PDF po stronach (pierwsze wywołanie z parametrem pages zwróciło błąd: brak pdftoppm). Drugie wywołanie, bez parametru pages, zwróciło CAŁY PDF (17 stron) naraz. Nie mogłem więc czytać dosłownie "po 1-2 strony i zapisywać notatek przed dalszą lekturą". Notatki poniżej pisałem porcjami, z perspektywy każdej porcji, ale wiedza o dalszych stronach była już dostępna. Dlatego:
- w punkcie "zrozumienie" staram się opisywać to, co wynika z danej porcji, ale mogłem nieświadomie korzystać z późniejszego kontekstu;
- odpowiedzi na wcześniejsze pytania oznaczam jako "odpowiedź dopiero na str. M" na podstawie faktycznego położenia w tekście, co jest wiarygodne.
Rysunki widziałem w postaci stron wyrenderowanych; opisy rysunków dotyczą tego, co widać w tej skali. Gdzie nie jestem pewien szczegółu na miniaturach, piszę to wprost.

---

## Porcja 1: str. 1-2 (streszczenie, wstęp, wkład, prace pokrewne, początek sekcji 3)

**1. Zrozumienie.** Artykuł jest "audytem" wcześniejszego badania. To badanie użyło regresji grzbietowej (Ridge) na aktywacjach modelu SANA-1.6B i odczytywało azymut światła z błędem 14° na kostkach (kształt, którego sonda nie widziała w treningu). Autorzy twierdzą, że ten wynik nie dowodzi wiedzy o 3D, bo: piksele dają to samo, szum był wstrzyknięty błędnie, etykieta to pozycja lampy, a odblask na podłodze jest skrótem 2D. Potem budują nowe dane (jedno światło kierunkowe, losowa pozycja obiektu), na których liniowa sonda pikselowa zawodzi, a sonda na SANA, SD 1.5 i SDXL nie, oraz robią interwencje (przesunięcie aktywacji, podmiana tokenów), które mają pokazać, że model "używa" kodu światła. Na końcu ma być lista kontrolna dla badań z sondami. Ogólna intencja jest jasna. Nie wiem jeszcze, co autorzy rozumieją przez "code" i "activation pattern".

**2. Pojęcia bez definicji (lub przed definicją).**
- str. 1: "The same probe on pixels is equally good." Nie wiadomo jeszcze, jakie piksele (rozdzielczość, przetwarzanie).
- str. 1: "The noise injection called a library function too early" (jaka biblioteka, jaka funkcja, "too early" względem czego), "35 times the expected norm" (oczekiwanej przez kogo i jak liczonej).
- str. 1: "The code does not depend on the position". Słowo "code" znaczy tu "reprezentacja kierunku światła". W tym samym streszczeniu, w ostatnim zdaniu, "release the code" znaczy kod programu. To samo słowo, dwa znaczenia, na jednej stronie.
- str. 1: "A small shift along the activation pattern of the light" ("activation pattern" nieznane; wyjaśnione dopiero na str. 9).
- str. 1: "autoencoder latent" (który autoenkoder: DC-AE pojawia się dopiero na str. 2).
- str. 1: "one key lamp" ("key lamp" bez definicji; wyjaśnione na str. 4 jako lampa obszarowa).
- str. 1: "MAE of 14° for the light azimuth" (azymut względem czego, jak liczony przy zawijaniu kąta; nigdy nie zdefiniowane).
- str. 1: "A patch of the object tokens from a scene with another light" (patch = podmiana tokenów, wyjaśnione dopiero na str. 9-10).
- str. 1: "block 10", "timestep t = 200" przed opisem wyodrębniania cech (str. 2).
- str. 1: "Table 1" cytowana przed opisem danych; sama tabela jest dopiero na str. 3.

**3. Zdania czytane więcej niż raz / niezrozumiałe.**
- str. 1: "With a corrected pipeline, the same data give 4.8°: the wrong noise hid the information." Dwukropek zastępuje logikę. Nie rozumiem, jak błędny szum może pogorszyć odczyt, skoro w audycie reszta "problemów" ma wyjaśniać wynik zbyt dobry.
- str. 2: "Each problem can explain the result without 3D knowledge." Nie zgadza się to z błędem szumu (patrz niżej, pkt 4).
- str. 1: "turns the shading of the object by 60° to 65° towards a target of 90°". Nie wiem, co znaczy "obrócić cieniowanie obiektu" i jak to zmierzono.

**4. Twierdzenia bez uzasadnienia / mocniejsze niż dowody (na tym etapie).**
- str. 1-2: "Our audit finds four problems." Lista w streszczeniu: (1) sonda pikselowa równie dobra, (2) błąd szumu, (3) etykieta = pozycja lampy, (4) odblask na podłodze. Pierwszy punkt to wynik kontroli, nie błąd potoku. Ponadto sprzeczność logiczna: str. 2 mówi, że KAŻDY problem "can explain the result without 3D knowledge", a błąd szumu według streszczenia ZANIŻAŁ wynik ("the wrong noise hid the information"). Błąd, który ukrywa informację, nie może wyjaśnić dobrego wyniku.
- str. 1: "Interventions show that the model uses the code." Silne. Przesunięcie aktywacji pokazuje, że kierunek może sterować wyjściem, niekoniecznie że model go naturalnie używa. (Później, str. 11: wymazywanie LEACE nie daje żadnego efektu.)
- str. 1: "The code does not depend on the position: a probe on the mean of all tokens, fitted on centred objects, gives 9.0° on moved objects." Nie podano, że 9.0° dotyczy sfer i walców (dla kostek będzie 17.6°, str. 7).

**5. Pytania i oczekiwania.**
- Kim jest autor "first study"? Nie ma cytowania. Czytelnik nie może sprawdzić żadnego twierdzenia o oryginale (str. 1-2: "A first study", "The study concluded"). Oczekuję, że wyjaśnienie nastąpi, ale w całym artykule go nie ma (patrz str. 15-17).
- Jakie są te "four problems" dokładnie? Oczekuję sekcji 4 z czterema punktami.
- Co znaczy "the same data give 4.8°": ta sama sonda, te same zbiory, inny szum? Oczekuję tabeli w sekcji 6.
- Jak "kod" ma być niezależny od pozycji, skoro sonda była już trenowana na wycentrowanych obiektach?

**6. Rysunki i tabele.** Brak na str. 1-2.

**7. Sprzeczności.** Oryginalny wynik jest podany trzy razy inaczej: "14°" (streszczenie), "14.0°" (wstęp, str. 1), "14.5°" ("instead of 14.5°", wkład 2, str. 2). Dopiero na str. 6 widać, że 14.5 to własne odtworzenie z oryginalnym szumem, a 14.0 to oryginał. Wkład 4 mówi o "two independent measures", ale na str. 9 będą trzy punkty (patrz porcja 5).

**8. Język.** Krótkie zdania pomagają w definicjach (np. opis danych na str. 2). W streszczeniu szkodzą: serię stwierdzeń "The same probe on pixels is equally good. The noise injection called ... The label was ... A floor highlight gave ..." czyta się jak listę, ale czytelnik nie wie, że to cztery "problemy". Zdanie "the wrong noise hid the information" jest zbyt skrócone, żeby było zrozumiałe. "turns the shading" brzmi nienaturalnie. Brak spójników przyczynowych ("because", "therefore") w miejscach, gdzie logika jest nietrywialna.

---

## Porcja 2: str. 3-4 (oryginalne badanie, audyt, tabele 1-2, rysunek 1)

**1. Zrozumienie.** Sekcja 3 opisuje dane (zmodyfikowany CLEVR: jeden fioletowy obiekt, lampa o losowym azymucie i elewacji, 3000 obrazów: 1600/200/200 sfer i walców oraz 1000 kostek) i potok (DC-AE, szum, bloki 4/10/18 SANA, uśrednianie 4x4, PCA 256, Ridge). Sekcja 4 to audyt: autorzy odtworzyli dokładny model kamery (jitter nie był zapisany), stwierdzili, że kierunek azymutu jest odwrotnie opisany w oryginale, że etykieta to pozycja lampy (średnio 35° różnicy względem światła na obiekcie), że lampa obszarowa daje odblask na podłodze (skrót), że w scenie została włączona dodatkowa lampa słoneczna, że funkcja add_noise została wywołana przed set_timesteps (zły sigma, norma 35 razy za duża i odwrócony sygnał dla t=200), że rysunki oryginału miały złą kamerę, a α w Ridge nie miało wpływu (PCA bez skalowania, konfiguracja wybrana na teście).

**2. Pojęcia bez definicji.**
- str. 3: "a track-to constraint on the scene centre" (żargon Blendera), "scene files", "key lamp", "world light", "sun lamp", "area lamp" (str. 4), "object-mask IoU", "A 4 × 4 average pool gives 8 × 8 × 2240 features" (pooling o rozmiarze 4x4 daje siatkę 8x8; potem "4x4 grid" będzie znaczyć 4x4 komórki; patrz porcja 4).
- str. 3: "Mean of the train set" i "Random guess" w Tabeli 1 (jak liczone dla kąta; "89.9°" dla zgadywania da się zrozumieć jako MAE rozkładu jednostajnego, ale to nie jest powiedziane).
- str. 3: "Same probe on 64 x 64 RGB pixels (ours)". Nie wiadomo, czy z PCA, z jakim skalowaniem (sekcja 5 mówi o "standard scores after the PCA", czyli inna receptura niż opis sondy z sekcji 3).
- str. 4: "variance-preserving sigmas", "flow sigmas", "set_timesteps", "DPMSolverMultistepScheduler" (biblioteka nigdy nie nazwana), "timestep embedding says sigma = 0.2".
- Tabela 2: "a (original), b (original), a (flow), b (flow)", "least-squares fit x_t = a x_0 + b epsilon". Symbole x_0 i epsilon (czysty latent i szum gaussowski) nie są nigdzie zdefiniowane, nie wiadomo o jakiej skali.
- str. 3: "dir(L - p)" (kierunek jednostkowy).
- str. 3: "The original report described the opposite direction." (opposite do czego? czytelnik bez dostępu do oryginału nic z tego nie wynosi).

**3. Zdania czytane więcej niż raz / niejasne.**
- str. 4: "With the white world light, it gives 19% of the floor brightness." Co daje 19%: lampa słoneczna? razem ze światłem otoczenia? względem czego?
- str. 4: "Its shadow falls behind the object as the camera sees it, so the effect is a constant front light." Dlaczego cień za obiektem czyni z tego "stałe światło z przodu"? Stały cień nadal jest stałą wskazówką na obrazie.
- str. 4: "The function finds the index of t in the descending list and reads the sigma of step 999 - t." Skąd w Tabeli 2 sigma = 25.53 dla t=200? Z tekstu nie da się tego odtworzyć.
- str. 4: "35 times the expected norm". Z Tabeli 2: sqrt(24.53^2 + 25.53^2) = 35.4, ale "oczekiwana" norma sqrt(0.8^2 + 0.2^2) = 0.82 (przy założeniu jednostkowych norm x_0 i epsilon) daje stosunek ok. 43. Nie umiem odtworzyć liczby 35 bez dodatkowych założeń. Możliwe, że 35 to norma bezwzględna, nie stosunek.
- str. 4: "in block 4 the mean token norm is about 11,100 at t = 200 and 720 at t = 800". Czy to oba przypadki z szumem oryginalnym? Na str. 5 z szumem flow norma przy t=200 to 718. Podejrzana zbieżność 720 i 718; stosunek 11,100/718 to ok. 15, nie 35. Niejasne, co z czym porównano.
- str. 4: "Because the arrows had a fixed length from the object centre, about 18% of them ended outside the image and the plot library did not draw them." Bez widoku oryginalnego rysunku nie da się tego zweryfikować.
- str. 4: "The same order problem affects the variance-preserving case of four schedulers of the library." Które cztery? Jakiej biblioteki?
- str. 4: "Finally, the Ridge alpha had no effect ..., because the PCA scores were not scaled, and the final configuration was selected with test data." Jedno zdanie z dwoma osobnymi zarzutami, schowane na końcu akapitu o rysunkach; zarzut o wyborze konfiguracji na zbiorze testowym jest poważny, a wygląda jak dopisek.

**4. Twierdzenia bez dostatecznego uzasadnienia.**
- str. 3: "The fit reproduces the pixel position of all 3000 objects with a maximum error of 0.56 px". Jitter ma 3 parametry na obraz, a dopasowanie używa 3 liczb (piksel x, y i głębokość). Układ jest wyznaczony dokładnie, więc mały błąd nie jest walidacją. Prawdziwa walidacja to ponowne renderowanie (0.14/255, IoU 0.98), ale tekst tego tak nie przedstawia.
- str. 3: "So the probe reads the position of the lamp in the scene, and not the light that falls on the object." 14.0° względem etykiety i 35.5° względem światła lokalnego pokazuje, że sonda śledzi etykietę, ale nie odróżnia "pozycji lampy" od "skrótu z odblaskiem na podłodze" (to podobna wskazówka).
- str. 4: "The spot is larger and brighter than the cast shadow of the object, and it does not depend on the object shape." Brak liczb, tylko Rysunek 1.
- str. 4: "We reported the problem to the maintainers of the library, with a fix and tests (the link is withheld for anonymity)." Niesprawdzalne; biblioteka nienazwana.

**5. Pytania.** Jak zdefiniowano elewację "lokalną"? Jak zachowuje się oryginalny szum dla t=100 (nie ma w Tabeli 2)? Czy 35x faktycznie przeszkadza (odpowiedź: dopiero str. 6, porównanie 14.5° z 4.8°)? Jak liczby dla sfer i walców z sekcji 5 mają się do Tabeli 1 (ta ma tylko kostki)? Oczekuję sekcji 5 z kontrolami.

**6. Rysunki i tabele.**
- Tabela 1: zrozumiała, podpis mówi, że to kostki i że wiersze "ours" są własne. Kolumna elewacji nie jest nigdzie omówiona.
- Rysunek 1: podpis mówi, że żółta linia wskazuje lampę. Miniatury są małe, strzałki cienkie, brak oznaczenia, gdzie jest kamera, więc "front right / back left" z tekstu nie da się sprawdzić na obrazku (brak widoku z góry). Odblask na podłodze widać.
- Tabela 2: podpis wyjaśnia dopasowanie najmniejszych kwadratów, ale kolumn "flow" nie wyjaśnia (wzór 1-t/1000 i t/1000 podany dopiero na str. 5). Widać, że a+b=1 w każdym wierszu, tekst tego nie komentuje.

**7. Sprzeczności.** Podpis Rysunku 1: "The large floor highlight lies under the lamp"; tekst 4.3: "a bright spot on the floor between the lamp and the centre". Streszczenie mówi o czterech problemach, sekcja 4 ma pięć podsekcji i w 4.5 wylicza jeszcze kilka kolejnych błędów (rysunki, α, wybór konfiguracji na teście).

**8. Język.** Podsekcje 4.1 i 4.2 czytają się dobrze (krótkie zdania, liczby). Podsekcja 4.4 jest gęsta: kilka technicznych szczegółów biblioteki w kilku krótkich zdaniach, bez nazw, więc zdania są poprawne, ale informacyjnie puste dla kogoś, kto nie zna biblioteki. "The original report described the opposite direction" gubi sens: opposite do czego.

---

## Porcja 3: str. 5-6 (kontrole na oryginalnych danych, poprawiony potok, zbiory v2 i v2r)

**1. Zrozumienie.** Sekcja 5: sonda pikselowa jest tak dobra jak sonda SANA (14.1° vs 14.0° na kostkach). Warianty obrazu (pełny, tylko podłoga, tylko obiekt, pozycja odblasku) pokazują, że sama podłoga daje 9.4° na kostkach, lepiej niż pełny obraz; obiekt sam nie przenosi się na kostki. Wniosek: transfer na kostki w oryginale pochodzi z odblasku na podłodze. Sekcja 6: poprawione wyodrębnianie (szum flow, stałe ziarno, pięć kroków czasowych, 20 bloków), kontrola zgodności z oryginałem przez normy tokenów, kontrola precyzji float16. Po powtórzeniu eksperymentu: z szumem oryginalnym 14.5° (odtwarza oryginał), z szumem flow 4.8° (blok 7, t=100), lepiej niż piksele o 8.0°, DC-AE 8.9°, losowe wagi 11.9°. Wniosek: błędny szum ukrył informację. Sekcja 7 (zaczyna się na str. 6): zbiór v2 (lampa słoneczna, obiekt w centrum): sonda pikselowa prawie idealna na sferach i walcach (2.5°), ale 13.7° na kostkach; zbiór v2r (losowa pozycja obiektu): sonda pikselowa 25°, a przycięcie do obiektu daje 4.4°.

**2. Pojęcia bez definicji.**
- str. 5: "Image regions", "the brightest floor region (two numbers)", "highlight (2 numbers)".
- str. 5: "dd3d" (nazwa pakietu), "the attention mask of the empty prompt".
- str. 5: "Table 4, rows v1" (odwołanie do tabeli ze str. 8), "best pixel grid (16 x 16: 9.6 and 12.8)" (pojęcie "pixel grid" zdefiniowane dopiero na str. 7-8), "95% interval [7.1, 8.9]" (bootstrap sparowany zdefiniowany dopiero na str. 8), "block 7, t = 100" (timestep 100 nie był w oryginale: {200, 500, 800}).
- str. 6: "v2", "v2r", "sun lamp" (bez znajomości Blendera nie wiadomo, czym różni się od "area lamp"), "curved backdrop", "camera jitter", "rotations" ("same lights, sizes and rotations": rotacja obiektu to nowa zmienna), "|r| < 0.01" (korelacja czego z czym?), "[-3, 3]^2" (jednostki), "the largest object", "Cue variants".
- str. 6: "the DC-AE latent" jako cecha sondy: jak spłaszczona/uśredniona, nie wiadomo (w Tabeli 4 kolumna "Block, t, grid" ma myślnik).
- str. 6: "the UNets of Stable Diffusion 1.5 and SDXL with the noise rule of their own schedules" bez opisu (jaki VAE, rozdzielczość, bloki).

**3. Zdania czytane więcej niż raz / niejasne.**
- str. 6: "A model with a representation of the light that does not depend on the position must do better than 25 degrees on v2r." Dlaczego 25? Tabela 4 pokaże potem, że najlepsza siatka pikseli ma 20.1°, a SANA z losowymi wagami (15.9°) ten próg też spełnia, więc próg nie rozróżnia nauczonej reprezentacji od losowej cechy nieliniowej.
- str. 6: "Without the cast shadow (the first variant, object at the centre), the floor alone still predicts the azimuth (4.6; 10.2 on cubes)." "The first variant" jest wyjaśnione dopiero w następnym akapicie ("Cue variants"), a "object at the centre" nie pasuje do v2r (losowa pozycja). Nie wiem, na jakim zbiorze to policzono.
- str. 6: "The same pixel probe on 300 images with the object at random positions has an azimuth MAE of 69.0" (jakie 300 obrazów? v2r jest wprowadzone później, a później liczba dla v2 -> v2r to 71.2 i 70.4; to ten sam eksperyment czy inny?).
- str. 5: "because the object only adds shape-specific variance" (wyjaśnienie bez testu).

**4. Twierdzenia bez dostatecznego uzasadnienia.**
- str. 6: "close to chance (90)" dla 69.0° (i później 71.2°, 70.4°). To 20° poniżej poziomu losowego, nie "blisko".
- str. 6: "the floor alone gives an elevation MAE of 1.7 on cubes" i "(2)" bez tabeli.
- str. 6: nagłówek "A centred object makes the pixel control almost perfect" a w treści 13.7° na kostkach, czyli tyle co w v1. "Almost perfect" dotyczy tylko sfer i walców.
- str. 6: "The cue is the light that the purple object reflects onto the floor on its lit side." Wyjaśnienie fizyczne podane bez testu.
- str. 6: "Thus the original study did not fail because SANA lacks the information: the wrong noise hid it." Porównanie 14.5° vs 4.8° jest uczciwe (ten sam protokół wyboru), ale nowy protokół zawiera więcej opcji (t=100, 20 bloków) niż oryginał. To rozsądnie kontrolowane, więc przekonuje w dużej mierze; autorzy słusznie dodają, że v1 nadal ma odblask.

**5. Pytania.** Dlaczego pikselowy próg dla v2r to 25°, skoro Tabela 4 (str. 8) pokazuje 21.5/20.1 dla najlepszej siatki? (odpowiedź dopiero na str. 7-8: to różne rozdzielczości, ale tekst tego nie mówi). Odpowiedź na pytanie z str. 4 (czy szum faktycznie ma znaczenie): str. 6, wynik 14.5° vs 4.8°. Odpowiedź na pytanie ze str. 4 o α: początek sekcji 5 (standardowe skalowanie po PCA).

**6. Rysunki i tabele.**
- Rysunek 2: lewa część: trzy miniatury wariantów obrazu, małe. Prawa część: słupki w czterech kolorach (dwa niebieskie, dwa zielone: etykieta global/local razy test/test_cubes), legenda drobna. Słupek "object only, global, cubes" (44.3°) wychodzi poza oś. Opis "SANA probe on cubes (original)" nakłada się na słupki. Czytelne w głównych zarysach, ale trudno odczytać liczby (są w Tabeli 3).
- Tabela 3: podpis definiuje "global" i "local". Nie mówi, w jakiej rozdzielczości są sondy pikselowe ani czy sondy były trenowane na odpowiedniej etykiecie. Tylko azymut ma przedziały.

**7. Sprzeczności.** Tabela 3 "full image, global" daje 11.8° (test) i 14.3° (kostki), a tekst i Tabela 1 mówią 11.5° i 14.1°. Prawdopodobnie inna rozdzielczość pikseli, ale nie jest to powiedziane. "Controls ... the input pixels (64 x 64)" (str. 6) vs "best pixel grid" w Tabeli 4. Kontrola precyzji: tu "On the same 12 images", w Ograniczeniach (str. 14) "on a test set". 4.4: "720 at t = 800" vs str. 5: "718" z szumem flow przy t = 200 (patrz porcja 2).

**8. Język.** Akapit "Numeric precision" i "Agreement with the original pipeline" są przykładem dobrego uproszczonego stylu: krótkie, konkretne, z liczbami. Akapit "Weak cues remain" gubi powiązania: nie wiadomo, do czego się odnosi "first variant". Zdanie "Thus the light direction is present, but a linear read-out of the pixels cannot find it at an arbitrary place in the image" jest dobre.

---

## Porcja 4: str. 7-8 (rysunek 3, sekcja 8, tabela 4, rysunek 4)

**1. Zrozumienie.** Rysunek 3 pokazuje, że sonda pikselowa w v1 rozkłada wagi po podłodze (odblask), a w v2 koncentruje je wokół obiektu i cienia. Sekcja 8: dla każdego modelu autorzy dopasowują sondę dla każdego bloku, kroku czasowego i siatki (1x1, 2x2, 4x4, 8x8 komórek), wybierają na walidacji; linią bazową są piksele w czterech rozdzielczościach. Kontrola z przetasowanymi etykietami daje 83-93°. Wynik na v2r: SANA 7.6° na kostkach vs piksele 20.1°, latent DC-AE 19.0°, losowe wagi SANA 15.9°, SD 1.5 i SDXL 11.1° i 7.3°. Wniosek autorów: model ma reprezentację niezależną od pozycji, bo średnia po wszystkich tokenach (1x1) wytrenowana na v2 przenosi się na v2r (9.0° i 17.6°). Oba sygnały (cieniowanie obiektu i cień rzucony) są odczytywalne osobno.

**2. Pojęcia bez definicji.**
- str. 7: "Exact spatial decomposition of the linear pixel probe" (co dokładnie jest rozkładane: wagi czy wkład do wyjścia uśredniony po obrazach?).
- str. 7: "token means on a grid of 1 x 1, 2 x 2, 4 x 4 or 8 x 8 cells". "4x4" ma tu znaczenie liczby komórek, a na str. 3 "4 x 4 average pool" oznaczało rozmiar okna uśredniania (daje siatkę 8x8). To samo oznaczenie, dwa znaczenia.
- str. 8: "for every source" ("source" = źródło cech; nowe słowo na to samo co wcześniej "features" lub "model").
- str. 8: "v2rns", "v2rso" (skróty wprowadzone tylko w nawiasach; Tabela 4 używa innych nazw: "no shadow", "only shadow").
- Tabela 4: "up2", "mid", "down" (bloki UNet), "(Block, t, grid)" dla wierszy pikseli to rozdzielczość.
- str. 8: "paired bootstrap" bez liczby próbek.

**3. Zdania czytane więcej niż raz / niejasne.**
- str. 7: "The 1 x 1 grid is the mean of all tokens and has no information about the position." Zbyt mocne: średnia po tokenach, w 2240 kanałach, może kodować pozycję obiektu (nieliniowo lub w konkretnych kanałach). Autor mógłby napisać "no explicit information".
- str. 7: "A coarse grid is often the stronger linear baseline, because it averages the position of the shadow." Dlaczego wtedy v2 pokazuje najlepszy wynik dla 64 px (Tabela 4), a Rysunek 3 dla 16x16 pokazuje 40.2° na kostkach?
- str. 8: "Thus the model has a direction code that is the same at every place in the image, and the cell features add a position-dependent part." Wniosek mocniejszy niż liczby: zob. punkt 4.
- str. 9: "A probe fitted on v2r, where both cues are present, partly transfers to the scenes with one cue. SANA gives 33.3 without the shadow and 30.2 with only the shadow". Na jakim podziale (kostki czy sfery/walce)? Czy 30° to "częściowy" transfer (poziom losowy to 90°)?

**4. Twierdzenia bez uzasadnienia lub mocniejsze niż dowody.**
- str. 7: "The representation does not depend on the position." Dowód: 1x1 po treningu na v2 daje 9.0° (sfery i walce) i 17.6° (kostki), SD 1.5: 10.6° i 22.0°. Dla kostek 17.6° to prawie tyle, co najlepsza sonda pikselowa trenowana na v2r (20.1°), a 22.0° dla SD 1.5 jest gorsze. Z siatką 8x8 wynik to 38.4/40.9°. To jest częściowa niezależność, nie "does not depend". Streszczenie podaje tylko 9.0°.
- Brak ważnej kontroli: ten sam test 1x1 (trening na v2, test na v2r) dla SANA z losowymi wagami i dla DC-AE. Średnia po tokenach cech lokalnych jest w dużej mierze niezmiennicza względem przesunięcia z konstrukcji (konwolucje w blokach, uśrednianie). Ta kontrola rozstrzygnęłaby, ile z "niezależności od pozycji" daje trening, a ile architektura i uśrednianie.
- Brak pikselowego odpowiednika 1x1 (średnia wszystkich pikseli); porównanie "pixel probe fails (71.2 and 70.4)" dotyczy siatek, które zachowują pozycję, więc to nie jest ten sam typ cechy.
- str. 8: "So the model encodes the light from the shading of the object and from the cast shadow, each one alone." Pokazano, że jest odczytywalne liniowo, nie że "model koduje".
- str. 9: "So the maps do not separate the models." Wynik zerowy: SANA i piksele rozkładają wagi podobnie (1.4x udziału powierzchni). Nie wiadomo, co ten podrozdział ma wnieść do głównego wniosku. "Share of weight" niezdefiniowane.

**5. Pytania.** Czy sonda na średniej tokenów z sieci o losowych wagach też się przenosi z v2 na v2r? Dlaczego w "Where the probe reads" użyto bloku 10, a nie najlepszego (5)? (Odpowiedź częściowa z str. 5: siatka 8x8 była zapisana tylko dla bloków 4, 10, 18.) Skoro siatka 8x8 jest tylko dla trzech bloków, to protokół ze str. 7 ("every block ... 1x1, 2x2, 4x4 or 8x8") jest nieprecyzyjny. Dlaczego kryterium wyboru to suma MAE azymutu i elewacji, skoro str. 6 mówi, że elewacja nie jest dobrym celem? Dlaczego v2 w Tabeli 4 nie ma SDXL ani losowych wag?

**6. Rysunki i tabele.**
- Rysunek 3: brak skali kolorów, brak liczb. Podpis mówi "brighter cells have a larger share", ale nie mówi, co jest sumowane. Tytuł prawego panelu pokazuje "pixels 16x16 cubes az MAE 40.2 deg", a tekst i Tabela 4 podają dla v2 13.7° (64 px). Rysunek pokazuje więc sondę, która na kostkach jest słaba (40°), bez wyjaśnienia dlaczego akurat 16x16. Lewy panel wygląda jak szum z kilkoma jasnymi komórkami; teza "weight spreads over the floor" jest wiarygodna, ale z samego rysunku trudna do zweryfikowania.
- Tabela 4: gęsta (25 wierszy), ale czytelna; kolumna "Cubes - pixels" bardzo pomocna; przedziały drobnym drukiem. Wiersze pikseli mają rozdzielczość w kolumnie "Block, t, grid".
- Rysunek 4: trzy panele, legenda drobna, a w panelu v2r linia t=800 (żółta) wychodzi poza oś. Pozioma linia "pixels" to 64x64 (w v2r ok. 26°), a nie najlepsza siatka (20.1° w Tabeli 4), więc rysunek używa słabszej bazy niż tabela. Podpis: "The third line is SANA with random weights" (są trzy linie poziome; "third" mylące). Oś Y opisana tylko w lewym panelu.

**7. Sprzeczności.** (a) Str. 5: siatka 8x8 tylko dla bloków 4/10/18; str. 7: protokół dla "every block" z 8x8. (b) Str. 6: sonda pikselowa na v2r 25.1° i 25.8°; Tabela 4: 21.5° i 20.1° (najlepsza siatka); tekst nie mówi, że to różne rozdzielczości. (c) Rysunek 3: 40.2° vs Tabela 4: 13.7°. (d) Str. 6: elewacja nieprzydatna, ale służy do wyboru konfiguracji.

**8. Język.** Pogrubione śródtytuły w stylu "The representation does not depend on the position." podają wniosek przed dowodem i bywają mocniejsze niż dane. "Random nonlinear features of the latent already help a linear probe. The trained model reduces this error by about half." Dobre, krótkie, konkretne zdania z liczbami.

---

## Porcja 5: str. 9-10 (odczyt wag, sekcja 9: definicje interwencji, wyniki przesunięcia i wymazywania)

**1. Zrozumienie.** Sekcja 9 sprawdza, czy model "używa" kierunku. Wszystkie interwencje zaczynają od renderu z v2, który jest kodowany, zaszumiany do σ0=0.7 i odszumiany w 20 krokach Eulera (SDEdit); hook na jednym bloku zmienia jego wyjście w każdym kroku. Mierzą obrót światła wyjściowego względem "base output" trzema miarami: sondą pikselową (wyuczoną na v2), miarą cienia (środek ciężkości ciemnych pikseli podłogi, bez dopasowanego modelu) i "kept share" (jaka część ostrego cienia bazowego nadal jest w cieniu). Interwencje: (a) przesunięcie wzdłuż "probe direction" (najkrótsza zmiana obracająca wyjście sondy o 90°) lub "activation pattern" (regresja cech na sin/cos), z kontrolą losową; (b) wymazywanie LEACE; (c) test podmiany tokenów ("patch") z renderem z światłem +90°. Wyniki: przesunięcie wzdłuż wzorca w skali ×1 obraca sondę pikselową o +60 do +65°, ale ostry cień zostaje; ×3 daje +120° i rozmyty ciemny obszar po stronie celu, bez nowego ostrego cienia; wagi sondy są gorszym uchwytem; losowy wektor nic nie zmienia; LEACE nie daje efektu.

**2. Pojęcia bez definicji.**
- str. 9: "base output", "Euler steps", "v2rot" (podobna do "v2r", co łatwo pomylić), "R90", "ŷ", "W", "h", "c" (to samo "c" oznacza kamerę w Dodatku A), "B", "y(φ)", "δ" (czy wspólne dla wszystkich komórek, czy zależne od komórki, jaki ma wymiar), "ground azimuth", "floor median", "grey levels", "darkness-weighted centroid", "crisp shadow (40 grey levels below the floor median)".
- str. 9-10: "natural size (×1)" ("natural" = średnia różnica cech dla zmiany o 90°, dopiero z kontekstu), "dose" (na str. 10), "handle" ("causal handle"), "activation pattern" (zdefiniowane na str. 9 jako regresja; termin z Haufe et al.).
- str. 11, Tabela 5: "probe_exact", "probe_r1", "random_probe", "pattern_x1", "pattern_x3", "random_pattern", "erase". Tekst nigdzie nie wyjaśnia "probe_r1" (r1 = skalowanie do normy tokenu?) ani "random_probe".
- str. 10: "SANA has no skip connections between blocks" (chodzi o brak długich połączeń typu U-Net; wewnątrz bloków są połączenia rezydualne).
- str. 10: "positive control" użyte jako "This condition is a positive control."

**3. Zdania czytane więcej niż raz / niejasne.**
- str. 9: "We report the signed rotation of the output light against the base output of the same image and noise, with two measures." Po tym zdaniu są TRZY punkty (pixel probe, shadow measure, kept share). Dalej zdanie "A centroid of dark pixels cannot separate a turned shadow from an old shadow plus a new dark region, so the kept share and a visual check complete it."
- str. 9: wzór δ = W^T (W W^T)^(-1) (R90 ŷ − ŷ). Nie wiadomo, czy W zawiera PCA i standaryzację (sonda z sekcji 3 to PCA + Ridge), czy ŷ jest wyjściem dla konkretnego obrazu (wtedy δ jest różne dla każdego obrazu), i jak obraca się trzeci wyjściowy element (elewacja).
- str. 9: "δ = (y(φ+90°) − y(φ))B": czy φ to prawdziwy azymut obrazu, czy przewidziany? (Wygląda na prawdziwy.)
- str. 9: "It removes all linear information about the azimuth from every token of the block." LEACE jest dopasowane na średnich z komórek, a stosowane do tokenów.
- str. 10: "Thus a patch of all tokens at all steps makes the output equal to the source output, up to the difference of the two inputs." Jak duża jest ta różnica (σ0=0.7 zachowuje 30% sygnału wejściowego)? Odpowiedź w liczbach dopiero na str. 11 (0.73).

**4. Twierdzenia bez uzasadnienia lub mocniejsze niż dowody.**
- str. 9: "so it mostly reads the shading of the object" (sonda pikselowa daje 59% wagi komórkom obiektu: "mostly" to niewiele; pozostałe 41% to inne obszary, a obszar obiektu zawiera też początek cienia).
- str. 10: "So the shift changes the shading of the object, not the cast shadow." Brak bezpośredniej miary cieniowania obiektu. Wniosek pochodzi z różnicy między sondą pikselową (która "czyta głównie obiekt") a miarą cienia. Alternatywa: sonda reaguje na ogólną zmianę jasności.
- str. 10: "Thus the pattern direction is a causal handle on the light of the output. The probe weights are a worse handle." Porównanie jest zaburzone przez normę δ: dla "probe_exact" norma nie jest podana, a dla "pattern_x1" to 2% normy tokenu. Nie wiadomo, czy porównanie jest przy równej normie. "probe_r1" ma normę tokenu i niszczy obraz.
- str. 10: "A random vector with the same norm in each cell does nothing (both rotation measures within ±2°)". Dotyczy "random_pattern". "random_probe" (tej samej dużej normy co "probe_r1") niszczy obraz w blokach 4 i 10 (Tabela 5). Zdanie jest więc prawdziwe tylko dla jednej z dwóch kontroli.
- str. 10: liczby "kept share" (68-80%, 32-40%, 74-89%, 0.93-0.99) nie występują w Tabeli 5 (tabela nie ma kolumny "kept"). Czytelnik nie może ich sprawdzić.
- str. 10: "At ×2 the shading reaches the target in all 60 images" (Tabela 6 podaje tylko średnią +79°).
- str. 10: "A random change of the base output, for example a random shift, keeps 82% to 88% of its crisp shadow, so this is the level of no effect." W Tabeli 6 losowy wektor daje 0.88, 0.73 i 0.78 dla różnych σ0, więc "level of no effect" zależy od ustawienia.

**5. Pytania.**
- Dlaczego kierunki do interwencji pochodzą z cech przy t=400, skoro sondy są najlepsze przy t=100? (odpowiedź tylko częściowa dopiero na str. 14, w Ograniczeniach: tam to jest wymienione jako ograniczenie, bez uzasadnienia).
- Jak wybrano 60 obrazów testowych (jakie kształty)?
- Jakie są trzy bloki (4, 10, 18)? W tekście nie ma ich wymienionych; wiem z Tabeli 5 na str. 11.
- Dlaczego pewne wyniki (np. +120°) przekraczają cel +90° i co to znaczy dla miary? (odpowiedź: nigdzie nie omówione).
- Czy jest jakakolwiek bezpośrednia miara cieniowania obiektu? (Oczekuję jej w dalszej części. Nigdy się nie pojawia: str. 14, Ograniczenia: "The pixel probe on the outputs mostly reads the shading of the object.")
- Jak wybrano wagę cienia ostrego (40 poziomów szarości)? Czy kept share = 1.00 dla tego samego obrazu i 0.00-0.01 dla 90° wystarcza jako kalibracja? Wygląda na rozsądną kalibrację.

**6. Rysunki i tabele.** Brak na tych stronach (Tabele 5 i 6 oraz Rysunek 5 na kolejnych).

**7. Sprzeczności.** (a) "three interventions" (shift, erasure, patch), a w sekcji "Shift along a direction" są jeszcze dwie osobne wersje (probe, pattern) z własnymi kontrolami. (b) "two measures" kontra trzy punkty i wkład 4/lista kontrolna (dwie niezależne miary). (c) Ta sama miara jest nazwana "shadow measure" (str. 9), "dark-region measure" (str. 10), "dark region" (Tabela 6) i "shadow" (Tabela 5). Czytelnik musi się domyślać, że to jedno i to samo. (d) Str. 10: "The probe reads best in block 5 at t = 100 (test MAE 2.6 on v2)", a Tabela 4 (str. 8) dla v2 pokazuje najlepszy wynik w bloku 4 z MAE 2.2° (patrz porcja 4: prawdopodobnie blok 5 tylko z siatką 4x4, ale to nie jest powiedziane).

**8. Język.** To najgęstsza sekcja. Krótkie zdania z pojedynczymi definicjami są zrozumiałe, ale razem tworzą ścianę definicji z wzorami bez motywacji (po co regresja cech na sin/cos zamiast sondy). Słowa "turn", "keep", "handle", "dose", "natural size", "pattern" mają kilka znaczeń lub są zbyt potoczne dla tekstu naukowego. "Thus" jest używane jako jedyne spoiwo.

---

## Porcja 6: str. 11-12 (rysunek 5, tabela 5, wyniki podmiany tokenów, rysunek 6, tabela 6, początek dyskusji)

**1. Zrozumienie.** Tabela 5 podaje obrót światła wyjściowego dla warunków w blokach 4, 10 i 18. Wzorzec ×1 daje +60 do +65° wg sondy pikselowej, a miara cienia tylko +5 do +7°; ×3 daje +120° i +92 do +109°; losowy wzorzec i LEACE nie dają nic. Tabela 6: dla bloku 5 efekt rośnie z dawką, ale ostry cień nie jest rysowany na nowo; przy większym szumie wejściowym (σ0=0.85, 0.95) cień nadal nie obraca się. Dodanie wzorca do wszystkich 20 bloków (Rysunek 5) daje duży obrót, ale z blokowymi artefaktami zamiast nowego cienia. Test podmiany tokenów: podmiana wszystkich tokenów bloku 10 daje 0.73 obrotu, tokenów obiektu (ok. 10% obrazu) 0.57 do 0.63 obrotu cieniowania i 33-38% cienia źródłowego, tokenów cienia źródłowego 59% cienia źródłowego (ale 92% cienia docelowego zostaje: dwa cienie). Wyniki są takie same w każdym bloku. Autorzy wnioskują, że tokeny obiektu niosą kod światła, który ustawia cieniowanie i część cienia.

**2. Pojęcia bez definicji.** "Pixels: mean absolute change ... mean over the blocks" w Tabeli 5 (średnia po blokach ukrywa, które bloki zostały zniszczone). "–" w Tabeli 5 (random_probe, blok 10, shadow) bez wyjaśnienia. "Kept" w Tabeli 6 (zdefiniowane jako "kept share" na str. 9). "patched block", "full source pass".

**3. Zdania czytane więcej niż raz / niejasne.**
- str. 12: "The results are the same in every block, because a patch of a token replaces its whole state, which later blocks only refine." Jeśli wyniki są takie same w bloku 0 i 18, to test nie lokalizuje w głębokości i może mierzyć zwykłe kopiowanie lokalnej zawartości (po podmianie w ostatnim bloku tokeny są prawie gotowym obrazem). Autor podaje to jako wyjaśnienie, nie jako ograniczenie.
- str. 12: "The patch of the other floor tokens gives the reference: 12% of the source shadow (the two shadows meet at the object base)". Czy 12% to poziom "braku efektu"? Zdanie w nawiasie sugeruje, że nie.
- str. 12: "A patch of the tokens where the source shadow lies copies 59% of the source shadow to its place, but keeps 92% of the target shadow: the output has two shadows." To raczej trywialne (kopiowanie lokalnych tokenów cienia kopiuje cień), a nie dowód "kodu światła".

**4. Twierdzenia bez uzasadnienia lub mocniejsze niż dowody.**
- str. 12: "Thus the object tokens carry the light code that sets the shading and part of the cast shadow." Region "object" zawiera początek cienia (autorzy sami tak piszą przy odczycie wag, str. 9), a lokalna kopia wyglądu obiektu tłumaczy wynik bez żadnego "kodu światła". Brak kontroli: podmiana tokenów obiektu z renderu z TYM SAMYM światłem, ale innym kształtem/położeniem (odróżnia kopiowanie wyglądu od kopiowania światła).
- str. 10-11: pozytywna kontrola ("makes the output equal to the source output") w praktyce daje 0.73 obrotu i 84% cienia źródłowego, a nie 1.0. Wynik "0.57 do 0.63" znormalizowany do 0.73 byłby ok. 0.8, ale tekst tego nie robi i nie tłumaczy luki (zapewne σ0=0.7 zostawia 30% sygnału wejściowego).
- Tabela 5/Rysunek 5: sonda pikselowa daje odczyty ok. +120 do +150° przy celu +90° (np. Rysunek 5, wiersz z etykietą 230: odczyt 358° przy celu 320°). Odczyt powyżej celu pokazuje, że sonda poza swoim rozkładem nie jest skalibrowana, a teza "turns the shading" opiera się na tej sondzie. Tekst nie komentuje przekroczenia celu.
- Tabela 5: wiersz "probe_r1" w bloku 18 (+84 [+80, +89], cień +46) wygląda jak sukces, a podpis tabeli mówi tylko o zniszczeniu obrazu w blokach 4 i 10; kolumna "pixels" (107) to średnia po blokach, więc nie widać, że w bloku 18 obraz też może być zniszczony.

**5. Pytania.** Dlaczego wynik podmiany tokenów obiektu jest równy w bloku 0 i 18 (odpowiedź: zdanie na końcu str. 12, które to czyni zaletą). Czy podmiana tokenów z innego obiektu przy tym samym świetle daje podobny efekt? (nie sprawdzone). Jak interpretować "0.73" dla pozytywnej kontroli? Czy ×2 w Tabeli 6 pasuje do ×1 i ×3 z Tabeli 5? (inne bloki, inne t, więc nie da się porównać.)

**6. Rysunki i tabele.**
- Rysunek 5: siatka 4 wierszy po 6 obrazów. Tytuły kolumn są drobne i praktycznie nieczytelne w tej skali ("pattern_all_x0.25 pixel probe 98"). Podpis mówi, co widać ("dark and bright blocks"), ale nie wymienia kolumn. Bloki ciemne/jasne w wierszach 3-4 są wyraźnie widoczne. W wierszach 1-2 obiekt jest zniekształcony i nie widać wyraźnie, że "shading of the object turns".
- Tabela 5: dobra struktura, przedziały ufności tylko dla sondy, bez przedziałów dla miary cienia, brak kolumn "kept" (które tekst cytuje), kody warunków nieobjaśnione.
- Rysunek 6: słupki z przedziałami, linia celu +90°; czytelny. Dla "probe_r1" i "random_probe" przedziały są ogromne i bez znaczenia. Rysunek pokazuje tylko sondę pikselową, czyli miarę, którą autorzy sami nazywają słabą (str. 14).
- Tabela 6: czytelna, trzy ustawienia w jednej tabeli, bez przedziałów.

**7. Sprzeczności.** Pozytywna kontrola: "equal to the source output up to the difference of the two inputs" (str. 10) vs 0.73 i 84% (str. 11). Str. 12: "in every block from 0 to 18" przy 20 blokach (0-19).

**8. Język.** Wypunktowanie ze str. 10 pomaga. Zdanie "A random vector with the same norm in each cell does nothing" jest zbyt potoczne. W podrozdziale o podmianie jest wiele procentów po sobie (33-38%, 60-67%, 12%, 96%, 59%, 92%) bez tabeli, więc czytelnik musi trzymać je w głowie; tabela byłaby lepsza.

---

## Porcja 7: str. 13-14 (rysunki 7-8, dyskusja, lista kontrolna, ograniczenia)

**1. Zrozumienie.** Dyskusja podsumowuje: oryginalny wynik nie dowodzi reprezentacji światła i nie dowodzi jej braku; po poprawkach sonda liniowa na aktywacjach trzech modeli czyta światło tam, gdzie liniowa sonda pikselowa nie może; kod przetrwa uśrednienie po tokenach i przenosi się z wycentrowanych obiektów; model czyta światło z cieniowania i z cienia rzuconego; część efektu daje architektura (losowe wagi), trening zmniejsza resztę błędu o połowę; mała zmiana wzdłuż wzorca zmienia cieniowanie obiektu, ale nie rysuje nowego cienia; tokeny obiektu niosą kod, który kontroluje cieniowanie i część cienia; reszta cienia z informacji rozproszonej. Lista kontrolna ma 9 punktów. Ograniczenia: jedna własność, proste sceny, pusty prompt, SDEdit, jedne bloki naraz, 60 obrazów, float16.

**2. Pojęcia bez definicji.** "SDEdit" (użyte na str. 9 jako cytowanie; nazwane dopiero w Ograniczeniach na str. 14), "the 3D relation between light, object and floor", "a second code for the shadow".

**3. Zdania czytane więcej niż raz / niejasne.**
- str. 14: "The shortest change that moves the probe output to the target moves the image by less than half of it" ("it" = cel).
- str. 14: "The cast shadow needs the 3D relation between light, object and floor." Podane jako fakt, nie jako hipoteza. Nie było żadnego testu na to.
- str. 14: "The rest of the shadow comes from information that is spread over the image." Dowód: tylko to, że podmiana wszystkich tokenów (bloku 10) daje 84%, a pojedynczy region mniej. Nie wyklucza to innych wyjaśnień (np. trzeba zmienić tokeny obiektu i cienia równocześnie).

**4. Twierdzenia bez uzasadnienia lub mocniejsze niż dowody.**
- str. 13: "So it does not depend on where the object stands in the image." Zob. porcja 4 (17.6°, 22.0°, 38.4° dla 8x8; brak kontroli losowych wag dla 1x1).
- str. 13: "The model reads the light from the shading of the object and from the cast shadow, each one alone." Odczytywalność liniowa, nie "czytanie przez model".
- str. 14: "A small shift along the activation pattern of the light changes the light of the output, and a random shift does not." Poprawne dla miary pikselowej; patrz uwagi o miarach.
- str. 14: "Large shifts, also in all blocks together, only remove it and add dark patches." Przesunięcie jest stałe na komórkę siatki 4x4 (lub 8x8), więc zgrubne artefakty blokowe mogą wynikać z grubości interwencji, a nie z braku w modelu zdolności rysowania cienia. Wniosek "None of our shifts drew a new crisp shadow" jest ostrożnie sformułowany, ale dyskusja idzie dalej niż dane. Autorzy częściowo to uznają ("The data do not show if the model uses one code for both, or a second code for the shadow that our linear handles do not reach"), co jest uczciwe.

**5. Pytania z wcześniejszych stron: gdzie jest odpowiedź.**
- Pytanie ze str. 1 (kto jest autorem pierwszego badania, jak je zweryfikować): nie ma odpowiedzi w całym artykule.
- Pytanie ze str. 3-4 (jaka biblioteka): nie ma odpowiedzi ("link withheld", biblioteka nienazwana).
- Pytanie ze str. 9 (po co t=400 do kierunków interwencji): str. 14 tylko wymienia to jako ograniczenie, bez uzasadnienia.
- Pytanie ze str. 9-10 (czy jest bezpośrednia miara cieniowania): nigdy; str. 14: "The pixel probe on the outputs mostly reads the shading of the object."
- Pytanie ze str. 7-8 (kontrola losowych wag dla 1x1): nigdy.
- Pytanie ze str. 2: czy "4 problems" są policzone dokładnie: nie, brak podsumowania w Dyskusji.

**6. Rysunki i tabele.**
- Rysunek 7: siatka 4x8 małych obrazów, tytuły (np. "pixel probe 334") praktycznie nieczytelne w druku. Podpis mówi: "Four test images: the render, the base output and each condition of the shift test in block 5 (t = 100 pattern)". Na rysunku są jednak kolumny jak w Tabeli 5 ("probe_exact", "erase") dla bloków 4/10/18, a nie dla bloku 5, więc podpis nie pasuje do zawartości. Podpis mówi "darken the floor towards the target", a na miniaturach w wierszach 3-4 wydają się widoczne jasne, białe obszary podłogi, których tekst nie opisuje (nie jestem pewien tego na miniaturach). Kolumn nie wymienia podpis.
- Rysunek 8: dwa panele, linie dla czterech regionów (tokeny obiektu, tokeny cienia źródłowego, pozostała podłoga) plus linie odniesienia (pełny przebieg źródłowy, wszystkie tokeny w bloku 10). Czytelny, linie praktycznie płaskie. Oś bloków 0-18 (a model ma 20 bloków).

**7. Sprzeczności.** (a) "in every block from 0 to 18" vs "20 blocks" (blok 19 pominięty bez wyjaśnienia). (b) Ograniczenia: "use directions from the features at t = 400" dotyczy tylko Tabeli 5; Tabela 6 i wszystkie bloki razem używają t=100, więc to ograniczenie jest przedstawione szerzej niż w faktach. (c) Ograniczenia: "float32 gives the same features on a test set" vs "12 images" na str. 5. (d) Wnioski w Dyskusji o niezależności od pozycji vs liczby ze str. 7.

**8. Język.** Lista kontrolna jest jasna i konkretna. Akapit "What the new results show" łączy sześć wniosków w jedno miejsce, zdania krótkie, ale dużo "So"/"Thus" bez rozróżnienia siły dowodu. Zdanie "A claim that a model 'moves the shadow' needs a measure of the shadow." jest bardzo dobre jako zasada.

---

## Porcja 8: str. 15-17 (zakończenie, odtwarzalność, literatura, Dodatek A)

**1. Zrozumienie.** Zakończenie powtarza tezę: wynik sondy jest "punktem wyjścia, nie dowodem wiedzy 3D"; pierwszy wynik był "right for the wrong reasons"; SANA i dwa inne modele niosą kierunek światła w formie, której wejście nie oferuje; SANA używa go do oświetlenia obiektu; miary cienia pokazują, że przesunięcie nie rysuje nowego cienia. Odtwarzalność: kod, specyfikacje zadań, tabele i testy w materiałach dodatkowych. Dodatek A podaje geometrię kamery i wzór cienia punktu na podłodze dla lampy punktowej.

**2. Pojęcia bez definicji.** W Dodatku A: "c" (pozycja kamery; to samo "c" jest wyrazem wolnym sondy na str. 9), "e_z", "x, y" (współrzędne pikseli; "x_t" w sekcji o szumie), "L" (lampa; w 4.2 "L" też lampa), "p" (środek obiektu).

**3. Zdania czytane więcej niż raz / niejasne.** "carry the light direction in a form that the input does not offer" (str. 15): wejście zawiera informację (przycięcie do obiektu daje 4.4°, str. 6); dokładniej: w formie, której liniowa sonda na pikselach nie odczyta.

**4. Twierdzenia bez uzasadnienia.** "Every number in this paper comes from a script in the repository." Dla czytelnika bez materiałów dodatkowych to oznacza utratę informacji w wielu miejscach (patrz część C). "SANA uses it to light the object." Wniosek z przesunięcia, bez miary cieniowania bezpośrednio.

**5. Pytania.** Dodatek A: nie ma odwołania z tekstu głównego (w sekcji 4.1 "We wrote the projection of Blender in NumPy" nie wskazuje dodatku), dodatek zawiera wzory tylko dla lampy punktowej z v1, nie dla lampy słonecznej z v2, ani wzoru na "ground azimuth" użytego w mierze cienia.

**6. Rysunki i tabele.** Brak.

**7. Sprzeczności.** Nie znalazłem nowych.

**8. Język.** Zakończenie jest krótkie i dobrze skonstruowane. Literatura jest kompletna w formie, ale brak w niej "first study", którego dotyczy cały audyt.

---

# CZĘŚĆ KOŃCOWA

## A. Główne pytanie badawcze, metoda i główne wnioski (z pamięci i notatek)

**Pytanie.** Czy model tekst-obraz SANA-1.6B (oraz SD 1.5 i SDXL) naprawdę reprezentuje kierunek światła (azymut) w scenie w formie, która wykracza poza wskazówki 2D, i co właściwie pokazał wcześniejszy wynik sondy liniowej (14° MAE na nieznanych kostkach)? Drugie pytanie poboczne: czy model tego kierunku faktycznie "używa" przy generowaniu.

**Metoda.**
1. Audyt oryginalnego potoku: dokładny model kamery z renderów (NumPy), analiza etykiety (pozycja lampy a światło przy obiekcie), kontrole sondami na pikselach i na regionach obrazu (podłoga, obiekt, odblask), odnalezienie błędu szumu (funkcja add_noise wywołana przed set_timesteps), błędnych rysunków i wyboru konfiguracji na zbiorze testowym.
2. Poprawiony potok (szum flow) i powtórzenie oryginalnego eksperymentu.
3. Nowe zbiory: v2 (lampa słoneczna, obiekt w centrum), v2r (ta sama scena, losowa pozycja obiektu), warianty bez cienia i tylko z cieniem. Sondy liniowe (PCA + Ridge) z kontrolami: piksele w kilku rozdzielczościach, latent autoenkodera DC-AE, ta sama sieć z losowymi wagami, przetasowane etykiety. Wybór bloku, kroku czasowego i siatki na walidacji, sparowany bootstrap na teście.
4. Interwencje podczas generowania (SDEdit od σ0=0.7): przesunięcie aktywacji wzdłuż "wzorca aktywacji" (regresja cech na sin/cos) lub wzdłuż wag sondy, kontrola losowa, wymazywanie LEACE, test podmiany tokenów z renderem ze światłem obróconym o 90°. Trzy miary wyjścia: sonda pikselowa, miara cienia (środek ciemnych pikseli), "kept share" ostrego cienia.
5. Lista kontrolna dla badań z sondami (9 punktów).

**Główne wnioski autorów (jak je zrozumiałem).**
1. Oryginalny wynik nie dowodzi, że SANA reprezentuje światło: piksele dają to samo, etykieta to pozycja lampy, a odblask na podłodze, niezależny od kształtu, tłumaczy transfer na kostki.
2. Błąd szumu ukrył informację: te same dane i ten sam protokół dają 4.8° zamiast 14.5° po poprawce.
3. Na czystych danych z losową pozycją obiektu sonda liniowa na aktywacjach daje 7.6° (kostki), a na pikselach 20.1°, latent DC-AE 19.0°, losowe wagi 15.9°; efekt jest podobny w SD 1.5 i SDXL.
4. Kod kierunku w dużej mierze nie zależy od pozycji i pochodzi zarówno z cieniowania obiektu, jak i z cienia rzuconego.
5. Przesunięcie wzdłuż wzorca aktywacji obraca cieniowanie obiektu (miara pikselowa +60 do +65° z +90°), a przesunięcie losowe nic nie zmienia; duże przesunięcia usuwają stary cień i ciemnią podłogę w stronę celu, ale nie rysują nowego ostrego cienia; wagi sondy są gorszym uchwytem; LEACE bez efektu.
6. Podmiana tokenów obiektu przenosi cieniowanie i ok. jedną trzecią cienia; cały cień wymaga podmiany wszystkich tokenów.
7. Pojedyncza sonda to punkt wyjścia, nie dowód wiedzy o 3D; lista kontrolna ma to uporządkować.

## B. Dowody dla każdego wniosku i zarzuty recenzenta

**Wniosek 1 (oryginał nie dowodzi).** Dowody: Tabela 1 (piksele 14.1° vs SANA 14.0°), Tabela 3 (sama podłoga 9.4° na kostkach, odblask w dwóch liczbach 17.6°, obiekt sam 21.5°/44.3°), analiza etykiety (35.0° średniej różnicy). Przekonuje bardzo (proste, konkretne kontrole). Zarzuty: oryginalne badanie nie jest cytowane ani opisane, więc czytelnik nie może sprawdzić, że to było oryginalne twierdzenie; w Tabeli 3 nie podano rozdzielczości pikseli i czy trenowano na odpowiedniej etykiecie; wyjaśnienie "object only adds shape-specific variance" jest bez testu.

**Wniosek 2 (błąd szumu).** Dowody: Tabela 2 (współczynniki a, b), normy tokenów (11,147 vs 11,100 zapisane, 718 po poprawce), Tabela 4 v1 (14.5° z szumem oryginalnym, 4.8° z flow, ten sam protokół wyboru). Przekonuje w dużej mierze, bo porównanie ten sam protokół / dwa szumy jest czyste. Zarzuty: liczby "35 times" nie da się odtworzyć z Tabeli 2 (35.4 to norma bezwzględna, stosunek do oczekiwanej to ok. 43); sigma 25.53 nie jest wyprowadzona; niejasne 720 vs 718; biblioteka i cztery harmonogramy nienazwane, link zgłoszenia ukryty, więc część dotycząca biblioteki jest niesprawdzalna; wynik 4.8° nadal dotyczy danych z odblaskiem (autorzy to przyznają). Dodatkowy zarzut logiczny: "Each problem can explain the result without 3D knowledge" (str. 2) kłóci się z tym, że ten błąd zaniżał wynik.

**Wniosek 3 (odczytywalność poza wejściem, v2r).** Dowody: Tabela 4 z przedziałami, wiele modeli, kontrole, przetasowane etykiety. Przekonuje co do tego, że sonda LINIOWA na pooled aktywacjach czyta kąt lepiej niż sonda LINIOWA na pikselach. Zarzuty: (a) baza jest słaba, bo piksele dla sondy liniowej nie są zbyt silną bazą (przycięcie do obiektu daje 4.4°; brak małej CNN lub losowych cech konwolucyjnych na pikselach jako bazy); sama sieć z losowymi wagami daje 15.9° (jedna trzecia poprawy), więc znaczna część efektu to nieliniowe cechy lokalne, a nie wiedza o świetle; (b) wybór spośród ok. 400 konfiguracji (20 bloków x 5 kroków x siatki; faktycznie mniej, bo 8x8 tylko dla 3 bloków) na walidacji liczącej 200 obrazów (sfery i walce), podczas gdy baza pikselowa ma 4 rozdzielczości; (c) przedziały bootstrap uwzględniają tylko losowość zbioru testowego, nie losowość wyboru konfiguracji, ziarna ani podziału; (d) jedna scena, jeden materiał i kolor, pusty prompt; (e) próg "must do better than 25°" nie rozróżnia nauczonej reprezentacji od losowej (losowe wagi go spełniają); (f) sformułowanie "far beyond the input" w streszczeniu i wkładzie jest mylące, bo wejście zawiera informację, a przewaga dotyczy odczytu liniowego.

**Wniosek 4 (kod niezależny od pozycji; oba sygnały).** Dowody: 1x1 po treningu na v2 daje 9.0°/17.6°; v2rns i v2rso w Tabeli 4. Przekonuje częściowo: oba sygnały są liniowo odczytywalne (Tabela 4, wyraźna przewaga nad pikselami i DC-AE), to jest solidne. Niezależność od pozycji jest przesadzona (17.6° na kostkach to praktycznie tyle co najlepsza sonda pikselowa w rozkładzie, 22.0° dla SD 1.5, 38° dla 8x8); streszczenie podaje tylko 9.0° bez wskazania, że to sfery i walce. Brak kontroli: 1x1 z losowych wag i z DC-AE; brak pikselowej średniej. Wzajemny transfer cieni/cieniowania (33.3° i 30.2°) nie ma bazy ani oznaczonego podziału, więc "share part of their code" jest słabe. Odczyt wag (str. 9) to wynik zerowy bez wyraźnej roli.

**Wniosek 5 (interwencje: przesunięcie, wymazywanie).** Dowody: Tabele 5-6, Rysunki 5-7; kontrole losowe; trzy bloki. Przekonuje co do tego, że przesunięcie wzdłuż wzorca zmienia obraz w sposób kontrolowany i że kontrola losowa niczego nie zmienia. Zarzuty: (a) "the model uses the code" jest mocniejsze niż dowód: sterowanie pokazuje zdolność wpływu, nie naturalne użycie, a wymazywanie daje wynik zerowy, którego streszczenie nie wspomina; (b) brak bezpośredniej miary cieniowania; sonda pikselowa (59% wagi na komórkach obiektu), stosowana do obrazów poza rozkładem, daje odczyty przekraczające cel (+120° przy +90°); (c) porównanie wzorca z wagami sondy jest zaburzone przez normę przesunięcia (norma probe_exact nie podana); (d) przesunięcie jest stałe na komórkę siatki, więc blokowe artefakty i "brak nowego ostrego cienia" mogą wynikać z grubości interwencji (autorzy to wymieniają tylko ogólnie w Ograniczeniach); (e) kierunki z t=400, a sondy najlepsze przy t=100 (bez uzasadnienia); (f) tylko 60 obrazów, jeden zbiór, brak przedziałów dla miary cienia i kept share, liczby kept share nie ma w Tabeli 5; (g) SDEdit przy σ0=0.7 zostawia stary cień w wejściu (częściowo adresowane przez σ0=0.85).

**Wniosek 6 (podmiana tokenów).** Dowody: Rysunek 8, liczby na str. 11-12. Zarzuty: wyniki są takie same w blokach 0-18, co sugeruje kopiowanie lokalnej zawartości, a nie "kod światła" (autorzy podają to jako zaletę); pozytywna kontrola daje 0.73, a nie 1.0, bez wyjaśnienia; brak kontroli z podmianą tokenów z renderu o tym samym świetle, ale innym obiektem; region "object" zawiera początek cienia; podmiana tokenów cienia kopiuje cień, co jest trywialne. 48 par.

**Wniosek 7 (lista kontrolna).** Rozsądna, część punktów (1, 2, 3, 5, 6, 8) jest konkretnie uzasadniona w tekście. Zarzut: zbyt ogólna reklama "checklist for generative models" na bazie jednej własności i jednego modelu.

## C. Czy dałbym radę odtworzyć eksperymenty tylko z artykułu?

Częściowo, ale nie w pełni. Mam wystarczające informacje, żeby odtworzyć: ogólny schemat sond (PCA 256, Ridge, siatki uśredniania), regułę szumu flow, model kamery lampy punktowej (Dodatek A), cel wyboru na walidacji, typ obrazów v2/v2r w zarysie, ideę przesunięć i podmian.

Czego brakuje:
- Tożsamość i opis oryginalnego badania; nazwa i wersja biblioteki oraz harmonogramów z błędem; parametry (shift) potrzebne, by odtworzyć Tabelę 2 (sigma 25.53).
- Wejście do DC-AE (obrazy 512x512 przeskalowane do 1024x1024?), skalowanie latentu, sposób uśredniania cech DC-AE w sondzie, rozmiar cech przy siatce 8x8 (8x8x2240 = 143,360 wymiarów) z PCA 256; siatka wartości α; liczba prób bootstrap; liczba ziaren.
- SD 1.5 i SDXL: który VAE, rozdzielczość, prompt, nazwy i indeksy bloków, mapowanie t na harmonogram, brak kontroli losowych wag dla UNet.
- Rendering v2/v2r: siła lampy słonecznej, materiał i kolor podłogi, rozmiary obiektów, rozkład rotacji, jednostki zakresu pozycji, sposób renderowania wariantów "bez cienia" i "obiekt niewidoczny", zbiór v2rot, liczba obrazów w poszczególnych wariantach (300 w jednym zdaniu, nigdzie więcej).
- Interwencje: wybór 60 obrazów (kształty), czy δ jest zależne od obrazu i od komórki, czy w W jest PCA i skalowanie, czy φ jest prawdziwe czy przewidziane, rozwiązanie dla elewacji, harmonogram Eulera (shift), definicja regionów (maski), progi miar (jest tylko 40 poziomów szarości dla "crisp shadow"), wzór na "ground azimuth" i lampę słoneczną.
- Definicja miary MAE kątowej (zawijanie), definicje nazw warunków w Tabeli 5 (probe_r1 itd.).
- Dokładne kryterium kompletnej tabeli "kept share" dla Tabeli 5.
Autorzy odsyłają do materiałów dodatkowych ("Every number in this paper comes from a script in the repository"); czytelnik bez nich traci te informacje.

## D. 10 najważniejszych problemów z czytelnością i logiką (od najpoważniejszego)

1. **Wniosek o "użyciu kodu" jest mocniejszy niż dowody (str. 1, 10-11, 14).** Streszczenie: "Interventions show that the model uses the code." Str. 11: "The erasure result does not show that the information is unused." Sterowanie pokazuje zdolność wpływu, wymazywanie jest wynikiem zerowym, którego streszczenie nie podaje, a miara "shading" to globalna sonda pikselowa z odczytami przekraczającymi cel (+120° przy +90°).
2. **Chaos w miarach interwencji (str. 9, 10-11, 2).** "with two measures" po czym trzy punkty (pixel probe, shadow measure, kept share); ta sama miara nazywa się "shadow measure", "dark-region measure", "dark region", "shadow"; wkład 4 mówi o "two independent measures". Sonda pikselowa jest jedyną miarą "shading", ale tekst mówi tylko "mostly reads the shading" (59% wagi na obiekcie).
3. **Wewnętrzna sprzeczność wkładu 1 i liczenia problemów (str. 1-2, 4).** "Our audit finds four problems" i "Each problem can explain the result without 3D knowledge" kontra błąd szumu, który "hid the information". Pierwszy z czterech "problemów" to wynik kontroli, nie błąd. Sekcja 4 ma pięć podsekcji, a 4.5 dodaje kolejne usterki (rysunki, α, wybór konfiguracji na teście).
4. **"The representation does not depend on the position" (str. 7, 13, 1).** Dowód: 9.0° (sfery i walce) i 17.6° (kostki), SD 1.5 22.0°, siatka 8x8 38.4°; dla kostek to prawie tyle, ile najlepsza sonda pikselowa w rozkładzie (20.1°). Streszczenie: "gives 9.0° on moved objects" bez wskazania kształtu. Brak kontroli 1x1 dla losowych wag.
5. **Niespójne i nieuzasadnione progi i liczby pikselowe (str. 6-8).** "must do better than 25° on v2r" (25.1/25.8°) vs Tabela 4 (21.5/20.1°, najlepsza siatka); losowe wagi (15.9°) próg spełniają; "close to chance (90°)" dla 69.0°, 71.2°, 70.4°; Rysunek 3 podaje 40.2° dla pikseli v2 (16x16), a tekst 13.7° (64 px); Rysunek 4 używa 64x64, nie najlepszej siatki.
6. **Test podmiany tokenów: logika pozytywnej kontroli i płaskość po blokach (str. 10-12).** "makes the output equal to the source output" vs 0.73 i 84%; "The results are the same in every block, because a patch of a token replaces its whole state" (to czyni wynik niemożliwym do zlokalizowania i podatnym na trywialne kopiowanie lokalnej zawartości); brak kontroli z tym samym światłem i innym obiektem; Rysunek 8 pokazuje bloki 0-18 przy 20 blokach.
7. **Liczby o błędzie szumu, których czytelnik nie odtworzy (str. 4-5).** "35 times the expected norm" (z Tabeli 2: 35.4 to norma bezwzględna), "in block 4 the mean token norm is about 11,100 at t = 200 and 720 at t = 800" vs "With the flow rule, the norm is 718"; sigma 25.53 bez wyprowadzenia; biblioteka i "four schedulers" nienazwane. Kluczowy techniczny wynik audytu nie jest sprawdzalny z samego tekstu.
8. **Terminologia o wielu znaczeniach i niezdefiniowane nazwy (str. 1, 3, 5, 8-9, 11).** "code" (reprezentacja vs kod programu, w jednym streszczeniu); "4 x 4" jako rozmiar okna uśredniania (str. 3) i liczba komórek (str. 5, 8); "activation pattern", "natural size", "dose", "handle"; warunki "probe_r1", "random_probe" nigdy nie wyjaśnione; nazwy zbiorów "v2r", "v2rot", "v2rns", "v2rso" łatwe do pomylenia; "Table 4, rows v1" użyte przed definicją "pixel grid".
9. **Audytowane badanie nie jest cytowane ani dostępne (str. 1-4).** "A first study", "The study concluded", "The original report described the opposite direction", "The figures of the original study drew ..." Czytelnik nie może sprawdzić żadnego z zarzutów o rysunki ani o kierunek azymutu. W bibliografii brak tej pozycji; biblioteka również nienazwana.
10. **Tabele i rysunki nie pozwalają zweryfikować tekstu (str. 3, 5, 7, 11, 13).** Tabela 5 nie ma kolumn "kept", które tekst cytuje (68-80%, 32-40%, 74-89%, 0.93-0.99); "At ×2 the shading reaches the target in all 60 images" nie ma pokrycia w Tabeli 6 (tylko średnia +79°); Tabela 3 (11.8/14.3) vs tekst i Tabela 1 (11.5/14.1); brak skali kolorów w Rysunku 3; tytuły w Rysunkach 5 i 7 nieczytelne; podpis Rysunku 7 mówi o bloku 5, a kolumny odpowiadają warunkom Tabeli 5.

Poza tymi dziesięcioma: sprzeczność "blok 5, 2.6° na v2" (str. 10) vs Tabela 4 (blok 4, 2.2°); "oryginał 14°/14.0°/14.5°"; porównanie wag sondy ze wzorcem bez normy przesunięcia; "float16 na 12 obrazach" vs "on a test set".

**Ocena języka (uproszczony angielski techniczny).** Pomaga: zdania są krótkie, słownictwo proste, definicje i opisy danych (str. 2-4, 5) czyta się szybko, a wnioski typu "A claim that a model 'moves the shadow' needs a measure of the shadow" są precyzyjne. Szkodzi: (a) wyrazy wielu znaczeń lub potoczne ("turn", "keep", "handle", "dose", "natural size", "pattern", "code", "does nothing"); (b) brak spójników przyczynowych tam, gdzie logika nie jest trywialna ("the wrong noise hid the information", "Thus" jako jedyne spoiwo); (c) streszczenie jako lista stwierdzeń bez struktury (nie widać, które zdania to "four problems"); (d) gęste ściany definicji w sekcji 9, bez motywacji; (e) nagłówki akapitów podające wniosek przed dowodem; (f) "turns the shading of the object by 60° to 65°" brzmi nienaturalnie (światło się obraca, nie cieniowanie). Zdania są poprawne, ale w kilku miejscach tak skompresowane, że tracą sens.

## E. Co jest w artykule zrobione dobrze

- Uczciwy audyt własnego lub cudzego badania, z liczbami (Tabele 1 i 3), w tym przyznanie, że wynik oryginalny "does not show the opposite" i że v1 nadal ma odblask.
- Kontrole: piksele w kilku rozdzielczościach, latent autoenkodera, sieć z losowymi wagami, przetasowane etykiety, wiele modeli (SD 1.5, SDXL), przedziały bootstrap i różnice sparowane w Tabeli 4.
- Wybór bloku, kroku i siatki wyłącznie na walidacji; wskazanie, że oryginał wybierał konfigurację na teście.
- Sprawdzenie zgodności własnego potoku z oryginalnym (normy tokenów 11,147 vs ok. 11,100) przed twierdzeniem o winie szumu; kontrola precyzji float16.
- Sprytne dane: sparowane rendery (v2, v2r, bez cienia, tylko cień), które rozdzielają wskazówki; wyraźna logika "decisive test" (losowa pozycja).
- Kalibracja miary "kept share" (1.00 dla tego samego obrazu, 0.00-0.01 dla obrotu o 90°) i zasada, że miara całego obrazu nie pokaże cienia.
- Uczciwe wyniki ujemne (LEACE bez efektu, brak nowego ostrego cienia, kontrola losowa) i podrozdział Ograniczeń; ostrożny wniosek "The data do not show if the model uses one code for both...".
- Rozróżnienie między wektorem wag sondy a wzorcem aktywacji (Haufe et al.) oraz sprawdzenie obu.
- Lista kontrolna jest konkretna; punkt 3 (dopasuj x_t = a x_0 + b epsilon i porównaj z harmonogramem) jest praktyczny. Dodatek A z wzorami geometrii.
- Dobrze skonstruowane krótkie sekcje (4.1, 4.2, "Numeric precision"); Tabela 4 z kolumną "Cubes - pixels".

## F. Pliki, które otworzyłem lub zapisałem

- Otwarty (Read): `<izolowana kopia>/article.pdf` (dwa wywołania Read: pierwsze z parametrem pages zwróciło błąd z powodu braku pdftoppm, drugie zwróciło cały dokument).
- Zapisany (Write): `<notatki>/article_reader.md`
- Żadnych innych plików nie otwierałem ani nie zapisywałem. Nie użyłem Bash, Grep, Glob, wyszukiwania w sieci ani WebFetch.
