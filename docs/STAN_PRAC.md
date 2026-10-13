# Stan prac: czy SANA „wie”, skąd pada światło? (13.10.2026, 23:50)

Krótki przegląd dla prowadzącego: co sprawdzamy, co już wiemy, co się teraz liczy i co rozstrzygnie wynik. Pełne szczegóły są w `docs/PODSUMOWANIE.md`.

---

## 1. Pytanie i metoda w jednym zdaniu

Czy aktywacje modelu tekst→obraz **SANA-1.6B** zawierają kierunek światła sceny 3D w formie, której **nie ma już wprost w pikselach**, i czy model tej informacji **używa**?

Narzędziem jest liniowa sonda (PCA, potem regresja Ridge) na uśrednionych aktywacjach bloku $b$ transformera, przy kroku szumu $t$:

$$
\hat{\mathbf y} = W\,\mathrm{PCA}_{256}\big(h_b(x_t)\big) + c
$$

$$
\mathbf y = [\sin\varphi,\ \cos\varphi,\ \theta], \qquad \hat\varphi = \operatorname{atan2}(\hat y_1, \hat y_2)
$$

gdzie $\varphi$ to azymut, a $\theta$ elewacja światła. Błąd azymutu liczymy kołowo: $e = \min(|\hat\varphi-\varphi|,\ 360^\circ - |\hat\varphi-\varphi|)$.

Model dostaje obraz zaszumiony regułą przepływu (*flow matching*), na której był trenowany:

$$
x_t = (1-\sigma)\,x_0 + \sigma\,\varepsilon,\qquad \sigma = t/1000,\qquad \varepsilon\sim\mathcal N(0, I)
$$

```mermaid
flowchart LR
    A[Render Blendera<br/>512×512] --> B[Enkoder DC-AE<br/>latent 32×32×32]
    B --> C["Szum: x_t = (1-σ)x0 + σε"]
    C --> D[Transformer SANA<br/>20 bloków]
    D --> E[Aktywacje bloku b<br/>1024 tokeny × 2240]
    E --> F[Uśrednienie do siatki<br/>4×4 lub 8×8]
    F --> G[PCA 256 + Ridge<br/>α wybrane na walidacji]
    G --> H[Azymut, elewacja<br/>błąd MAE + 95% CI]
```

---

## 2. Co już wiemy

| Ustalenie | Liczby |
|---|---|
| Wynik z ZZSN (SANA, blok 10, t = 200) **nie bije pikseli** | sześciany: SANA 14,0°, ta sama sonda na pikselach 14,1° |
| W starym kodzie **szum był błędny** (pułapka w bibliotece diffusers, zgłoszona jako issue #15021) | dla t = 200 model dostawał $-24{,}5\,x_0 + 25{,}5\,\varepsilon$ zamiast $0{,}8\,x_0 + 0{,}2\,\varepsilon$ |
| Etykieta to **pozycja lampy wokół środka sceny**, a nie światło przy obiekcie | różnica średnio 35° w azymucie |
| Sondę niosła **plama światła na podłodze**, nie obiekt | sama podłoga 9,4°, sam obiekt 44,3° (sześciany) |
| Nowy pipeline **odtwarza stary** we wszystkim poza szumem | norma tokenu ze starym szumem: 11 147 (stare dane: ok. 11 100), z poprawnym: 718 |

Gdzie sonda pikselowa czyta kierunek (dokładny rozkład sondy liniowej na komórki obrazu; jaśniej = większy udział):

![Mapy sondy pikselowej: v1 i v2](figures/probe_maps_pixels.png)

Na starych danych (v1) waga leży na całej podłodze, gdzie wędruje plama od lampy.

---

## 3. Dlaczego powstały nowe dane

```mermaid
flowchart TD
    V1["<b>v1 (ZZSN)</b><br/>lampa krąży wokół środka, plama na podłodze"] -->|piksele = SANA| Q1{Czy to wiedza 3D?}
    Q1 -->|nie: skrót przez podłogę| V2["<b>v2</b><br/>lampa SUN (sam kierunek), płaska podłoga,<br/>bez plamy, obiekt w środku"]
    V2 -->|piksele: 2,5°| Q2{Czy to wystarczy?}
    Q2 -->|nie: cień zawsze w tym samym miejscu kadru| V2R["<b>v2r</b><br/>to samo światło, obiekt w losowym miejscu"]
    V2R --> T["Test rozstrzygający:<br/>czy SANA bije piksele, gdy położenie cienia<br/>zależy od położenia obiektu?"]
    V2R --> V["Warianty v2rns / v2rso:<br/>ta sama scena bez cienia / sam cień"]
```

| Zbiór | Co zmienia | Liczba obrazów | Stan |
|---|---|---:|---|
| v2 | jedno światło kierunkowe, płaska podłoga, obiekt w środku | 3000 | gotowy |
| v2pos | jak v2, obiekt w losowym miejscu (tylko test) | 300 | gotowy |
| v2ns | jak v2, obiekt nie rzuca cienia | 3000 | 4 z 6 części |
| **v2r** | jak v2 (to samo światło, obraz w obraz), obiekt w losowym miejscu | 3000 | renderuje się |
| v2rns, v2rso | jak v2r: bez cienia / sam cień | 2 × 3000 | w kolejce |

![12 obrazów v2](figures/v2_grid.png)

Ta sama scena i to samo światło: v2 (góra, obiekt w środku) i v2r (dół, obiekt w losowym miejscu):

![Pary v2 i v2r](figures/v2_v2r_pairs.png)

Pozycja w v2r jest losowana tylko tam, gdzie największy obiekt mieści się cały w kadrze. Ten obszar zależy wyłącznie od kamery, więc pozycja nie koreluje ze światłem ($|r| < 0{,}01$).

---

## 4. Pierwsze wyniki na nowych danych (tylko piksele)

| Sonda na pikselach 64×64 | Kule i walce | Sześciany |
|---|---:|---:|
| v2: uczona i testowana na v2 | **2,5°** | 13,7° |
| v2: test na obiektach w losowych miejscach (v2pos) | **69,0°** (losowo: 90°) | |
| uczona na v2, test na **tych samych scenach bez cienia** | **77,4°** | |
| uczona i testowana na v2ns (samo cieniowanie obiektu) | 4,2° | |

Ta sama scena z cieniem (góra) i bez cienia (dół):

![Pary scen: z cieniem i bez](figures/v2_pairs_shadow.png)

Wnioski:
1. Przy obiekcie w środku **piksele wystarczają** do kierunku (2,5°). Na takim zbiorze nie da się pokazać przewagi modelu.
2. Sonda pikselowa **opiera się na cieniu**: po jego usunięciu błąd rośnie z 2,5° do 77°.
3. Piksele **nie przenoszą** kierunku na nowe pozycje (69°). To jest próg, który SANA musi pobić na v2r.
4. **Elewację zdradza jasność podłogi** (1,7° z samej podłogi), więc główną miarą jest azymut.

---

## 5. Kontrole: co każda wyklucza

| Kontrola | Co wyklucza, jeśli SANA jest wyraźnie lepsza |
|---|---|
| ta sama sonda na **pikselach** | informacja jest w obrazie w prostej, liniowej formie |
| ta sama sonda na **latencie DC-AE** | informację daje już enkoder obrazu, a nie transformer |
| SANA z **losowymi wagami** | wynik daje sama architektura, a nie uczenie |
| SANA ze **starym szumem** | wynik ZZSN był artefaktem złego szumu |
| **SD 1.5 i SDXL** | wynik jest cechą jednego modelu |
| wybór bloku i $t$ **tylko na walidacji**, 95% CI z bootstrapu | dopasowanie do zbioru testowego |

---

## 6. Test przyczynowy: czy model *używa* kierunku

Sonda pokazuje tylko, że informację da się odczytać. Żeby sprawdzić, czy model jej używa, zmieniamy aktywacje podczas generowania obrazu i patrzymy, czy przesuwa się cień.

1. Render v2 kodujemy, zaszumiamy do $\sigma_0 = 0{,}7$ i odszumiamy w 20 krokach Eulera (SDEdit): $x_{\sigma'} = x_\sigma + (\sigma' - \sigma)\,v_\theta(x_\sigma, \sigma)$.
2. W każdym kroku hook w bloku $b$ zmienia tokeny na jeden z trzech sposobów:
   - **przesunięcie wzdłuż sondy** do azymutu $\varphi + 90^\circ$, najkrótszym wektorem: $\delta = W^\top (W W^\top)^{-1} \Delta\mathbf y$, przeskalowanym przez $s \in \{1, 4, 16\}$;
   - **losowe przesunięcie** o tej samej normie (kontrola);
   - **wymazanie LEACE**: $r(h) = h - A\,(h - \mu)$; po nim żadna funkcja liniowa nie koreluje z $(\sin\varphi, \cos\varphi)$.
3. Azymut obrazu wyjściowego mierzy sonda pikselowa.

```mermaid
flowchart LR
    R[Render v2] --> N["Szum do σ0 = 0,7"]
    N --> S["20 kroków Eulera<br/>SANA"]
    H["Hook w bloku b:<br/>+δ (sonda) / +losowy / LEACE"] -.-> S
    S --> O[Obraz wyjściowy]
    O --> P["Sonda pikselowa:<br/>o ile obrócił się cień?"]
```

Jeśli kierunek jest przyczynowy, przesunięcie wzdłuż sondy obraca cień w stronę celu i bardziej niż przesunięcie losowe. Jeśli model go używa, wymazanie psuje zgodność cienia z etykietą.

---

## 7. Co się teraz liczy

Wszystko liczy się na Kaggle (klaster Athena wyczerpał przydział GPU). Kolejka sama wysyła joby (najwyżej 5 CPU i 2 GPU naraz) i pobiera wyniki, a automat na laptopie sam liczy sondy, gdy zestaw jest kompletny. Czasy poniżej to szacunek.

```mermaid
gantt
    title Obliczenia na Kaggle, 13-14.10 (szacunek)
    dateFormat YYYY-MM-DD HH:mm
    axisFormat %H:%M
    section CPU: rendery
    v2 (3000)              :done, 2026-10-13 21:07, 2026-10-13 22:33
    v2pos (300)            :done, 2026-10-13 22:07, 2026-10-13 22:54
    v2ns (3000)            :active, 2026-10-13 22:25, 2026-10-14 01:00
    v2r (3000)             :active, 2026-10-13 22:55, 2026-10-14 01:00
    v2rns, v2rso (6000)    :2026-10-14 01:00, 2026-10-14 04:00
    section GPU: cechy i testy
    próby techniczne       :done, 2026-10-13 21:09, 2026-10-13 22:32
    v1 (5 modeli i kontroli) :active, 2026-10-13 22:36, 2026-10-14 02:00
    próba testu przyczynowego :2026-10-13 23:55, 2026-10-14 00:10
    v2                     :2026-10-14 02:00, 2026-10-14 04:30
    v2pos, v2r             :2026-10-14 04:30, 2026-10-14 07:00
    v2rns, v2rso           :2026-10-14 07:00, 2026-10-14 09:00
```

Zużycie GPU: ok. 20 h z tygodniowego limitu 30 h na jednym koncie.

---

## 8. Co rozstrzygnie wynik

| Wynik na v2r (sześciany i nowe pozycje) | Wniosek do artykułu |
|---|---|
| SANA **wyraźnie lepsza** od pikseli, latentu i losowych wag, a przesunięcie wzdłuż sondy obraca cień | model ma **reprezentację kierunku światła niezależną od pozycji** i jej używa (wynik pozytywny, przyczynowy) |
| SANA lepsza od kontroli, ale przesunięcie nic nie zmienia | informacja jest w modelu, ale ten kierunek nie steruje cieniem (reprezentacja „uboczna”) |
| SANA **nie lepsza** od kontroli | brak dowodu na wiedzę 3D; wynik negatywny z pełnymi kontrolami (wartościowy dla TMLR) |

W każdym z trzech przypadków powstaje artykuł z jasną tezą. Do tego dochodzą lista kontrolna dla badań sondami i poprawka do biblioteki diffusers.
