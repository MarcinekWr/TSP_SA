## TSP-SA

Projekt realizuje problem komiwojażera (TSP) w rozszerzonej wersji z oknami czasowymi.

Problem rozwiązywany jest za pomocą algorytmu symulowanego wyżarzania (SA) w rozszerzonej wersji z lokalnym przeszukiwaniem 2-opt.

### Wymagania

- Python 3.11 lub nowszy
- `uv`

### Instalacja

1. Utwórz środowisko i zainstaluj zależności:

```bash
uv sync
```

2. Uruchom projekt:

```bash
uv run python main.py
```

Możesz też uruchomić go przez zdefiniowany skrypt:

```bash
uv run tsp-sa
```

Uruchomienie komendą `uv run tsp-sa` działa bez podawania parametrów.
W takim przypadku program uruchamia się na ustawieniach bazowych zapisanych w kodzie (`main.py`):
- `num_city=7`
- `iterations=5000`
- `initial_temp=50.0`
- `cooling_rate=0.995`

Możesz też uruchamiać program z parametrami:

```bash
uv run tsp-sa --num-city 10 --iterations 12000 --initial-temp 80 --cooling-rate 0.997
```

Dla powtarzalnych wyników możesz dodać ziarno losowe:

```bash
uv run tsp-sa --seed 42
```

Lista wszystkich dostępnych opcji:

```bash
uv run tsp-sa --help
```

### Opis działania

- Generowana jest losowa instancja TSP z oknami czasowymi.
- Najpierw wyznaczane jest rozwiązanie metodą brute force dla porównania.
- Następnie uruchamiana jest heurystyka najbliższego sąsiada.
- Na końcu wykonywane jest symulowane wyżarzanie z ruchem 2-opt i oceną zgodności z oknami czasowymi.
