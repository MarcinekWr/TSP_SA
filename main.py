import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from itertools import permutations
import time
from functools import wraps
import argparse
import math


def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"Time for {fn.__name__}: {time.perf_counter() - start:.6f}s")
        return result

    return wrapper


@dataclass
class TSPTWInstance:
    n: int
    dist: np.ndarray
    time_windows: list[tuple]
    coords: list[tuple]


NUM_CITIES = 20
BRUTE_FORCE_CITY_LIMIT = 10
TIME_WINDOW = 999_999


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return parsed


def cooling_rate_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("Cooling rate must be in (0, 1)")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSP-TW solved with Simulated Annealing and local 2-opt search"
    )
    parser.add_argument(
        "--num-city",
        "--num_city",
        type=positive_int,
        default=NUM_CITIES,
        help=f"Number of cities in generated instance (default: {NUM_CITIES})",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=10000,
        help="Number of SA iterations (default: 10000)",
    )
    parser.add_argument(
        "--initial-temp",
        type=positive_float,
        default=50.0,
        help="Initial SA temperature (default: 50.0)",
    )
    parser.add_argument(
        "--cooling-rate",
        type=cooling_rate_float,
        default=0.995,
        help="SA cooling rate in range (0, 1), e.g. 0.995 (default: 0.995)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible runs",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="insert_node",
        choices=["two_opt_swap", "swap_nodes", "insert_node"],
        help="Neighborhood move function used by SA (default: insert_node)",
    )
    parser.add_argument(
        "--acceptance-mode",
        "--acceptance_mode",
        type=str,
        default="violation",
        choices=["violation", "distance"],
        help=(
            "Acceptance criterion when both routes are infeasible: "
            "'violation' minimizes sum of time-window violations (default), "
            "'distance' minimizes total route distance (standard SA)"
        ),
    )
    return parser.parse_args()


@timed
def nearest_neighbour(instance: TSPTWInstance):
    visited = {0}
    route = [0]
    current_time = 0

    while len(visited) < instance.n:
        current = route[-1]
        best_next = None
        best_dist = float("inf")

        for city in range(instance.n):
            if city in visited:
                continue

            travel_time = instance.dist[current][city]
            arrival_time = current_time + travel_time
            e, l = instance.time_windows[city]
            if arrival_time < l:
                if travel_time < best_dist:
                    best_dist = travel_time
                    best_next = city
        if best_next is None:
            unvisited = [c for c in range(instance.n) if c not in visited]
            best_next = np.random.choice(unvisited)

        visited.add(best_next)
        route.append(best_next)
        arrival_time = current_time + instance.dist[current][best_next]
        e, _ = instance.time_windows[best_next]
        current_time = max(arrival_time, e)

    route.append(0)
    return [route]


def eval_instance(instance: TSPTWInstance, route: list[tuple]) -> tuple[float, bool]:
    current_time = 0
    total_distance = 0
    flag = True

    for i in range(len(route) - 1):
        current_location = route[i]
        next_location = route[i + 1]

        arrival_time = current_time + instance.dist[current_location][next_location]
        e_next, l_next = instance.time_windows[next_location]

        if arrival_time < e_next:
            current_time = e_next
        elif arrival_time > l_next:
            flag = False
            current_time = arrival_time
        else:
            current_time = arrival_time

        total_distance += instance.dist[current_location][next_location]

    return total_distance, flag


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds / 60:.2f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.2f} h"
    if seconds < 31536000:
        return f"{seconds / 86400:.2f} days"
    return f"{seconds / 31536000:.2f} years"


def estimate_bruteforce_time(instance: TSPTWInstance, sample_count: int = 1000) -> tuple[float, float]:
    """
    Estimate brute-force runtime from average time needed to evaluate one route.
    Returns: (estimated_total_seconds, avg_seconds_per_route)
    """
    sample_count = max(1, sample_count)
    cities = np.arange(1, instance.n)
    start = time.perf_counter()

    for _ in range(sample_count):
        perm = np.random.permutation(cities)
        route = [0, *perm.tolist(), 0]
        eval_instance(instance, route)

    elapsed = time.perf_counter() - start
    avg_per_route = elapsed / sample_count
    total_routes = math.factorial(instance.n - 1)
    estimated_seconds = total_routes * avg_per_route
    return estimated_seconds, avg_per_route


def generate_instance():
    cities_coords = [(10, 10)]
    while len(cities_coords) < NUM_CITIES:
        new_coord = tuple(np.random.randint(0, 21, (2,)))
        if new_coord not in cities_coords:
            cities_coords.append(new_coord)
    cities_coords = np.array(cities_coords)
    dists = np.linalg.norm(cities_coords[:, None] - cities_coords[None, :], axis=2)

    time_windows = [(0, TIME_WINDOW)]
    for i in range(NUM_CITIES):
        earliest = int(dists[0][i])

        window_width = np.random.randint(20, 40)
        shift = np.random.randint(0, 30)
        e = earliest + shift
        l = e + window_width
        time_windows.append((e, l))

    return TSPTWInstance(n=NUM_CITIES, dist=dists, time_windows=time_windows, coords=cities_coords)


def generate_feasible_instance():
    cities_coords = [(10, 10)]
    while len(cities_coords) < NUM_CITIES:
        new_coord = tuple(np.random.randint(0, 21, (2,)))
        if new_coord not in cities_coords:
            cities_coords.append(new_coord)
    cities_coords = np.array(cities_coords)
    dists = np.linalg.norm(cities_coords[:, None] - cities_coords[None, :], axis=2)

    inner = list(range(1, NUM_CITIES))
    np.random.shuffle(inner)
    known_route = [0] + inner + [0]

    time_windows = [None] * NUM_CITIES
    current_time = 0
    for i in range(len(known_route) - 1):
        src, dst = known_route[i], known_route[i + 1]
        current_time += dists[src][dst]
        if dst == 0:
            break
        window_width = np.random.randint(20, 40)
        e = int(current_time)
        l = e + window_width
        time_windows[dst] = (e, l)

    # Okno depotu obejmuje cały czas trasy known_route z marginesem
    depot_return_time = int(current_time)
    time_windows[0] = (0, depot_return_time + 50)

    return TSPTWInstance(n=NUM_CITIES, dist=dists, time_windows=time_windows, coords=cities_coords), known_route


def evaluate_feasibility(tswp_init, routes, feasibles):
    for route in routes:
        calc_dist, feasible = eval_instance(tswp_init, route)
        feasibles.append((int(feasible), calc_dist))


def create_route_permutation():
    route = permutations(list(range(1, NUM_CITIES)))
    routes = [(0, *p, 0) for p in route]
    return routes


@timed
def run_brute_force(instance: TSPTWInstance):
    routes = create_route_permutation()
    feasibles = []
    evaluate_feasibility(instance, routes, feasibles)
    best_brute = min(feasibles, key=lambda x: x[1] if x[0] else float('inf'))
    return best_brute, routes, feasibles


def plot_routes(instance: TSPTWInstance, routes, feasibles):
    x, y = zip(*instance.coords)
    plt.scatter(x, y, zorder=5)

    for i, (cx, cy) in enumerate(instance.coords):
        e, l = instance.time_windows[i]
        plt.annotate(f"City {i} {e}, {l}", (cx, cy), textcoords="offset points", xytext=(-13, 8))

    colors = plt.cm.tab10.colors
    color_idx = 0

    for (f, d), route in zip(feasibles, routes):
        if f:
            coords_ord = [instance.coords[c] for c in route]
            xs, ys = zip(*coords_ord)
            plt.plot(xs, ys, "-o", color=colors[color_idx % len(colors)], label=f"{route}, d={d:.1f}", alpha=0.6)
            color_idx += 1

    plt.legend(fontsize=7)


# SA ----------- Funkcje sąsiedztwa (generatory ruchu) ----------------------

def two_opt_swap(route: list, i: int, j: int) -> list:
    """
    usuwa dwie krawędzie (route[i-1]→route[i])
    i (route[j]→route[j+1]), a następnie łączy segmenty w nowej kolejności,
    odwracając podciąg między i a j.

    Dla trasy [0, A, B, C, D, E, 0] z i=2, j=4:
      Usuwane krawędzie: A→B  oraz  D→E
      Nowa trasa:        [0, A, D, C, B, E, 0]

    Krawędź (route[i-1] → route[i]) jest zastępowana przez (route[i-1] → route[j]),
    a krawędź (route[j] → route[j+1]) przez (route[i] → route[j+1]).
    Segment route[i..j] zostaje odwrócony, co realizuje właśnie to przepięcie.
    """
    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
    return new_route


def swap_nodes(route: list, n: int) -> list:
    """
    Zamiana miejscami dwóch losowo wybranych miast (bez depotu).
    Nie zmienia kolejności pozostałych wierzchołków.
    """
    i, j = np.random.choice(range(1, n), size=2, replace=False)
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def insert_node(route: list, n: int) -> list:
    """
    Wyjęcie jednego miasta z pozycji i i wklejenie go na losową pozycję j.
    Zmiana chirurgiczna – modyfikuje kontekst jednego wierzchołka,
    nie naruszając względnej kolejności pozostałych.
    """
    i = np.random.randint(1, n)
    city = route[i]
    new_route = route[:i] + route[i + 1:]   # wyciągnij miasto
    j = np.random.randint(1, n)              # nowa pozycja (w skróconej liście)
    new_route.insert(j, city)
    return new_route


# Rejestr dostępnych metod – łatwo rozszerzalny
NEIGHBORHOOD_FUNCTIONS = {
    "two_opt_swap": None,       # obsługa osobna – wymaga i, j
    "swap_nodes":   swap_nodes,
    "insert_node":  insert_node,
}


# SA ----------- Akceptacja --------------------------------------------------

def calculate_time_window_violation(instance: TSPTWInstance, route: list[tuple]) -> float:
    """
    Return the total time-window violation (sum of late-arrival delays).
    """
    current_time = 0
    total_violation = 0

    for i in range(len(route) - 1):
        current_location = route[i]
        next_location = route[i + 1]

        arrival_time = current_time + instance.dist[current_location][next_location]
        e_next, l_next = instance.time_windows[next_location]

        if arrival_time > l_next:
            violation = arrival_time - l_next
            total_violation += violation

        if arrival_time < e_next:
            current_time = e_next
        else:
            current_time = arrival_time

    return total_violation


def should_accept_improved(current_route, new_route, instance,
                           current_cost, current_feasible,
                           new_cost, new_feasible, temperature,
                           acceptance_mode: str = "violation"):
    """
    Decide whether to accept a candidate move using feasibility-aware SA logic.

    Parametr ``acceptance_mode`` wybiera strategię akceptacji gdy obie trasy
    są niefeasible:
      - 'violation' : porównuj sumy przekroczeń okien czasowych (domyślne)
      - 'distance'  : porównuj łączną odległość trasy (standardowe SA)
    """
    if not current_feasible and new_feasible:
        return True

    if current_feasible and not new_feasible:
        return False

    if current_feasible and new_feasible:
        delta_cost = new_cost - current_cost
        if delta_cost < 0:
            return True
        probability = np.exp(-delta_cost / temperature)
        return np.random.random() < probability

    if not current_feasible and not new_feasible:
        if acceptance_mode == "distance":
            # Wariant B: porównuj łączną odległość (standardowe kryterium SA)
            delta_cost = new_cost - current_cost
            if delta_cost < 0:
                return True
            probability = np.exp(-delta_cost / temperature)
            return np.random.random() < probability
        else:
            # Wariant A (domyślny): porównuj sumę przekroczeń okien czasowych
            current_violation = calculate_time_window_violation(instance, current_route)
            new_violation = calculate_time_window_violation(instance, new_route)

            if new_violation < current_violation:
                return True

            delta_violation = new_violation - current_violation
            probability = np.exp(-delta_violation / temperature)
            return np.random.random() < probability


# SA ----------- Główny algorytm ---------------------------------------------

@timed
def simulated_annealing(instance: TSPTWInstance, initial_route,
                        num_iterations=10000, initial_temp=100.0,
                        cooling_rate=0.995, verbose=False,
                        progress_interval=1000,
                        method: str = "insert_node",
                        acceptance_mode: str = "violation"):
    """
    Run simulated annealing with feasibility-aware acceptance.

    Parametr ``method`` wybiera funkcję sąsiedztwa:
      - 'two_opt_swap' : prawdziwy ruch 2-opt – usuwa dwie krawędzie i przepina
                         segmenty; odwraca podciąg między indeksami i oraz j
      - 'swap_nodes'   : zamiana miejscami dwóch losowych miast
      - 'insert_node'  : przeniesienie jednego miasta w nowe miejsce trasy
                         (najlepsza skuteczność przy restrykcyjnych oknach czasowych)

    Parametr ``acceptance_mode`` wybiera strategię akceptacji gdy obie trasy są
    niefeasible:
      - 'violation' : minimalizuj sumę przekroczeń okien czasowych (domyślne)
      - 'distance'  : minimalizuj łączną odległość trasy (standardowe SA)
    """
    current_route = list(initial_route)
    current_cost, current_feasible = eval_instance(instance, current_route)

    best_route = current_route
    best_cost = current_cost
    best_feasible = current_feasible

    temperature = initial_temp
    costs_history = []
    feasible_history = []
    violation_history = []

    # Wybierz funkcję generującą nową trasę
    if method == "two_opt_swap":
        def generate_neighbour(route):
            i = np.random.randint(1, instance.n - 1)
            j = np.random.randint(i + 1, instance.n)
            return two_opt_swap(route, i, j)
    elif method == "swap_nodes":
        def generate_neighbour(route):
            return swap_nodes(route, instance.n)
    else:  # insert_node (domyślny)
        def generate_neighbour(route):
            return insert_node(route, instance.n)

    for iteration in range(num_iterations):
        new_route = generate_neighbour(current_route)

        new_cost, new_feasible = eval_instance(instance, new_route)

        should_accept = should_accept_improved(
            current_route, new_route, instance,
            current_cost, current_feasible,
            new_cost, new_feasible, temperature,
            acceptance_mode=acceptance_mode
        )

        if should_accept:
            current_route = new_route
            current_cost = new_cost
            current_feasible = new_feasible

            if new_feasible and (not best_feasible or new_cost < best_cost):
                best_route = new_route
                best_cost = new_cost
                best_feasible = new_feasible

            elif not best_feasible and new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost
                best_feasible = new_feasible

        temperature *= cooling_rate
        costs_history.append(current_cost)
        feasible_history.append(int(current_feasible))
        if not current_feasible:
            violation_history.append(calculate_time_window_violation(instance, current_route))
        else:
            violation_history.append(0)

        if temperature < 1e-8:
            break

        if progress_interval > 0 and (iteration + 1) % progress_interval == 0:
            print(
                f"SA progress: iteration {iteration + 1}/{num_iterations}, "
                f"current_cost={current_cost:.2f}, feasible={current_feasible}"
            )

    return best_route, best_cost, best_feasible, costs_history, feasible_history, violation_history


def main():
    global NUM_CITIES
    args = parse_args()
    NUM_CITIES = args.num_city
    if args.seed is not None:
        np.random.seed(args.seed)

    tswp_init, known_route = generate_feasible_instance()

    total_possible_solutions = math.factorial(args.num_city - 1)
    print(
        f"Possible TSP solutions: ({args.num_city - 1})! = "
        f"{total_possible_solutions:,}"
    )
    estimated_bf_seconds, avg_route_seconds = estimate_bruteforce_time(tswp_init, sample_count=1000)
    print(
        f"Estimated brute force time: {format_duration(estimated_bf_seconds)} "
        f"(avg route eval: {avg_route_seconds * 1e6:.2f} us)"
    )

    best_brute = None
    if args.num_city <= BRUTE_FORCE_CITY_LIMIT:
        best_brute, routes, feasibles = run_brute_force(tswp_init)
        print(f"Brute Force Best: Cost={best_brute[1]:.2f}, Feasible={bool(best_brute[0])}\n")
    else:
        print(
            f"Skipping brute force for {args.num_city} cities "
            f"(limit: {BRUTE_FORCE_CITY_LIMIT}) to avoid excessive memory/time.\n"
        )

    routes_nn = nearest_neighbour(tswp_init)
    nn_cost, nn_feas = eval_instance(tswp_init, routes_nn[0])
    print(f"NN: Cost={nn_cost:.2f}, Feasible={nn_feas}")

    print("\nRunning SA...")
    print(
        f"Params: num_city={args.num_city}, iterations={args.iterations}, "
        f"initial_temp={args.initial_temp}, cooling_rate={args.cooling_rate}, "
        f"method={args.method}, acceptance_mode={args.acceptance_mode}"
    )

    best_route, best_cost, best_feas, history, feas_hist, viol_hist = simulated_annealing(
        tswp_init,
        routes_nn[0],
        num_iterations=args.iterations,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        method=args.method,
        acceptance_mode=args.acceptance_mode,
    )

    print(f"\nSA Best: Costs={best_cost:.2f}, Feasible={best_feas}")

    print(f"\n{'='*60}")
    if best_brute is not None:
        print(f"Brute Force: {best_brute[1]:>8.2f}  (Feasible)")
    else:
        print("Brute Force:    skipped")
    print(f"NN:          {nn_cost:>8.2f}  (Feasible={nn_feas})")
    print(f"SA:          {best_cost:>8.2f}  (Feasible={best_feas}, method={args.method}, acceptance={args.acceptance_mode})")
    print(f"{'='*60}")

    feasible_iter = next((i for i, f in enumerate(feas_hist) if f), None)
    if feasible_iter is not None:
        print(f"Feasible solution found at iteration: {feasible_iter + 1}")
    else:
        print(f"No feasible solution found")


if __name__ == "__main__":
    main()