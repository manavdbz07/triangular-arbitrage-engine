"""
================================================================================
  TRIANGULAR ARBITRAGE DETECTION ENGINE
  Using Graph Theory & the Bellman-Ford Algorithm — Pure Python + NumPy
================================================================================

AUTHOR  : Quantitative Systems Design
PURPOSE : Detect profitable triangular arbitrage cycles in a FX rate graph
          without using any black-box graph libraries (e.g., NetworkX).

CORE MATHEMATICAL INSIGHT
--------------------------
Arbitrage exists when a cycle of trades yields a product of exchange rates > 1:

    r(A→B) × r(B→C) × r(C→A) > 1

To use shortest-path algorithms (which operate on sums, not products), we apply
the negative logarithm transform to each edge weight:

    w(u→v)  =  −log( r(u→v) × (1 − fee) )

By log-product identity:  log(a×b) = log(a) + log(b)
Therefore the cycle profit condition becomes:

    −log(r₁) + −log(r₂) + −log(r₃) < 0    ← a NEGATIVE weight cycle

Bellman-Ford detects exactly this. Full proof is in the companion explainer.
================================================================================
"""

import math
import random
import numpy as np
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ Market Environment Construction
# ─────────────────────────────────────────────────────────────────────────────

# Currency labels — indices 0..4 map to these names
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD"]
N = len(CURRENCIES)

# Transaction fee: 0.1% per trade leg (Binance spot taker fee as reference)
TRANSACTION_FEE = 0.001   # 0.1%

# A realistic synthetic FX rate matrix (mid-market approximations, Sept 2026)
# RAW_RATES[i][j]  =  how many units of currency j you receive per 1 unit of i
# RAW_RATES[i][i]  =  0.0  (no self-loop; undefined / unused)
#
# We deliberately embed ONE mild arbitrage opportunity in the EUR→GBP→CAD→USD
# cycle so we can confirm the engine finds it.
#
#              USD     EUR     GBP     JPY      CAD
RAW_RATES_BASE = np.array([
    # USD →
    [0.0000, 0.9200, 0.7800, 149.50, 1.3600],
    # EUR →
    [1.0870, 0.0000, 0.8478, 162.54, 1.4783],
    # GBP →
    [1.2821, 1.1794, 0.0000, 191.67, 1.7436],
    # JPY →
    [0.0067, 0.0062, 0.0052, 0.0000, 0.0091],
    # CAD →
    [0.7353, 0.6764, 0.5736, 110.00, 0.0000],
], dtype=np.float64)


def inject_arbitrage_opportunity(
    rates: np.ndarray,
    cycle: list[int],
    profit_target: float = 1.004   # ~0.4% gross profit before fees
) -> np.ndarray:
    """
    Inject a detectable (but still realistic) arbitrage opportunity into the
    rate matrix along a specified cycle, for demonstration purposes.

    The function scales the LAST edge of the cycle upward so that the product
    of rates along the full cycle equals `profit_target`.

    Parameters
    ----------
    rates        : N×N exchange rate matrix (modified in place on a copy)
    cycle        : list of currency indices, e.g. [0, 1, 2, 0] for USD→EUR→GBP→USD
    profit_target: desired gross multiplier across the cycle

    Returns
    -------
    Modified rate matrix (copy)
    """
    rates = rates.copy()

    # Product of all edges except the last one
    partial_product = 1.0
    for k in range(len(cycle) - 2):
        u, v = cycle[k], cycle[k + 1]
        partial_product *= rates[u][v]

    # Solve for the last edge rate needed to hit profit_target
    u_last, v_last = cycle[-2], cycle[-1]
    required_last_rate = profit_target / partial_product
    rates[u_last][v_last] = required_last_rate

    return rates


def build_synthetic_market(
    seed: int = 42,
    inject_cycle: Optional[list[int]] = None,
    profit_target: float = 1.004,
    fee: float = TRANSACTION_FEE
) -> tuple[np.ndarray, list[str]]:
    """
    Build a synthetic FX rate matrix with optional noise and an injected
    arbitrage cycle.

    Parameters
    ----------
    seed         : RNG seed for reproducibility
    inject_cycle : currency index path to inject an arb opportunity into
    profit_target: gross multiplier for the injected cycle
    fee          : per-leg transaction fee (not yet applied here — raw rates only)

    Returns
    -------
    (rate_matrix, currency_names)
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Apply small random spread noise (±0.05%) to simulate bid/ask midpoint drift
    noise = 1.0 + np.random.uniform(-0.0005, 0.0005, size=(N, N))
    np.fill_diagonal(noise, 1.0)

    rates = RAW_RATES_BASE * noise
    np.fill_diagonal(rates, 0.0)

    if inject_cycle:
        rates = inject_arbitrage_opportunity(rates, inject_cycle, profit_target)

    return rates, CURRENCIES


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ Graph Transformation
# ─────────────────────────────────────────────────────────────────────────────

def build_log_weight_graph(
    rate_matrix: np.ndarray,
    fee: float = TRANSACTION_FEE
) -> list[tuple[int, int, float]]:
    """
    Transform the raw exchange rate matrix into a directed edge list with
    log-transformed weights suitable for Bellman-Ford.

    Transformation applied per edge (u → v):
        net_rate(u→v)  =  rate_matrix[u][v] × (1 − fee)
        weight(u→v)    =  −log( net_rate(u→v) )

    Mathematical consequence:
        • If net_rate > 1  →  log(net_rate) > 0  →  weight < 0  (cheap edge)
        • A profitable cycle means Π net_rate > 1
          ⟺  log(Π net_rate) > 0
          ⟺  Σ log(net_rate_i) > 0
          ⟺  Σ −log(net_rate_i) < 0   ← NEGATIVE CYCLE in Bellman-Ford

    Parameters
    ----------
    rate_matrix : N×N raw exchange rate array
    fee         : per-leg transaction fee fraction

    Returns
    -------
    List of (source_index, dest_index, weight) tuples — one per valid edge
    """
    edges = []
    n = rate_matrix.shape[0]

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            raw_rate = rate_matrix[u][v]
            if raw_rate <= 0.0:
                continue  # skip undefined / zero rates

            net_rate = raw_rate * (1.0 - fee)

            if net_rate <= 0.0:
                # Guard against log(≤0) which is undefined
                continue

            weight = -math.log(net_rate)
            edges.append((u, v, weight))

    return edges


def print_transformed_graph(
    edges: list[tuple[int, int, float]],
    currencies: list[str]
) -> None:
    """Pretty-print the transformed edge list with raw and log weights."""
    print("\n" + "═" * 70)
    print("  TRANSFORMED GRAPH  (−log weights, fee-adjusted)")
    print("═" * 70)
    print(f"  {'Edge':<18}  {'Net Rate':>12}  {'Weight = −log(rate)':>22}")
    print("─" * 70)
    for u, v, w in edges:
        net_rate = math.exp(-w)   # reconstruct from weight for display
        edge_label = f"{currencies[u]} → {currencies[v]}"
        print(f"  {edge_label:<18}  {net_rate:>12.6f}  {w:>22.8f}")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ Bellman-Ford Engine (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

INF = float("inf")


def bellman_ford(
    n_vertices: int,
    edges: list[tuple[int, int, float]],
    source: int = 0
) -> tuple[list[float], list[Optional[int]], Optional[int]]:
    """
    Classic Bellman-Ford shortest-path algorithm with negative cycle detection.

    Algorithm Overview
    ------------------
    1. Initialise dist[source] = 0, dist[v] = +∞ for all other v.
    2. Relax every edge (u, v, w) exactly (N−1) times.
       After k relaxations, dist[v] holds the shortest path using at most k edges.
    3. On the N-th iteration, if ANY edge can STILL be relaxed, a negative
       weight cycle is reachable from `source`. Record the "entry vertex"
       into that cycle.

    Time Complexity  : O(V × E)
    Space Complexity : O(V)

    Parameters
    ----------
    n_vertices : total number of vertices (currencies)
    edges      : list of (u, v, w) directed edges
    source     : starting vertex index

    Returns
    -------
    dist          : shortest-distance array after (N−1) relaxations
    predecessor   : predecessor array for path reconstruction
    neg_cycle_v   : a vertex known to be ON a negative cycle, or None
    """
    dist        = [INF] * n_vertices
    predecessor = [None] * n_vertices

    dist[source] = 0.0

    # ── Phase 1: Relax all edges (N − 1) times ──────────────────────────────
    for iteration in range(n_vertices - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] == INF:
                continue
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                predecessor[v] = u
                updated = True

        # Early exit: if no relaxation occurred in this full pass, no further
        # relaxation is possible (the graph has converged).
        if not updated:
            break

    # ── Phase 2: N-th relaxation pass — detect negative cycles ──────────────
    neg_cycle_vertex = None

    for u, v, w in edges:
        if dist[u] == INF:
            continue
        if dist[u] + w < dist[v]:
            # This vertex v is either ON or REACHABLE FROM a negative cycle.
            # We'll trace back via predecessor to find a vertex guaranteed
            # to be INSIDE the cycle.
            neg_cycle_vertex = v
            # Update predecessor so backtrace correctly follows the cycle.
            predecessor[v] = u
            break   # one confirmation is enough; backtrace handles the rest

    return dist, predecessor, neg_cycle_vertex


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ Path Reconstruction & Profit Calculation
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_negative_cycle(
    neg_cycle_vertex: int,
    predecessor: list[Optional[int]],
    n_vertices: int
) -> list[int]:
    """
    Given a vertex known to be on (or reachable from) a negative cycle,
    trace the predecessor chain to isolate exactly the vertices IN the cycle.

    Technique
    ---------
    Walk backward through `predecessor` for N steps starting from
    `neg_cycle_vertex`. After N steps we are guaranteed to be inside the cycle
    (not just approaching it). Then collect the cycle by following predecessor
    until we see the entry vertex again.

    Parameters
    ----------
    neg_cycle_vertex : any vertex confirmed to be relaxed in the N-th pass
    predecessor      : predecessor array from Bellman-Ford
    n_vertices       : total vertex count (used to guarantee we're in the cycle)

    Returns
    -------
    List of vertex indices forming the cycle (first == last element)
    """
    # Step 1: Walk N steps into the cycle to escape any "tail" that merely
    # leads TO the cycle without being part of it.
    v = neg_cycle_vertex
    for _ in range(n_vertices):
        v = predecessor[v]

    # Step 2: Record the cycle starting from v
    cycle = []
    current = v
    while True:
        cycle.append(current)
        current = predecessor[current]
        if current == v:
            cycle.append(current)  # close the loop
            break

    # Step 3: Reverse so it reads in forward trade direction
    cycle.reverse()
    return cycle


def calculate_cycle_profit(
    cycle: list[int],
    rate_matrix: np.ndarray,
    fee: float,
    currencies: list[str]
) -> tuple[float, float]:
    """
    Given an arbitrage cycle (list of vertex indices), compute the gross
    multiplier and net percentage profit.

    For a cycle  [A, B, C, A]:
        gross = r(A→B) × r(B→C) × r(C→A)
        net   = r(A→B)×(1−fee) × r(B→C)×(1−fee) × r(C→A)×(1−fee)

    Parameters
    ----------
    cycle       : ordered list of vertex indices (first == last)
    rate_matrix : N×N raw exchange rate matrix
    fee         : per-leg transaction fee
    currencies  : currency name list

    Returns
    -------
    (gross_multiplier, net_profit_pct)
    """
    gross = 1.0
    net   = 1.0
    legs  = len(cycle) - 1   # number of trade legs

    print("\n" + "═" * 70)
    print("  ARBITRAGE CYCLE DETAIL")
    print("═" * 70)

    for k in range(legs):
        u, v        = cycle[k], cycle[k + 1]
        raw_rate    = rate_matrix[u][v]
        net_rate    = raw_rate * (1.0 - fee)
        gross      *= raw_rate
        net        *= net_rate

        print(
            f"  Leg {k+1}: {currencies[u]:>3} → {currencies[v]:<3}  "
            f"| Raw rate = {raw_rate:.6f}  "
            f"| Net (after {fee*100:.2f}% fee) = {net_rate:.6f}"
        )

    gross_profit_pct = (gross - 1.0) * 100.0
    net_profit_pct   = (net   - 1.0) * 100.0

    print("─" * 70)
    print(f"  Gross multiplier  : {gross:.8f}  ({gross_profit_pct:+.4f}%)")
    print(f"  Net  multiplier   : {net:.8f}   ({net_profit_pct:+.4f}%)")
    print("═" * 70)

    return gross, net_profit_pct


def print_trade_sequence(
    cycle: list[int],
    currencies: list[str]
) -> None:
    """Print the human-readable trade execution sequence."""
    path_str = "  →  ".join(currencies[v] for v in cycle)
    print(f"\n  TRADE SEQUENCE: {path_str}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ Full Engine Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_engine(
    rate_matrix: np.ndarray,
    currencies: list[str],
    fee: float = TRANSACTION_FEE,
    source: int = 0,
    verbose: bool = True
) -> None:
    """
    Main orchestration function. Runs the full pipeline:
      1. Build log-weight graph
      2. Run Bellman-Ford from each source vertex (to find ALL reachable cycles)
      3. Reconstruct and report any negative cycles found

    Note on scanning from all sources
    ----------------------------------
    A negative cycle reachable from vertex S may NOT be reachable from vertex 0.
    To guarantee exhaustive detection, we run Bellman-Ford once per source
    vertex, giving us O(V × V × E) total time — perfectly acceptable for N=5.

    Parameters
    ----------
    rate_matrix : N×N raw exchange rate matrix
    currencies  : list of currency name strings
    fee         : per-leg transaction fee
    source      : starting vertex for the first pass (we iterate over all)
    verbose     : if True, print transformed graph and detailed logs
    """
    print("\n" + "█" * 70)
    print("  TRIANGULAR ARBITRAGE DETECTION ENGINE")
    print("█" * 70)
    print(f"  Currencies : {', '.join(currencies)}")
    print(f"  Fee/leg    : {fee*100:.3f}%")
    print(f"  Graph size : {N} vertices, {N*(N-1)} directed edges")

    # ── Step 1: Transform the graph ─────────────────────────────────────────
    edges = build_log_weight_graph(rate_matrix, fee)

    if verbose:
        print_transformed_graph(edges, currencies)

    # ── Step 2: Scan from every source vertex ────────────────────────────────
    found_cycles: list[tuple[list[int], float]] = []   # (cycle, net_profit_pct)
    seen_cycle_signatures: set[frozenset] = set()

    print("\n" + "─" * 70)
    print("  BELLMAN-FORD SCAN  (running from each source vertex)")
    print("─" * 70)

    for src in range(len(currencies)):
        print(f"\n  ▶ Source: {currencies[src]}", end="")

        dist, predecessor, neg_v = bellman_ford(len(currencies), edges, source=src)

        if neg_v is None:
            print("  →  No negative cycle reachable.")
            continue

        # Reconstruct the cycle
        cycle = reconstruct_negative_cycle(neg_v, predecessor, len(currencies))

        # De-duplicate: treat cycles as equivalent regardless of rotation
        sig = frozenset(zip(cycle[:-1], cycle[1:]))
        if sig in seen_cycle_signatures:
            print(f"  →  Cycle already found (duplicate rotation). Skipping.")
            continue
        seen_cycle_signatures.add(sig)

        print(f"  →  ⚠  NEGATIVE CYCLE DETECTED!")

        # Print trade sequence
        print_trade_sequence(cycle, currencies)

        # Calculate profit
        _, net_profit = calculate_cycle_profit(cycle, rate_matrix, fee, currencies)
        found_cycles.append((cycle, net_profit))

    # ── Step 3: Summary Report ───────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  FINAL SUMMARY")
    print("█" * 70)

    if not found_cycles:
        print("  ✓ No arbitrage opportunities found. Market is in equilibrium.")
    else:
        print(f"  ⚠  {len(found_cycles)} arbitrage opportunity/ies detected:\n")
        for i, (cycle, net_pct) in enumerate(found_cycles, 1):
            path = " → ".join(currencies[v] for v in cycle)
            status = "PROFITABLE ✓" if net_pct > 0 else "UNPROFITABLE after fees ✗"
            print(f"  [{i}] {path}")
            print(f"       Net profit: {net_pct:+.4f}%  |  {status}")
    print("█" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ Diagnostic Mode: Raw Rates (No Log Transform)
# ─────────────────────────────────────────────────────────────────────────────

def bellman_ford_raw_rates(
    n_vertices: int,
    rate_matrix: np.ndarray,
    source: int = 0
) -> tuple[list[float], list[Optional[int]], Optional[int]]:
    """
    Runs Bellman-Ford using RAW exchange rates as edge weights (no log transform).

    THIS IS INTENTIONALLY BROKEN — used only for educational demonstration.

    What goes wrong:
    ─────────────────
    • Bellman-Ford uses ADDITION: dist[v] = dist[u] + w(u,v)
    • Profitable arbitrage is a MULTIPLICATIVE condition: Π rate_i > 1
    • Adding raw rates has NO meaningful financial interpretation.
    • Even a clearly profitable cycle will NOT produce a "negative" sum
      (rates are all positive > 0), so Bellman-Ford will NEVER flag it.
    • Worse, very high rates (e.g., USD→JPY ≈ 149) will make Bellman-Ford
      report a "shortest path" that minimises the SUM of rates — completely
      orthogonal to finding profitable multiplicative loops.

    This function is included purely for the Critical Analysis section.
    """
    # Build edges from raw rates (treating them directly as weights)
    raw_edges = []
    for u in range(n_vertices):
        for v in range(n_vertices):
            if u == v:
                continue
            w = rate_matrix[u][v]
            if w > 0:
                raw_edges.append((u, v, w))

    dist        = [INF] * n_vertices
    predecessor = [None] * n_vertices
    dist[source] = 0.0

    for _ in range(n_vertices - 1):
        for u, v, w in raw_edges:
            if dist[u] == INF:
                continue
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                predecessor[v] = u

    neg_v = None
    for u, v, w in raw_edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            neg_v = v
            break

    return dist, predecessor, neg_v


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   TRIANGULAR ARBITRAGE DETECTION ENGINE — Demo Run                  ║
║   Pure Python + NumPy | Bellman-Ford | No black-box graph libraries  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # ── Demo 1: No injected opportunity (should find none / marginal) ────────
    print("=" * 70)
    print("  DEMO 1 — Baseline market (no injected opportunity)")
    print("=" * 70)
    rates_baseline, ccy = build_synthetic_market(seed=42, inject_cycle=None)
    run_engine(rates_baseline, ccy, fee=TRANSACTION_FEE, verbose=False)

    # ── Demo 2: Inject a USD→EUR→GBP→USD arbitrage loop ─────────────────────
    print("=" * 70)
    print("  DEMO 2 — Market with injected USD → EUR → GBP → USD arb loop")
    print("         (gross ~0.4% profit injected; net after 3×0.1% fees)")
    print("=" * 70)
    inject_path = [
        CURRENCIES.index("USD"),
        CURRENCIES.index("EUR"),
        CURRENCIES.index("GBP"),
        CURRENCIES.index("USD"),
    ]
    rates_with_arb, ccy = build_synthetic_market(
        seed=42,
        inject_cycle=inject_path,
        profit_target=1.004
    )
    run_engine(rates_with_arb, ccy, fee=TRANSACTION_FEE, verbose=True)

    # ── Demo 3: Zero-fee scenario (should find MORE cycles) ──────────────────
    print("=" * 70)
    print("  DEMO 3 — Zero-fee scenario (fee = 0%)")
    print("         Expect: more cycles detected due to removed friction")
    print("=" * 70)
    run_engine(rates_with_arb, ccy, fee=0.0, verbose=False)

    # ── Demo 4: Diagnostic — raw rates (broken, educational) ─────────────────
    print("=" * 70)
    print("  DEMO 4 — Diagnostic: Bellman-Ford on RAW rates (no log transform)")
    print("         Expected: FAILS to detect arbitrage — see Critical Analysis")
    print("=" * 70)
    dist_raw, pred_raw, neg_v_raw = bellman_ford_raw_rates(
        N, rates_with_arb, source=0
    )
    if neg_v_raw is None:
        print("""
  RESULT: No negative cycle found — CORRECT failure.
  Reason: All raw exchange rates are positive numbers.
          Bellman-Ford on raw rates computes min-sum-of-rates, which has
          no relationship to multiplicative profit. Positive rates can
          never sum to a negative cycle. The arb opportunity is invisible.
        """)
    else:
        print(f"  RESULT: Spurious cycle flagged at vertex {neg_v_raw} — incorrect.")
