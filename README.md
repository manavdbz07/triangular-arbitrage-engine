# Triangular Arbitrage Detection Engine
### Theoretical Framework and Algorithmic Implementation

---

## 1. Problem Statement: Triangular Arbitrage in Decentralized Markets

**Market Microstructure Context**
Global foreign exchange (FX) and cryptocurrency markets operate as massive, decentralized networks. Because exchange rates are driven by localized supply and demand across disparate order books, the direct spot price between two assets can temporarily diverge from the implied cross-rate routed through intermediate assets.

**Arbitrage Mechanics**
Theoretically, converting a principal amount of currency A to currency B, then to currency C, and finally back to currency A should yield the exact initial principal (net of transaction fees). However, during periods of high volatility or fragmented liquidity, pricing discrepancies create closed cyclical loops where the final output strictly exceeds the initial input. Executing this cycle constitutes a risk-free exploit known as triangular arbitrage.

**Algorithmic Formulation**
While calculating the profit for a single predefined 3-leg cycle is trivial, real-world markets contain hundreds of tradable assets, generating millions of potential cyclical permutations of varying lengths. Exhaustive search (brute-forcing) is computationally intractable within a quantitative trading environment where pricing inefficiencies dissipate in milliseconds. 

This project solves this engineering constraint by transforming a raw financial matrix of exchange rates into a continuous directed graph, enabling a polynomial-time shortest-path algorithm to systematically isolate profitable loops without brute-force enumeration.

---

## 2. Mathematical Foundation: The Logarithmic Transformation

### 2.1 The Multiplicative Arbitrage Condition
Consider a cycle of $k$ currencies. For each trade leg $i$, let $r_i$ be the net exchange rate (after fees). Arbitrage profit exists when the product of the rates exceeds unity:

$$\prod_{i=1}^{k} r_i > 1$$

This is a multiplicative condition. Standard graph traversal algorithms (e.g., Dijkstra, Bellman-Ford) compute path costs additively. A mathematical transformation is required to align the financial logic with the algorithmic constraints.

### 2.2 The Logarithmic Conversion
Applying the natural logarithm to both sides preserves the inequality direction, as the logarithmic function is monotonically increasing:

$$\log\!\left(\prod_{i=1}^{k} r_i\right) > \log(1) = 0$$

Applying the log-product identity ($\log(ab) = \log a + \log b$) translates the product into a sum:

$$\sum_{i=1}^{k} \log(r_i) > 0$$

### 2.3 Negation and Negative Cycle Detection
The Bellman-Ford algorithm is designed to detect negative weight cycles—cycles where the sum of edge weights is strictly less than zero. Multiplying the inequality by $-1$ aligns the condition with the algorithm:

$$\sum_{i=1}^{k} -\log(r_i) < 0$$

This represents a mathematically exact negative weight cycle. The edge weight $w$ for any directed edge from node $u$ to node $v$ is therefore defined as:

$$w(u \to v) = -\log\!\bigl(r(u \to v) \times (1 - f)\bigr)$$

*(where $f$ represents the per-leg transaction fee fraction).*

---

## 3. Algorithmic Implementation: Bellman-Ford

### 3.1 Algorithm Phases
The engine executes the Bellman-Ford algorithm across two distinct phases:

1. **Initialization:** Set the distance to the source vertex to 0 and all other vertices to $+\infty$.
2. **Relaxation (Phase 1):** Iterate through all graph edges $N-1$ times (where $N$ is the total number of vertices). If the calculated distance to a destination vertex is less than its current known distance, update the distance and record the predecessor vertex.
3. **Detection (Phase 2):** Conduct an $N$-th pass over all edges. If any edge can still be relaxed, the graph contains a negative weight cycle, confirming the presence of an arbitrage opportunity.

### 3.2 Cycle Isolation and Path Reconstruction
Identifying a vertex during the $N$-th relaxation pass does not guarantee that the vertex itself is part of the cycle; it may simply reside on a path leading to it. To accurately isolate the arbitrage loop:
1. Traverse the predecessor array backward for exactly $N$ iterations to ensure the pointer enters the bounds of the negative cycle.
2. Record vertices sequentially until the entry vertex is revisited.
3. Reverse the array to output the forward-facing execution path.

---

## 4. Critical System Analysis and Edge Cases

### 4.1 Impact of Omitting the Logarithmic Transformation
Executing the Bellman-Ford algorithm directly on raw exchange rates invalidates the model:
* **Algebraic Misalignment:** The algorithm aggregates edge weights via addition. Adding raw exchange rates (e.g., $1.09 + 0.84 = 1.93$) yields a mathematically meaningless scalar that does not represent financial profit.
* **Absence of Negative Cycles:** Raw exchange rates are strictly positive real numbers. Because the sum of positive numbers cannot be negative, the algorithm's detection phase will never trigger, rendering the model permanently blind to existing arbitrage.

### 4.2 Impact of Transaction Costs on Edge Weights
Introducing a transaction fee $f$ increases every edge weight by $-\log(1-f)$. For a standard 0.1% exchange fee ($f = 0.001$), each edge weight increases by approximately $0.001001$.
* **The Fee Threshold:** For a 3-leg cycle, the cumulative weight penalty is roughly $0.003$. Consequently, a cycle must generate a gross yield exceeding 1.003 to register as a negative cycle.
* **Zero-Fee Environments:** If $f = 0$, the algorithm becomes hypersensitive, detecting marginal cycles that are profitable only in a theoretical frictionless vacuum but would result in net capital loss during live execution. Transaction fees act as a necessary mathematical filter against phantom arbitrage.

### 4.3 Limitations in Live High-Frequency Trading (HFT) Environments
While this Python engine serves as a rigorous historical detection and research tool, deploying it in live market environments introduces significant execution risks:
* **Execution Latency:** FX arbitrage windows typically close within microseconds. Python’s Global Interpreter Lock (GIL) and garbage collection introduce millisecond-level latency, rendering it uncompetitive against institutional market makers utilizing kernel-bypass networking and FPGA-based order routing.
* **The Observation-Execution Gap:** The engine calculates paths based on static order book snapshots. Due to market micro-volatility, the rates observed at $T=0$ often diverge from the executable rates at $T+10\text{ms}$, leading to negative slippage.
* **Liquidity Constraints:** The model assumes infinite depth at the best bid/ask. In reality, executing large orders against thin liquidity results in partial fills, potentially leaving the system with unhedged directional exposure in an intermediate currency.

---

## 5. File Architecture

| File | Purpose | Core Functions |
|---|---|---|
| `triangular_arbitrage_engine.py` | Complete Detection Engine | `build_synthetic_market`, `build_log_weight_graph`, `bellman_ford`, `calculate_cycle_profit` |
