import math, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, yfinance as yf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD"]
N = 5
START, END = "2023-01-01", "2024-12-31"
FEE_IB, FEE_RT = 0.0002, 0.0010           # interbank / retail
BASE  = Path(__file__).parent
PLOTS = BASE / "plots"; PLOTS.mkdir(exist_ok=True)
INF   = float("inf")

DARK, GRID, TEXT   = "#0d1117", "#21262d", "#e6edf3"
BLUE, GREEN, RED, YELLOW = "#58a6ff", "#3fb950", "#f85149", "#d29922"
plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": GRID, "text.color": TEXT,
    "legend.facecolor": "#161b22", "legend.edgecolor": GRID,
    "font.family": "monospace",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── 1. Fetch FX data ──────────────────────────────────────────────────────────
def fetch():
    print("\n[1/4] Fetching real FX data from Yahoo Finance...")
    tickers = {"EUR": "USDEUR=X", "GBP": "USDGBP=X",
               "JPY": "USDJPY=X", "CAD": "USDCAD=X"}
    frames = {}
    for ccy, ticker in tickers.items():
        df = yf.download(ticker, start=START, end=END,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"Download failed: {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        frames[ccy] = df["Close"].rename(ccy)
        print(f"   OK {ticker}  ({len(df)} trading days)")
    combined = pd.concat(frames.values(), axis=1).dropna()
    combined.index = pd.to_datetime(combined.index)
    combined.to_csv(BASE / "fx_rates_raw.csv")
    print(f"   Saved fx_rates_raw.csv  ({len(combined)} rows)")
    return combined

# ── 2. Rate matrix builder ────────────────────────────────────────────────────
def build_matrix(row):
    u = {"USD": 1.0, "EUR": row["EUR"], "GBP": row["GBP"],
         "JPY": row["JPY"], "CAD": row["CAD"]}
    m = np.zeros((N, N))
    for i, a in enumerate(CURRENCIES):
        for j, b in enumerate(CURRENCIES):
            if i != j:
                m[i][j] = u[b] / u[a]
    return m

# ── 3. Bellman-Ford engine ────────────────────────────────────────────────────
def get_edges(m, fee):
    return [(u, v, -math.log(m[u][v] * (1 - fee)))
            for u in range(N) for v in range(N)
            if u != v and m[u][v] > 0 and m[u][v] * (1 - fee) > 0]

def bf(edg, src):
    d = [INF] * N; p = [None] * N; d[src] = 0.0
    for _ in range(N - 1):
        upd = False
        for u, v, w in edg:
            if d[u] != INF and d[u] + w < d[v]:
                d[v] = d[u] + w; p[v] = u; upd = True
        if not upd:
            break
    nv = None
    for u, v, w in edg:
        if d[u] != INF and d[u] + w < d[v]:
            p[v] = u; nv = v; break
    return d, p, nv

def cycle_from(nv, p):
    v = nv
    for _ in range(N): v = p[v]
    c = []; s = v
    while True:
        c.append(v); v = p[v]
        if v == s: c.append(v); break
    c.reverse()
    return c


def detect(m, fee):
    # Run BF at given fee (net) AND at fee=0 (gross). Returns dict with both.
    null = {"detected": False, "net_pct": 0.0, "cycle": "", "legs": 0,
            "gross_mult": 1.0, "gross_pct": 0.0, "gross_cycle": ""}

    # --- Pass 1: detect at actual fee level (net) ---
    edg = get_edges(m, fee); seen = set(); best_net = None
    for src in range(N):
        _, p, nv = bf(edg, src)
        if nv is None: continue
        cyc = cycle_from(nv, p)
        sig = frozenset(zip(cyc[:-1], cyc[1:]))
        if sig in seen: continue
        seen.add(sig); g = n = 1.0
        for k in range(len(cyc) - 1):
            r = m[cyc[k]][cyc[k + 1]]; g *= r; n *= r * (1 - fee)
        pct = (n - 1) * 100
        if pct > 0 and (best_net is None or pct > best_net["net_pct"]):
            best_net = {"detected": True, "net_pct": round(pct, 6),
                        "cycle": " -> ".join(CURRENCIES[i] for i in cyc),
                        "legs": len(cyc) - 1, "gross_mult": round(g, 8),
                        "gross_pct": round((g - 1) * 100, 6),
                        "gross_cycle": " -> ".join(CURRENCIES[i] for i in cyc)}

    # --- Pass 2: detect at fee=0 to find gross-positive cycles ---------------
    edg0 = get_edges(m, 0.0); seen0 = set(); best_gross = None
    for src in range(N):
        _, p, nv = bf(edg0, src)
        if nv is None: continue
        cyc = cycle_from(nv, p)
        sig = frozenset(zip(cyc[:-1], cyc[1:]))
        if sig in seen0: continue
        seen0.add(sig); g = 1.0
        for k in range(len(cyc) - 1):
            g *= m[cyc[k]][cyc[k + 1]]
        g_pct = (g - 1) * 100
        if g_pct > 0 and (best_gross is None or g_pct > best_gross["gross_pct"]):
            best_gross = {"gross_mult": round(g, 8), "gross_pct": round(g_pct, 6),
                          "gross_cycle": " -> ".join(CURRENCIES[i] for i in cyc),
                          "gross_legs": len(cyc) - 1}

    if best_net:
        return best_net
    if best_gross:
        # Gross opportunity exists but fee destroys it
        r = {"detected": False, "net_pct": 0.0, "cycle": "", "legs": 0}
        r.update(best_gross)
        return r
    return null

# ── 4. Run across all trading days ───────────────────────────────────────────
def run(fx):
    print(f"\n[2/4] Running Bellman-Ford across {len(fx)} trading days...")
    recs = []
    for i, (dt, row) in enumerate(fx.iterrows()):
        m = build_matrix(row); ib = detect(m, FEE_IB); rt = detect(m, FEE_RT)
        recs.append({"date": dt.date(),
            "usd_eur": round(row["EUR"], 6), "usd_gbp": round(row["GBP"], 6),
            "usd_jpy": round(row["JPY"], 6), "usd_cad": round(row["CAD"], 6),
            # Interbank results
            "ib_detected":  ib.get("detected",  False),
            "ib_net_pct":   ib.get("net_pct",   0.0),
            "ib_cycle":     ib.get("cycle",      ""),
            "ib_legs":      ib.get("legs",       0),
            "ib_gross_mult": ib.get("gross_mult", 1.0),
            "ib_gross_pct": ib.get("gross_pct",  0.0),
            "ib_gross_cycle": ib.get("gross_cycle", ""),
            # Retail results
            "rt_detected":  rt.get("detected",  False),
            "rt_net_pct":   rt.get("net_pct",   0.0),
            "rt_cycle":     rt.get("cycle",      ""),
            "rt_legs":      rt.get("legs",       0),
            "rt_gross_mult": rt.get("gross_mult", 1.0),
            "rt_gross_pct": rt.get("gross_pct",  0.0),
            "rt_gross_cycle": rt.get("gross_cycle", ""),
        })
        if (i + 1) % 100 == 0 or i == len(fx) - 1:
            nc = sum(r["ib_detected"] for r in recs)
            rc = sum(r["rt_detected"] for r in recs)
            gc = sum(r["ib_gross_pct"] > 0 for r in recs)
            print(f"   Day {i+1:>4}/{len(fx)} | Gross arbs: {gc:>4} | IB net: {nc:>4} | RT net: {rc:>4}")
    df = pd.DataFrame(recs); df["date"] = pd.to_datetime(df["date"])
    lr = np.log((1 / df["usd_eur"]) / (1 / df["usd_eur"]).shift(1))
    df["eur_usd_vol_5d"] = lr.rolling(5).std() * math.sqrt(252) * 100
    # Fee drag: how much fees consumed of the gross opportunity
    df["ib_fee_drag_pct"] = df["ib_gross_pct"] - df["ib_net_pct"]
    df.to_csv(BASE / "arbitrage_results.csv", index=False)
    print(f"\n   Saved arbitrage_results.csv  ({len(df)} rows x {len(df.columns)} cols)")
    print("\n" + "=" * 65 + "\n  ANALYSIS SUMMARY  (2023-2024)\n" + "=" * 65)
    gross_days = int((df["ib_gross_pct"] > 0).sum()); tot = len(df)
    print(f"  Gross-positive cycles (before any fee) : {gross_days}/{tot} ({gross_days/tot*100:.1f}%)")
    if gross_days > 0:
        sg = df[df["ib_gross_pct"] > 0]
        print(f"    Avg gross profit  : {sg['ib_gross_pct'].mean():.5f}%")
        print(f"    Max gross profit  : {sg['ib_gross_pct'].max():.5f}%")
        print(f"    Most common cycle : {sg['ib_gross_cycle'].mode()[0]}")
    for lbl, dc, pc, cc in [("Interbank 0.02%", "ib_detected", "ib_net_pct", "ib_cycle"),
                              ("Retail    0.10%", "rt_detected", "rt_net_pct", "rt_cycle")]:
        n = int(df[dc].sum())
        print(f"  Net arb days [{lbl}] : {n}/{tot} ({n/tot*100:.1f}%)", end="")
        if n > 0:
            s = df[df[dc]]
            print(f"  avg={s[pc].mean():.5f}%  max={s[pc].max():.5f}%  top: {s[cc].mode()[0]}")
        else:
            print("  (no net-positive arb — fees exceed gross opportunity)")
    return df

# ── 5. Plots ─────────────────────────────────────────────────────────────────
def sav(name):
    plt.savefig(PLOTS / name, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(); print(f"   Saved plots/{name}")

def p1(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Best Detected Cycle Net Profit Over Time  (2023-2024 G10 FX)",
                 fontsize=14, fontweight="bold")
    for ax, col, title, c in [
        (axes[0], "ib_net_pct", "Interbank fee: 0.02% (institutional / prime brokerage)", BLUE),
        (axes[1], "rt_net_pct", "Retail fee: 0.10% (online FX broker)", GREEN)
    ]:
        det = df[df[col] > 0]; ndet = df[df[col] == 0]
        ax.axhline(0, color=GRID, lw=0.8, ls="--")
        ax.scatter(ndet["date"], ndet[col], color=GRID, alpha=0.3, s=4, label="No arb detected")
        ax.scatter(det["date"], det[col], color=c, alpha=0.75, s=20, label="Arb detected", zorder=3)
        roll = df.set_index("date")[col].rolling("30D").max()
        ax.plot(roll.index, roll.values, color=c, alpha=0.5, lw=1.5, label="30-day rolling max")
        ax.set_ylabel("Net Profit (%)")
        ax.set_title(title, fontsize=11, loc="left", pad=4)
        ax.legend(fontsize=8, loc="upper right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}%"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); sav("01_net_profit_over_time.png")

def p2(df):
    ec = defaultdict(int)
    for cs in df[df["ib_detected"]]["ib_cycle"]:
        nodes = [c.strip() for c in cs.replace("->", "|").split("|")]
        for k in range(len(nodes) - 1):
            a, b = nodes[k].strip(), nodes[k + 1].strip()
            if a in CURRENCIES and b in CURRENCIES:
                ec[(a, b)] += 1
    mat = np.zeros((N, N), dtype=int)
    for (a, b), cnt in ec.items():
        mat[CURRENCIES.index(a)][CURRENCIES.index(b)] = cnt
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list("h", [DARK, "#1f4e79", BLUE, GREEN], N=256)
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Days edge appeared in profitable cycle")
    ax.set_xticks(range(N)); ax.set_xticklabels(CURRENCIES, fontsize=11)
    ax.set_yticks(range(N)); ax.set_yticklabels(CURRENCIES, fontsize=11)
    ax.set_xlabel("To currency"); ax.set_ylabel("From currency")
    ax.set_title("Trade Edge Frequency in Detected Cycles (Interbank 0.02%)",
                 fontsize=12, fontweight="bold")
    for i in range(N):
        for j in range(N):
            if mat[i][j] > 0:
                ax.text(j, i, str(mat[i][j]), ha="center", va="center",
                        color="white", fontsize=12, fontweight="bold")
    plt.tight_layout(); sav("02_cycle_frequency_heatmap.png")

def p3(df):
    df2 = df.copy(); df2["month"] = df2["date"].dt.to_period("M")
    mon = df2.groupby("month").agg(ib=("ib_detected", "sum"),
                                    rt=("rt_detected", "sum"),
                                    d=("date", "count")).reset_index()
    mon["mdt"] = mon["month"].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5)); w = 10
    ax.bar(mon["mdt"] - pd.Timedelta(days=w // 2 + 1), mon["ib"],
           width=w, color=BLUE, alpha=0.85, label="Interbank 0.02%")
    ax.bar(mon["mdt"] + pd.Timedelta(days=w // 2 + 1), mon["rt"],
           width=w, color=GREEN, alpha=0.85, label="Retail 0.10%")
    ax.set_xlabel("Month"); ax.set_ylabel("Trading days with detected arbitrage")
    ax.set_title("Monthly Arbitrage Detection Frequency by Fee Regime (2023-2024)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, alpha=0.3, axis="y"); plt.tight_layout(); sav("03_fee_sensitivity.png")

def p4(df):
    sub = df.dropna(subset=["eur_usd_vol_5d"])
    det = sub[sub["ib_detected"]]; ndet = sub[~sub["ib_detected"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(ndet["eur_usd_vol_5d"], ndet["ib_net_pct"],
               color=GRID, alpha=0.3, s=6, label="No arb")
    ax.scatter(det["eur_usd_vol_5d"], det["ib_net_pct"],
               color=BLUE, alpha=0.75, s=22, label="Arb detected", zorder=3)
    if len(det) > 3:
        z = np.polyfit(det["eur_usd_vol_5d"], det["ib_net_pct"], 1)
        xs = np.linspace(det["eur_usd_vol_5d"].min(), det["eur_usd_vol_5d"].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color=YELLOW, lw=1.5, ls="--",
                label=f"Linear trend (slope={z[0]:.5f})")
    ax.set_xlabel("EUR/USD 5-day Realised Volatility (annualised %)")
    ax.set_ylabel("Best Net Profit % (IB fee)")
    ax.set_title("Market Volatility Regime vs. Arbitrage Opportunity Size",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); sav("04_volatility_vs_arb_frequency.png")

def p5(df):
    df2 = df.copy()
    df2["ib_cum"] = df2["ib_net_pct"].clip(lower=0).cumsum()
    df2["rt_cum"] = df2["rt_net_pct"].clip(lower=0).cumsum()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(df2["date"], df2["ib_cum"], alpha=0.18, color=BLUE)
    ax.fill_between(df2["date"], df2["rt_cum"], alpha=0.18, color=GREEN)
    ax.plot(df2["date"], df2["ib_cum"], color=BLUE, lw=2,
            label=f"Interbank 0.02%  cumulative: {df2['ib_cum'].iloc[-1]:.3f}%")
    ax.plot(df2["date"], df2["rt_cum"], color=GREEN, lw=2,
            label=f"Retail    0.10%  cumulative: {df2['rt_cum'].iloc[-1]:.3f}%")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Paper P&L (%)")
    ax.set_title("Cumulative Paper P&L - Executing All Detected Cycles (no slippage, 1 unit/day)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, alpha=0.3); plt.tight_layout(); sav("05_cumulative_paper_pnl.png")

def p6(df):
    ib_p = df[df["ib_detected"]]["ib_net_pct"].values
    rt_p = df[df["rt_detected"]]["rt_net_pct"].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribution of Net Profit on Detected Arbitrage Days (Histogram + KDE)",
                 fontsize=13, fontweight="bold")
    for ax, profits, label, color in [
        (axes[0], ib_p, "Interbank 0.02%", BLUE),
        (axes[1], rt_p, "Retail    0.10%", GREEN)
    ]:
        if len(profits) < 5:
            ax.text(0.5, 0.5, f"n={len(profits)} (insufficient)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label); continue
        ax.hist(profits, bins=30, color=color, alpha=0.45, density=True, label="Histogram")
        kde = gaussian_kde(profits, bw_method=0.35)
        xs = np.linspace(profits.min() - 0.001, profits.max() + 0.001, 300)
        ax.plot(xs, kde(xs), color=color, lw=2.2, label="KDE")
        ax.axvline(profits.mean(), color=YELLOW, lw=1.5, ls="--",
                   label=f"Mean: {profits.mean():.5f}%")
        ax.axvline(np.median(profits), color=RED, lw=1.5, ls=":",
                   label=f"Median: {np.median(profits):.5f}%")
        ax.set_xlabel("Net Profit (%)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}  (n={len(profits)} days)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); sav("06_profit_distribution.png")

def p7(df):
    # Fee-drag analysis: gross profit vs fee cost on days with gross cycles
    sub = df[df["ib_gross_pct"] > 0].copy()
    if len(sub) < 3:
        print("   p7: insufficient gross-cycle days, skipping fee-drag chart")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fee Drag Analysis: Gross Opportunity vs Fee Cost (Interbank 0.02%)",
                 fontsize=13, fontweight="bold")
    # Left: time series of gross pct vs fee drag
    ax = axes[0]
    ax.fill_between(sub["date"], sub["ib_gross_pct"], alpha=0.35, color=YELLOW, label="Gross profit (before fees)")
    ax.fill_between(sub["date"], sub["ib_net_pct"].clip(lower=0), alpha=0.5, color=GREEN, label="Net profit (after fees)")
    ax.set_xlabel("Date"); ax.set_ylabel("Profit %")
    ax.set_title("Gross vs Net Profit on Days with Positive Gross Cycles", fontsize=11, loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    # Right: scatter gross vs fee_drag, colored by detected
    ax2 = axes[1]
    ax2.scatter(sub["ib_gross_pct"], sub["ib_fee_drag_pct"],
                c=sub["ib_detected"].map({True: GREEN, False: RED}),
                alpha=0.7, s=25, zorder=3)
    max_val = max(sub["ib_gross_pct"].max(), sub["ib_fee_drag_pct"].max()) * 1.1
    ax2.plot([0, max_val], [0, max_val], color=GRID, lw=1, ls="--", label="Break-even line")
    ax2.set_xlabel("Gross Profit % (before fees)")
    ax2.set_ylabel("Fee Drag % (fees paid)")
    ax2.set_title("Gross Profit vs Fee Drag (green=net+, red=fee-killed)", fontsize=11, loc="left")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); sav("07_fee_drag_analysis.png")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  TRIANGULAR ARBITRAGE DETECTION ENGINE -- DATA PIPELINE")
    print("  Data   : Yahoo Finance (no API key)")
    print("  Period : 2023-01-01 to 2024-12-31  (2 years)")
    print("  Graph  : USD / EUR / GBP / JPY / CAD  (5 nodes, 20 edges/day)")
    print("  Fees   : 0.02% interbank  |  0.10% retail FX broker")
    print("  Method : Bellman-Ford on -log(rate) transformed graph")
    print("=" * 65)
    fx  = fetch()
    res = run(fx)
    print("\n[3/4] Generating 7 analysis charts...")
    p1(res); p2(res); p3(res); p4(res); p5(res); p6(res); p7(res)
    print(f"\n[4/4] Pipeline complete.")
    print(f"  arbitrage_results.csv : {len(res)} rows x {len(res.columns)} columns")
    print(f"  plots/                : 7 PNG files at 150 DPI")

if __name__ == "__main__":
    main()
