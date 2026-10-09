# V3 Nigeria Price Intelligence Dashboard

**G3 Ukpejine Resources**  
Agent-Based Computational Economics for Nigerian food,
fuel and transport markets.

---

## What This Is

A multi-agent reinforcement learning simulation of Nigerian
food and transport price dynamics across 6 cities,
built on real NBS and SBM Jollof Index data.

- **6 cities:** Abuja · Bauchi · Kano · Lagos · Onitsha · Port Harcourt
- **16 indicators:** 12 food commodities + PMS + Bus fares + USD/NGN + CBN MPR
- **384 agents:** Traders · Buyers · Transporters · Banking · Shock
- **Timeline:** January 2019 – December 2026

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| Phase 1 — Migration Map | Animated Nigeria map showing trader liquidity flows Jan 2024 – Jun 2026 |
| Phase 1 — Emergence Analytics | Action distributions, capital trajectories, surplus analysis |
| Phase 2 — Scenario Comparison | LSTM forecast Jul–Dec 2026 under Baseline, MPR Tighten, FX Shock |
| City Stress Tracker | Capital depletion heatmap, fragility rankings, transporter fuel response |

---

## Methodology

1. **Takens Embedding** — phase space reconstruction per city
2. **Granger Causality** — directed causal network (383 within-city pairs)
3. **VAR Regression** — rolling coefficient tensor for CTDE initialisation
4. **Persistent Homology** — TDA stress flags and attractor distance
5. **NetworkX** — 86-node dynamic indicator graph
6. **Mesa A2C MARL** — Easley-Kleinberg intermediary model with LSTM world model

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Data Sources

- SBM Intelligence Jollof Index
- National Bureau of Statistics (NBS) Nigeria
- Central Bank of Nigeria (CBN) MPR
