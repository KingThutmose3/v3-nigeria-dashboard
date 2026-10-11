# ============================================================
# app.py — V3 Nigeria Price Intelligence Dashboard
# G3 Ukpejine Resources
# Agent-Based Computational Economics
# 6 cities · 384 agents · Jan 2019 – Dec 2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os

# ── Page configuration ───────────────────────────────────────
st.set_page_config(
    page_title="V3 Nigeria Price Intelligence",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ────────────────────────────────────────────────
DATA_PATH = "data/data/"

CITY_COORDS = {
    "Abuja":         (9.0765,  7.3986),
    "Bauchi":        (10.3158, 9.8442),
    "Kano":          (12.0022, 8.5920),
    "Lagos":         (6.5244,  3.3792),
    "Onitsha":       (6.1409,  6.7868),
    "Port_Harcourt": (4.8156,  7.0498),
}

CITY_COLORS = {
    "Abuja":         "#1565C0",
    "Bauchi":        "#F44336",
    "Kano":          "#FF9800",
    "Lagos":         "#4CAF50",
    "Onitsha":       "#9C27B0",
    "Port_Harcourt": "#00BCD4",
}

CITIES = list(CITY_COORDS.keys())

SCENARIO_LABELS = {
    "baseline":    "📗 Baseline",
    "mpr_tighten": "📙 MPR Tighten",
    "fx_shock":    "📕 FX Shock",
}

SCENARIO_DESC = {
    "baseline":    "No shocks — LSTM trend continuation",
    "mpr_tighten": "CBN MPR +300bps · PMS +₦100 · Bus +₦60",
    "fx_shock":    "Imports ×1.35 · PMS +₦100 · Bus +₦60",
}

SCENARIO_COLORS = {
    "baseline":    "#1565C0",
    "mpr_tighten": "#E65100",
    "fx_shock":    "#6A1B9A",
}

TRADER_ACTIONS = {
    0: "sell_local",
    1: "travel",
    2: "hold",
    3: "reduce_price",
    4: "increase_price",
}

BUYER_ACTIONS = {
    0: "buy_local",
    1: "request_adjacent",
    2: "delay_purchase",
}

BANKING_ACTIONS = {
    0: "maintain_rate",
    1: "tighten_credit",
    2: "ease_credit",
}

TRANSPORTER_ACTIONS = {
    0: "operate",
    1: "suspend",
    2: "increase_fare",
    3: "switch_route",
}


# ── Helper functions ─────────────────────────────────────────
def capital_colour(capital, initial=500_000):
    ratio = capital / initial
    if ratio >= 0.95:
        return "#2E7D32"
    elif ratio >= 0.80:
        return "#F9A825"
    elif ratio >= 0.60:
        return "#E65100"
    else:
        return "#B71C1C"


def safe_load(filename, date_col="date"):
    path = DATA_PATH + filename
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        return df
    except Exception:
        return pd.DataFrame()


# ── Data loading ─────────────────────────────────────────────
@st.cache_data
def load_all_data():
    d = {}

    # Phase 1
    d["p1_migrations"]  = safe_load("Mesa_P1_migration_flows.csv")
    d["p1_steps"]       = safe_load("Mesa_P1_step_records.csv")
    d["p1_transactions"]= safe_load("Mesa_P1_transactions.csv")
    d["p1_bankruptcies"]= safe_load("Mesa_P1_bankruptcies.csv")

    # Phase 2 — all three scenarios, all five file types
    for scenario in ["baseline", "mpr_tighten", "fx_shock"]:
        for suffix in ["steps", "migrations",
                       "transactions", "bankruptcies", "prices"]:
            key  = f"p2_{scenario}_{suffix}"
            fname = f"Mesa_P2_{scenario}_{suffix}.csv"
            d[key] = safe_load(fname)

    return d


# ── Load data ────────────────────────────────────────────────
with st.spinner("Loading simulation data…"):
    D = load_all_data()

p1_steps = D["p1_steps"]
p1_mig   = D["p1_migrations"]
p1_tx    = D["p1_transactions"]


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://flagcdn.com/w40/ng.png", width=40)
    st.title("V3 Nigeria\nPrice Intelligence")
    st.caption("G3 Ukpejine Resources")
    st.divider()

    tab_choice = st.radio(
        "Navigation",
        [
            "📍 Phase 1 — Migration Map",
            "📊 Phase 1 — Emergence Analytics",
            "🔮 Phase 2 — Scenario Comparison",
            "⚠️  City Stress Tracker",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "**Dataset**  \n"
        "SBM Jollof Index + NBS  \n"
        "Jan 2019 – Jun 2026  \n"
        "6 cities · 16 indicators"
    )
    st.markdown(
        "**Simulation**  \n"
        "384 Mesa agents  \n"
        "E&K intermediary model  \n"
        "A2C + LSTM + CTDE"
    )
    st.markdown(
        "**Phase 2 scenarios**  \n"
        "Baseline | MPR Tighten | FX Shock  \n"
        "Jul – Dec 2026 forecast"
    )


# ============================================================
# TAB 1 — PHASE 1 MIGRATION MAP
# ============================================================
if tab_choice == "📍 Phase 1 — Migration Map":

    st.title("Phase 1 — Liquidity Migration Map")
    st.markdown(
        "Trader movements across 6 Nigerian cities, "
        "January 2024 – June 2026. "
        "Traders source goods cheaply at one city and sell "
        "at home city at consumer prices — "
        "the Easley-Kleinberg intermediary model."
    )

    if p1_mig.empty:
        st.warning("Migration data not found in data/ folder.")
        st.stop()

    # ── Controls ─────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        months = sorted(
            p1_mig["date"].dt.to_period("M").unique()
        )
        month_labels = [str(m) for m in months]
        selected_month = st.select_slider(
            "Month", options=month_labels,
            value=month_labels[-1]
        )
    with ctrl2:
        show_all = st.checkbox("Show all months", value=False)
    with ctrl3:
        min_mov = st.slider("Min movements", 1, 15, 2)

    # ── Filter ───────────────────────────────────────────────
    if show_all:
        mig_f = p1_mig.copy()
    else:
        mig_f = p1_mig[
            p1_mig["date"].dt.to_period("M").astype(str)
            == selected_month
        ]

    if not mig_f.empty:
        corridors = (
            mig_f
            .groupby(["from_city", "to_city"])
            .agg(
                movements=("agent_id", "count"),
                avg_spread=("spread", "mean")
            )
            .reset_index()
        )
        corridors = corridors[
            corridors["movements"] >= min_mov
        ]
    else:
        corridors = pd.DataFrame()

    # ── Map and metrics side by side ─────────────────────────
    map_col, info_col = st.columns([2, 1])

    with map_col:
        nigeria_map = folium.Map(
            location=[9.0, 8.0],
            zoom_start=6,
            tiles="CartoDB positron",
        )

        # City markers
        for city, (lat, lon) in CITY_COORDS.items():
            outbound = len(
                p1_mig[p1_mig["from_city"] == city]
            )
            inbound  = len(
                p1_mig[p1_mig["to_city"] == city]
            )
            avg_cap  = 500_000
            if (not p1_steps.empty and
                    "capital" in p1_steps.columns):
                sub = p1_steps[
                    (p1_steps["breed"] == "TraderAgent") &
                    (p1_steps["home_city"] == city)
                ]
                if not sub.empty:
                    avg_cap = sub["capital"].mean()

            fill = capital_colour(avg_cap)

            folium.CircleMarker(
                location=[lat, lon],
                radius=12 + outbound / 30,
                color="white",
                weight=2,
                fill=True,
                fill_color=fill,
                fill_opacity=0.88,
                tooltip=folium.Tooltip(
                    f"<b>{city}</b><br>"
                    f"Outbound: {outbound}<br>"
                    f"Inbound:  {inbound}<br>"
                    f"Avg capital: ₦{avg_cap:,.0f}"
                ),
            ).add_to(nigeria_map)

            folium.Marker(
                location=[lat + 0.28, lon],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:10px;'
                        f'font-weight:bold;color:#212121;'
                        f'white-space:nowrap">{city}</div>'
                    ),
                    icon_size=(130, 18),
                    icon_anchor=(0, 0),
                ),
            ).add_to(nigeria_map)

        # Flow lines
        if not corridors.empty:
            max_m = corridors["movements"].max()
            for _, row in corridors.iterrows():
                src, dst = row["from_city"], row["to_city"]
                n    = row["movements"]
                sprd = row.get("avg_spread", 0.2)
                if src not in CITY_COORDS or dst not in CITY_COORDS:
                    continue

                weight = 2 + (n / max(max_m, 1)) * 8
                colour = (
                    "#1B5E20" if sprd > 0.3 else
                    "#F57F17" if sprd > 0.15 else
                    "#B71C1C"
                )

                folium.PolyLine(
                    locations=[
                        list(CITY_COORDS[src]),
                        list(CITY_COORDS[dst]),
                    ],
                    weight=weight,
                    color=colour,
                    opacity=0.75,
                    tooltip=folium.Tooltip(
                        f"{src} → {dst}<br>"
                        f"Movements: {n}<br>"
                        f"Avg spread: {sprd:.3f}"
                    ),
                ).add_to(nigeria_map)

                # Movement count label at midpoint
                mlat = (
                    CITY_COORDS[src][0] +
                    CITY_COORDS[dst][0]
                ) / 2
                mlon = (
                    CITY_COORDS[src][1] +
                    CITY_COORDS[dst][1]
                ) / 2
                folium.Marker(
                    location=[mlat, mlon],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:9px;'
                            f'color:{colour};font-weight:bold">'
                            f'{n}▶</div>'
                        ),
                        icon_size=(36, 14),
                        icon_anchor=(18, 7),
                    ),
                ).add_to(nigeria_map)

        st_folium(nigeria_map, width=680, height=500)

    with info_col:
        st.markdown("#### Summary metrics")
        n_uniq = (
            p1_mig
            .groupby(["from_city", "to_city"])
            .ngroups
            if not p1_mig.empty else 0
        )
        top_dest = (
            p1_mig["to_city"].value_counts().index[0]
            if not p1_mig.empty else "—"
        )
        top_src = (
            p1_mig["from_city"].value_counts().index[0]
            if not p1_mig.empty else "—"
        )

        st.metric("Total movements",  f"{len(p1_mig):,}")
        st.metric("Unique corridors", n_uniq)
        st.metric("Top destination",  top_dest)
        st.metric("Top source",       top_src)
        if (not p1_mig.empty and
                "spread" in p1_mig.columns):
            st.metric(
                "Avg spread signal",
                f"{p1_mig['spread'].mean():.3f}"
            )

        st.divider()
        st.markdown("#### Colour legend")
        st.markdown(
            "🟢 Capital > ₦475k (growing)  \n"
            "🟡 Capital ₦400–475k (mild stress)  \n"
            "🟠 Capital ₦300–400k (moderate)  \n"
            "🔴 Capital < ₦300k (severe)"
        )
        st.markdown(
            "**Line colour:**  \n"
            "🟢 Strong spread (> 0.3)  \n"
            "🟡 Moderate spread  \n"
            "🔴 Weak spread (< 0.15)"
        )

    # ── Corridor bar chart ────────────────────────────────────
    st.divider()
    st.subheader("Top migration corridors — Phase 1 total")
    if not p1_mig.empty:
        all_corr = (
            p1_mig
            .groupby(["from_city", "to_city"])
            .size()
            .reset_index(name="movements")
            .sort_values("movements", ascending=False)
            .head(12)
        )
        all_corr["corridor"] = (
            all_corr["from_city"] + " → " +
            all_corr["to_city"]
        )
        fig_corr = px.bar(
            all_corr,
            x="movements", y="corridor",
            orientation="h",
            color="movements",
            color_continuous_scale="Viridis",
            title="Trader movements by corridor (all 30 months)",
        )
        fig_corr.update_layout(
            height=420,
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# TAB 2 — PHASE 1 EMERGENCE ANALYTICS
# ============================================================
elif tab_choice == "📊 Phase 1 — Emergence Analytics":

    st.title("Phase 1 — Emergence Analytics")
    st.markdown(
        "What 384 agents collectively discovered from "
        "30 months of real NBS price data without "
        "being told what strategies to use."
    )

    if p1_steps.empty:
        st.warning("Step records not found in data/ folder.")
        st.stop()

    # ── Top metrics ───────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total migrations",   f"{len(p1_mig):,}")
    m2.metric("Total transactions", f"{len(p1_tx):,}")

    away_count = (
        p1_tx["is_away_trade"].sum()
        if (not p1_tx.empty and
            "is_away_trade" in p1_tx.columns) else 0
    )
    m3.metric("Away trades", f"{away_count:,}")

    n_bankrupt = (
        p1_steps[
            p1_steps.get("is_bankrupt", False) == True
        ].shape[0]
        if "is_bankrupt" in p1_steps.columns else 0
    )
    m4.metric("Bankruptcies", n_bankrupt)

    if (not p1_tx.empty and
            "seller_surplus" in p1_tx.columns):
        m5.metric(
            "Avg seller surplus",
            f"₦{p1_tx['seller_surplus'].mean():,.0f}"
        )
        m6.metric(
            "Avg buyer surplus",
            f"₦{p1_tx['buyer_surplus'].mean():,.0f}"
        )

    st.divider()

    # ── Trader and buyer actions ──────────────────────────────
    st.subheader("What strategies did agents choose?")
    st.markdown(
        "Each trader agent chose one of five actions every month "
        "based on what it observed in the market — prices, "
        "network stress and its own capital level. "
        "No agent was told which action to pick. "
        "The chart below shows what each city's trader "
        "population collectively chose most often across "
        "30 months of real NBS data.  \n\n"
        "**What each action means in plain English:**  \n"
        "🔵 **sell_local** — trader buys stock from local "
        "wholesalers and sells to buyers in their own city  \n"
        "🟢 **travel** — trader travels to a cheaper source city, "
        "buys stock there and brings it back home to sell  \n"
        "⚫ **hold** — trader does not buy or sell this month, "
        "waits for a better opportunity  \n"
        "🟠 **reduce_price** — trader lowers their asking price "
        "to sell stock that is not moving  \n"
        "🟣 **increase_price** — trader raises their asking price "
        "when market conditions allow higher margins  \n\n"
        "**What to look for:** Cities where travel dominates "
        "are active arbitrage markets. Cities where reduce_price "
        "dominates are under competitive or inflationary pressure."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Trader actions by city")
        trader_df = p1_steps[
            p1_steps["breed"] == "TraderAgent"
        ].copy()
        if (not trader_df.empty and
                "action" in trader_df.columns):
            trader_df["action_name"] = (
                trader_df["action"]
                .map(TRADER_ACTIONS)
                .fillna("unknown")
            )
            act_dist = (
                trader_df
                .groupby(["home_city", "action_name"])
                .size()
                .reset_index(name="count")
            )
            act_dist["pct"] = (
                act_dist
                .groupby("home_city")["count"]
                .transform(lambda x: x / x.sum() * 100)
            )
            fig_ta = px.bar(
                act_dist,
                x="pct", y="home_city",
                color="action_name",
                orientation="h",
                title="Trader action distribution (%)",
                color_discrete_map={
                    "sell_local":     "#1565C0",
                    "travel":         "#2E7D32",
                    "hold":           "#757575",
                    "reduce_price":   "#E65100",
                    "increase_price": "#6A1B9A",
                },
            )
            fig_ta.update_layout(height=340)
            st.plotly_chart(fig_ta, use_container_width=True)

    with col_b:
        st.subheader("Buyer behaviour by city")
        st.markdown(
            "**What each buyer action means:**  \n"
            "🔵 **buy_local** — buyer purchases from traders "
            "in their own city  \n"
            "🟠 **request_adjacent** — buyer cannot find "
            "enough supply locally and looks to neighbouring "
            "cities — a signal of local supply shortage  \n"
            "🔴 **delay_purchase** — buyer postpones buying "
            "this month, usually because prices are too high  \n\n"
            "**Key signal:** High request_adjacent in Bauchi "
            "confirms it is the most supply-constrained market."
        )
        buyer_df = p1_steps[
            p1_steps["breed"] == "BuyerAgent"
        ].copy()
        if (not buyer_df.empty and
                "action" in buyer_df.columns):
            buyer_df["action_name"] = (
                buyer_df["action"]
                .map(BUYER_ACTIONS)
                .fillna("unknown")
            )
            buy_dist = (
                buyer_df
                .groupby(["home_city", "action_name"])
                .size()
                .reset_index(name="count")
            )
            buy_dist["pct"] = (
                buy_dist
                .groupby("home_city")["count"]
                .transform(lambda x: x / x.sum() * 100)
            )
            fig_bu = px.bar(
                buy_dist,
                x="pct", y="home_city",
                color="action_name",
                orientation="h",
                title="Buyer action distribution (%)",
                color_discrete_map={
                    "buy_local":        "#1565C0",
                    "request_adjacent": "#F57F17",
                    "delay_purchase":   "#B71C1C",
                },
            )
            fig_bu.update_layout(height=340)
            st.plotly_chart(fig_bu, use_container_width=True)

    # ── Trader capital trajectory ─────────────────────────────
    st.divider()
    st.subheader("Trader capital trajectory by city")
    st.markdown(
        "Each of the 120 trader agents started January 2024 "
        "with ₦500,000 in capital — money they use to buy stock, "
        "pay travel costs and cover credit charges.  \n\n"
        "Every month their capital goes up when they successfully "
        "sell goods at a profit, and goes down when they pay "
        "travel costs, buy stock they cannot sell, "
        "or hold inventory while paying credit charges.  \n\n"
        "The chart shows the average capital remaining for "
        "traders in each city across the 30-month simulation.  \n\n"
        "**What to look for:**  \n"
        "🔵 Lines **above** the ₦500k dashed line — that city's "
        "traders grew their capital (Port Harcourt is the only one).  \n"
        "🔴 Lines **below** the ₦500k line — traders are spending "
        "more than they earn. Bauchi falls fastest, "
        "confirming it as the most financially stressed market."
    )
    if (not trader_df.empty and
            "capital" in trader_df.columns):
        cap_tl = (
            trader_df
            .groupby(["date", "home_city"])["capital"]
            .mean()
            .reset_index()
        )
        fig_cap = px.line(
            cap_tl,
            x="date", y="capital",
            color="home_city",
            title=(
                "Mean trader capital per city "
                "(Jan 2024 – Jun 2026)"
            ),
            color_discrete_map=CITY_COLORS,
        )
        fig_cap.add_hline(
            y=500_000,
            line_dash="dash",
            line_color="grey",
            annotation_text="Starting capital ₦500k",
        )
        fig_cap.update_layout(
            height=380,
            yaxis_title="Capital (₦)",
        )
        st.plotly_chart(fig_cap, use_container_width=True)

    # ── Transaction surplus and away/local split ──────────────
    st.divider()
    st.subheader("Who profits from market transactions?")
    st.markdown(
        "Every time a trader sells goods to a buyer, "
        "two things happen: the trader earns a profit "
        "(seller surplus) and the buyer saves money versus "
        "what they would have paid at the full market price "
        "(buyer surplus).  \n\n"
        "This simulation uses the **Easley-Kleinberg "
        "intermediary model** from the book "
        "*Networks, Crowds and Markets* (Chapter 11). "
        "In that model, traders who sit between producers "
        "and consumers — buying cheap and selling at a markup "
        "— capture most of the available profit. "
        "The buyer saves a little but the trader earns a lot "
        "because they take on all the risk of sourcing, "
        "travelling and holding stock.  \n\n"
        "**What the charts show:**  \n"
        "The pie chart on the left shows that traders captured "
        "approximately 87% of total market surplus — "
        "exactly as the E&K model predicts for intermediary "
        "markets with limited buyer-seller connectivity.  \n"
        "The pie chart on the right shows that nearly half "
        "of all transactions involved a trader who had "
        "travelled to a cheaper source city first — "
        "proving that inter-city arbitrage is a major "
        "driver of food market activity in Nigeria."
    )

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Surplus split — E&K model")
        if (not p1_tx.empty and
                "seller_surplus" in p1_tx.columns):
            surp = pd.DataFrame({
                "type": ["Seller surplus", "Buyer surplus"],
                "amount": [
                    p1_tx["seller_surplus"].sum(),
                    p1_tx["buyer_surplus"].sum(),
                ],
            })
            fig_sp = px.pie(
                surp,
                values="amount", names="type",
                title=(
                    "Seller captures ~87% of market surplus  \n"
                    "(E&K intermediary model confirmed)"
                ),
                color_discrete_map={
                    "Seller surplus": "#1565C0",
                    "Buyer surplus":  "#F9A825",
                },
            )
            fig_sp.update_layout(height=320)
            st.plotly_chart(fig_sp, use_container_width=True)

    with col_d:
        st.subheader("Away vs local transactions")
        if (not p1_tx.empty and
                "is_away_trade" in p1_tx.columns):
            tc = p1_tx["is_away_trade"].value_counts()
            trade_df = pd.DataFrame({
                "type":  ["Away trade", "Local trade"],
                "count": [
                    tc.get(True,  0),
                    tc.get(False, 0),
                ],
            })
            fig_aw = px.pie(
                trade_df,
                values="count", names="type",
                title=(
                    "~48% of transactions are away trades  \n"
                    "(trader sourced from another city, "
                    "sold at home)"
                ),
                color_discrete_map={
                    "Away trade":  "#2E7D32",
                    "Local trade": "#1565C0",
                },
            )
            fig_aw.update_layout(height=320)
            st.plotly_chart(fig_aw, use_container_width=True)

    # ── Banking agent actions ─────────────────────────────────
    st.divider()
    st.subheader(
        "Banking agent credit policy — "
        "Lagos ease_credit is the key emergent finding"
    )
    st.markdown(
        "Each city has 3 banking agents whose job is to "
        "set the local credit cost that traders pay every "
        "month. The credit cost is based on the CBN Monetary "
        "Policy Rate (MPR) — the national interest rate set "
        "by the Central Bank of Nigeria.  \n\n"
        "Every month each banking agent chooses one of "
        "three policies:  \n"
        "🔵 **maintain_rate** — keep credit cost at the "
        "current CBN MPR level, neither helping nor hurting traders  \n"
        "🔴 **tighten_credit** — raise credit costs above the "
        "CBN rate, making it more expensive for traders to "
        "hold stock or travel  \n"
        "🟢 **ease_credit** — lower credit costs below the "
        "CBN rate, giving traders cheaper access to working "
        "capital to encourage market activity  \n\n"
        "**The key emergent finding:**  \n"
        "Lagos banking agents independently learned to "
        "**ease_credit** — no other city's banking agents did this. "
        "No rule told the Lagos agents to ease credit. "
        "They discovered it through 30 months of trial and error, "
        "because Lagos has the highest food prices, "
        "the most competitive market and the most capital-stressed "
        "traders — and easing credit kept traders active "
        "and earned the banking agents better rewards.  \n\n"
        "This is emergence — the banking agents in Lagos "
        "independently converged on a policy that mirrors "
        "what a smart central banker would do in a high-cost "
        "urban market."
    )
    banking_df = p1_steps[
        p1_steps["breed"] == "BankingAgent"
    ].copy()
    if (not banking_df.empty and
            "action" in banking_df.columns):
        banking_df["action_name"] = (
            banking_df["action"]
            .map(BANKING_ACTIONS)
            .fillna("unknown")
        )
        bank_dist = (
            banking_df
            .groupby(["home_city", "action_name"])
            .size()
            .reset_index(name="count")
        )
        bank_dist["pct"] = (
            bank_dist
            .groupby("home_city")["count"]
            .transform(lambda x: x / x.sum() * 100)
        )
        fig_bk = px.bar(
            bank_dist,
            x="pct", y="home_city",
            color="action_name",
            orientation="h",
            title=(
                "Banking agent policy by city — "
                "Lagos is the only city with significant ease_credit"
            ),
            color_discrete_map={
                "maintain_rate":  "#1565C0",
                "tighten_credit": "#B71C1C",
                "ease_credit":    "#2E7D32",
            },
        )
        fig_bk.update_layout(height=320)
        st.plotly_chart(fig_bk, use_container_width=True)

    # ── WoLF status ───────────────────────────────────────────
    st.divider()
    st.subheader("Trader WoLF status — winning vs losing")
    st.markdown(
        "**What is WoLF?**  \n"
        "WoLF stands for **Win or Learn Fast** — "
        "a learning mechanism from multi-agent reinforcement "
        "learning (MARL) research. It was designed to help "
        "agents behave differently depending on whether "
        "they are currently doing well or poorly.  \n\n"
        "In this simulation, a trader is **winning** if their "
        "cumulative profit is above the average profit of all "
        "traders in their city. They are **losing** if they "
        "are below that average.  \n\n"
        "**Why it matters:**  \n"
        "WoLF status tells us whether each city's trader "
        "population is converging on a successful strategy "
        "or still struggling. When most traders in a city "
        "are winning, it means the market has found a "
        "stable equilibrium — traders know what to do and "
        "are doing it profitably.  \n\n"
        "**What the chart shows:**  \n"
        "The lines start low in January 2024 because agents "
        "are still learning. All cities converge toward "
        "high winning percentages by mid-2024 as agents "
        "learn profitable strategies from experience. "
        "Port Harcourt reaches near 100% winning earliest — "
        "its oil economy premium makes trading there "
        "consistently profitable. "
        "Bauchi has the most volatile WoLF trajectory — "
        "its fragile market means strategies that work one "
        "month may not work the next."
    )
    if (not trader_df.empty and
            "wolf_status" in trader_df.columns):
        wolf_tl = (
            trader_df
            .groupby(["date", "home_city", "wolf_status"])
            .size()
            .reset_index(name="count")
        )
        wolf_tl["total"] = wolf_tl.groupby(
            ["date", "home_city"]
        )["count"].transform("sum")
        wolf_tl["pct"]   = (
            wolf_tl["count"] / wolf_tl["total"] * 100
        )
        wolf_win = wolf_tl[
            wolf_tl["wolf_status"] == "winning"
        ]
        fig_wf = px.line(
            wolf_win,
            x="date", y="pct",
            color="home_city",
            title=(
                "% Traders in winning WoLF status by city — "
                "lines rising toward 100% = agents learning "
                "profitable strategies"
            ),
            color_discrete_map=CITY_COLORS,
        )
        fig_wf.add_hline(
            y=50, line_dash="dash",
            line_color="grey",
            annotation_text="50% threshold",
        )
        fig_wf.update_layout(
            height=360,
            yaxis_title="% winning",
        )
        st.plotly_chart(fig_wf, use_container_width=True)
        st.caption(
            "All cities cross the 50% threshold by mid-2024 "
            "as agents learn from experience. "
            "Port Harcourt (cyan) is the fastest and most stable. "
            "Bauchi (red) is the most volatile — "
            "consistent with its 100% market stress finding "
            "from the topological analysis."
        )


# ============================================================
# TAB 3 — PHASE 2 SCENARIO COMPARISON
# ============================================================
elif tab_choice == "🔮 Phase 2 — Scenario Comparison":

    st.title("Phase 2 — Scenario Comparison")
    st.markdown(
        "LSTM world model predictions July – December 2026 "
        "under three economic scenarios. "
        "Agents carry Phase 1 learned policies into the forecast."
    )

    # ── Scenario description cards ────────────────────────────
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.info(
            "**📗 Baseline**  \n"
            "No new shocks  \n"
            "LSTM trend continuation  \n"
            "PMS at forecast level"
        )
    with sc2:
        st.warning(
            "**📙 MPR Tighten**  \n"
            "CBN MPR +300bps  \n"
            "PMS +₦100 per litre  \n"
            "Bus +₦60 (60% pass-through)"
        )
    with sc3:
        st.error(
            "**📕 FX Shock**  \n"
            "Imports ×1.35 (35% rise)  \n"
            "PMS +₦100 per litre  \n"
            "Bus +₦60 (60% pass-through)"
        )

    st.divider()

    # ── Prediction vs History summary ─────────────────────────
    st.subheader("📈 LSTM prediction vs Phase 1 history")
    st.markdown(
        "The charts below are **predictions** — not historical "
        "data. The LSTM inside each agent was trained on 90 months "
        "of real NBS prices (Jan 2019 – Jun 2026). "
        "In Phase 2 agents roll their LSTM forward one month at a "
        "time to generate these forecasts. "
        "No NBS data exists after June 2026 — every number "
        "from July 2026 onward is LSTM-generated."
    )

    ph1c, ph2c, ph3c, ph4c = st.columns(4)
    p1_migrations_count = len(p1_mig) if not p1_mig.empty else 0
    p1_surplus_val = (
        p1_tx["seller_surplus"].mean()
        if not p1_tx.empty else 0
    )
    p2_base_mig = D.get("p2_baseline_migrations", pd.DataFrame())
    p2_base_tx  = D.get("p2_baseline_transactions", pd.DataFrame())
    p2_base_surplus = (
        p2_base_tx["seller_surplus"].mean()
        if not p2_base_tx.empty else 0
    )
    ph1c.metric(
        "Phase 1 migrations (30 months — real)",
        f"{p1_migrations_count:,}",
        help="Actual trader movements Jan 2024–Jun 2026"
    )
    ph2c.metric(
        "Baseline predicted (6 months — forecast)",
        f"{len(p2_base_mig):,}",
        delta=f"{len(p2_base_mig) - int(p1_migrations_count/5):+,} vs Phase 1 pace",
        help="LSTM-predicted movements Jul–Dec 2026"
    )
    ph3c.metric(
        "Phase 1 avg seller surplus (real)",
        f"₦{p1_surplus_val:,.0f}",
        help="Actual surplus from real NBS market clearing"
    )
    ph4c.metric(
        "Baseline predicted surplus (forecast)",
        f"₦{p2_base_surplus:,.0f}",
        delta=f"₦{p2_base_surplus - p1_surplus_val:+,.0f} vs Phase 1",
        help="LSTM-predicted surplus under no-shock scenario"
    )

    st.divider()

    # ── Price trajectory selector — per city per scenario ─────
    st.subheader("Forecast price trajectories — per city")
    st.markdown(
        "Select an indicator to see how each of the 6 cities "
        "is predicted to move under each scenario. "
        "Each line is one city. Line style = scenario."
    )

    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        ind_choice = st.selectbox(
            "Select indicator",
            ["Tomatoes_Fresh", "PMS_Price",
             "Bus_Intercity", "Veg_Oil", "Rice"],
            index=0,
            help=(
                "Tomatoes_Fresh: domestic food staple  |  "
                "PMS_Price: petrol (city-level)  |  "
                "Bus_Intercity: intercity fare  |  "
                "Veg_Oil & Rice: import-sensitive commodities"
            )
        )
    with sel_col2:
        scenario_filter = st.selectbox(
            "Select scenario to highlight",
            ["All scenarios", "📗 Baseline",
             "📙 MPR Tighten", "📕 FX Shock"],
            index=0,
        )

    # Map display scenario to key
    scenario_key_map = {
        "All scenarios":  None,
        "📗 Baseline":    "baseline",
        "📙 MPR Tighten": "mpr_tighten",
        "📕 FX Shock":    "fx_shock",
    }
    active_scenario = scenario_key_map[scenario_filter]

    line_styles = {
        "baseline":    dict(dash="solid", width=2),
        "mpr_tighten": dict(dash="dash",  width=2),
        "fx_shock":    dict(dash="dot",   width=2),
    }

    fig_pr = go.Figure()
    scenarios_to_show = (
        ["baseline", "mpr_tighten", "fx_shock"]
        if active_scenario is None
        else [active_scenario]
    )

    for scenario in scenarios_to_show:
        pdf = D.get(f"p2_{scenario}_prices", pd.DataFrame())
        if pdf.empty or ind_choice not in pdf.columns:
            continue
        if "city" not in pdf.columns:
            # Fallback: show average if city column missing
            monthly = (
                pdf.groupby("date")[ind_choice]
                .mean().reset_index()
            )
            fig_pr.add_trace(go.Scatter(
                x=monthly["date"],
                y=monthly[ind_choice],
                mode="lines+markers",
                name=f"{SCENARIO_LABELS[scenario]} (avg)",
                line=dict(
                    color=SCENARIO_COLORS[scenario],
                    **line_styles[scenario],
                ),
            ))
        else:
            # Show one line per city per scenario
            for city in CITIES:
                city_pdf = pdf[pdf["city"] == city]
                if city_pdf.empty:
                    continue
                monthly = (
                    city_pdf.groupby("date")[ind_choice]
                    .mean().reset_index()
                )
                fig_pr.add_trace(go.Scatter(
                    x=monthly["date"],
                    y=monthly[ind_choice],
                    mode="lines+markers",
                    name=f"{city} — {SCENARIO_LABELS[scenario]}",
                    line=dict(
                        color=CITY_COLORS[city],
                        **line_styles[scenario],
                    ),
                    hovertemplate=(
                        f"<b>{city}</b> "
                        f"({SCENARIO_LABELS[scenario]})<br>"
                        f"%{{x|%b %Y}}: ₦%{{y:,.0f}}"
                        f"<extra></extra>"
                    ),
                ))

    fig_pr.update_layout(
        title=(
            f"{ind_choice} forecast Jul–Dec 2026 — "
            f"per city, per scenario  |  "
            f"Line style: solid=Baseline  "
            f"dashed=MPR Tighten  dotted=FX Shock"
        ),
        xaxis_title="Month",
        yaxis_title="Price (₦)",
        height=480,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            font=dict(size=9),
        ),
    )
    st.plotly_chart(fig_pr, use_container_width=True)
    st.caption(
        f"Each coloured line = one city (colour matches city "
        f"colour on the migration map).  "
        f"Line style shows the scenario.  "
        f"**{ind_choice}** prices shown averaged across "
        f"each city's trader agents for that month."
    )

    # ── Fuel section — clearly framed as part of scenarios ───
    st.divider()
    st.subheader("⛽ How fuel costs are built into the scenarios")
    st.markdown(
        "Fuel price is **not a separate scenario** — it is a "
        "component embedded inside both MPR Tighten and FX Shock.  \n\n"
        "The economic logic is:  \n"
        "- Under **MPR Tighten**, CBN raises interest rates "
        "to cool inflation. In Nigeria, monetary tightening "
        "historically coincides with fuel price pressure because "
        "the CBN is responding to imported inflation — "
        "so PMS_Price rises ₦100 and Bus_Intercity rises ₦60.  \n"
        "- Under **FX Shock**, the naira falls 35% against the "
        "dollar. Since petrol is an imported commodity priced in "
        "dollars, naira depreciation immediately raises PMS costs — "
        "so the same ₦100 fuel rise and ₦60 bus fare rise applies.  \n"
        "- Under **Baseline**, no fuel shock — PMS stays at "
        "whatever the LSTM forecast produces from recent trend.  \n\n"
        "**Why this matters for migration:**  \n"
        "Traders decide whether to travel to a cheaper source city "
        "by comparing the price margin they can earn against the "
        "travel cost. When Bus_Intercity rises ₦60, more corridors "
        "become unprofitable and fewer traders travel — "
        "so migration falls under both shock scenarios vs baseline."
    )

    # Three-column comparison of fuel levels and migration effect
    fc1, fc2, fc3 = st.columns(3)
    fuel_summary = {}
    for scenario in ["baseline", "mpr_tighten", "fx_shock"]:
        pdf = D.get(f"p2_{scenario}_prices", pd.DataFrame())
        mdf = D.get(f"p2_{scenario}_migrations", pd.DataFrame())
        tdf = D.get(f"p2_{scenario}_transactions", pd.DataFrame())
        if pdf.empty:
            fuel_summary[scenario] = {}
            continue
        dec = pdf[pdf["date"] == pdf["date"].max()]
        fuel_summary[scenario] = {
            "pms":  dec["PMS_Price"].mean() if "PMS_Price" in dec.columns else 0,
            "bus":  dec["Bus_Intercity"].mean() if "Bus_Intercity" in dec.columns else 0,
            "mig":  len(mdf),
            "tx":   len(tdf),
        }

    base_pms = fuel_summary.get("baseline", {}).get("pms", 0)
    base_bus = fuel_summary.get("baseline", {}).get("bus", 0)
    base_mig = fuel_summary.get("baseline", {}).get("mig", 0)

    for col, scenario, label in zip(
        [fc1, fc2, fc3],
        ["baseline", "mpr_tighten", "fx_shock"],
        ["📗 Baseline\n(no fuel shock)",
         "📙 MPR Tighten\n(CBN tightening + fuel rise)",
         "📕 FX Shock\n(naira fall + fuel rise)"]
    ):
        fs = fuel_summary.get(scenario, {})
        with col:
            st.markdown(f"**{label}**")
            pms_val = fs.get("pms", 0)
            bus_val = fs.get("bus", 0)
            mig_val = fs.get("mig", 0)
            pms_delta = f"+₦{pms_val - base_pms:,.0f}" if scenario != "baseline" else None
            bus_delta = f"+₦{bus_val - base_bus:,.0f}" if scenario != "baseline" else None
            mig_delta = f"{mig_val - base_mig:+,}" if scenario != "baseline" else None
            st.metric("PMS Dec 2026", f"₦{pms_val:,.0f}", delta=pms_delta)
            st.metric("Bus fare Dec 2026", f"₦{bus_val:,.0f}", delta=bus_delta)
            st.metric("Migrations Jul–Dec", mig_val, delta=mig_delta,
                      delta_color="inverse")

    # PMS and Bus trajectory charts side by side
    st.markdown(
        "**PMS_Price and Bus_Intercity trajectories "
        "Jul–Dec 2026 — Baseline vs shock scenarios:**"
    )
    fig_ft = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "PMS_Price (₦ per litre) — all 6 cities avg",
            "Bus_Intercity (₦ per trip) — all 6 cities avg",
        ],
    )
    for scenario in ["baseline", "mpr_tighten", "fx_shock"]:
        pdf = D.get(f"p2_{scenario}_prices", pd.DataFrame())
        if pdf.empty:
            continue
        for ci, ind in enumerate(
            ["PMS_Price", "Bus_Intercity"], start=1
        ):
            if ind not in pdf.columns:
                continue
            monthly = pdf.groupby("date")[ind].mean().reset_index()
            fig_ft.add_trace(
                go.Scatter(
                    x=monthly["date"],
                    y=monthly[ind],
                    mode="lines+markers",
                    name=SCENARIO_LABELS[scenario],
                    line=dict(
                        color=SCENARIO_COLORS[scenario],
                        **line_styles[scenario],
                    ),
                    showlegend=(ci == 1),
                    hovertemplate=(
                        f"{SCENARIO_LABELS[scenario]}<br>"
                        f"%{{x|%b %Y}}: ₦%{{y:,.0f}}<extra></extra>"
                    ),
                ),
                row=1, col=ci,
            )
    fig_ft.update_yaxes(title_text="₦", col=1)
    fig_ft.update_yaxes(title_text="₦", col=2)
    fig_ft.update_layout(
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_ft, use_container_width=True)
    st.caption(
        "Baseline PMS stays at LSTM trend level.  "
        "MPR Tighten and FX Shock both add ₦100 to PMS "
        "and ₦60 to Bus_Intercity on top of the LSTM forecast.  "
        "The gap between the blue (Baseline) and orange/purple "
        "lines is the fuel shock effect on transport costs."
    )

    # ── Scenario migration maps ───────────────────────────────
    st.divider()
    st.subheader("Where do traders move? Migration corridors by scenario")
    st.markdown(
        "These three maps show the **predicted** trader migration "
        "corridors from July to December 2026 under each scenario.  \n"
        "**How to read the maps:**  "
        "Each dot is a city — hover to see its name.  "
        "Each line is a corridor traders used to source goods.  "
        "Thicker line = more trader movements along that corridor.  "
        "The colour matches the scenario."
    )

    map_c1, map_c2, map_c3 = st.columns(3)
    map_cols_map = {
        "baseline":    map_c1,
        "mpr_tighten": map_c2,
        "fx_shock":    map_c3,
    }

    for scenario, mcol in map_cols_map.items():
        with mcol:
            mdf = D.get(
                f"p2_{scenario}_migrations", pd.DataFrame()
            )
            tdf = D.get(
                f"p2_{scenario}_transactions", pd.DataFrame()
            )

            # Top corridor for this scenario
            top_corridor = "—"
            if not mdf.empty:
                tc = (
                    mdf.groupby(["from_city","to_city"])
                    .size().idxmax()
                )
                top_corridor = f"{tc[0]} → {tc[1]}"

            st.markdown(
                f"**{SCENARIO_LABELS[scenario]}**  \n"
                f"*{SCENARIO_DESC[scenario]}*  \n"
                f"Top corridor: **{top_corridor}**"
            )

            s_map = folium.Map(
                location=[9.0, 8.0],
                zoom_start=5,
                tiles="CartoDB positron",
            )

            # City markers with name labels
            for city, (lat, lon) in CITY_COORDS.items():
                inbound = (
                    len(mdf[mdf["to_city"] == city])
                    if not mdf.empty else 0
                )
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=9 + inbound // 5,
                    color="white",
                    weight=2,
                    fill=True,
                    fill_color=CITY_COLORS[city],
                    fill_opacity=0.92,
                    tooltip=folium.Tooltip(
                        f"<b>{city}</b><br>"
                        f"Inbound movements: {inbound}"
                    ),
                ).add_to(s_map)

                # City name label
                folium.Marker(
                    location=[lat + 0.3, lon],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:9px;'
                            f'font-weight:bold;color:#212121;'
                            f'white-space:nowrap">{city}</div>'
                        ),
                        icon_size=(110, 16),
                        icon_anchor=(0, 0),
                    ),
                ).add_to(s_map)

            if not mdf.empty:
                corr = (
                    mdf
                    .groupby(["from_city", "to_city"])
                    .size()
                    .reset_index(name="movements")
                )
                mx = corr["movements"].max() if len(corr) else 1
                for _, row in corr.iterrows():
                    src, dst = row["from_city"], row["to_city"]
                    n = row["movements"]
                    if (src not in CITY_COORDS or
                            dst not in CITY_COORDS):
                        continue
                    folium.PolyLine(
                        locations=[
                            list(CITY_COORDS[src]),
                            list(CITY_COORDS[dst]),
                        ],
                        weight=2 + (n / mx) * 7,
                        color=SCENARIO_COLORS[scenario],
                        opacity=0.80,
                        tooltip=folium.Tooltip(
                            f"<b>{src} → {dst}</b><br>"
                            f"Movements: {n}"
                        ),
                    ).add_to(s_map)

            st_folium(
                s_map, width=290, height=310,
                key=f"map_{scenario}",
            )
            st.caption(
                f"Movements: {len(mdf):,}  |  "
                f"Transactions: {len(tdf):,}"
            )

    # ── Surplus comparison ────────────────────────────────────
    st.divider()
    st.subheader(
        "Transaction surplus comparison — "
        "who earns more under each scenario?"
    )
    st.markdown(
        "**Seller surplus** = profit the trader (intermediary) "
        "earns per transaction after sourcing costs.  \n"
        "**Buyer surplus** = saving the buyer makes versus the "
        "full NBS market reference price.  \n"
        "Higher seller surplus under FX Shock means traders who "
        "can still source domestically capture larger margins "
        "because import prices spiked 35% — widening the spread."
    )
    surp_rows = []
    for scenario in ["baseline", "mpr_tighten", "fx_shock"]:
        tdf = D.get(
            f"p2_{scenario}_transactions", pd.DataFrame()
        )
        if tdf.empty:
            continue
        surp_rows.append({
            "Scenario":           SCENARIO_LABELS[scenario],
            "Transactions":       len(tdf),
            "Avg seller surplus": f"₦{tdf['seller_surplus'].mean():,.0f}",
            "Avg buyer surplus":  f"₦{tdf['buyer_surplus'].mean():,.0f}",
            "Away trades":        (
                f"{tdf['is_away_trade'].mean()*100:.0f}%"
                if "is_away_trade" in tdf.columns else "N/A"
            ),
        })
    if surp_rows:
        st.dataframe(
            pd.DataFrame(surp_rows).set_index("Scenario"),
            use_container_width=True,
        )


# ============================================================
# TAB 4 — CITY STRESS TRACKER
# ============================================================
elif tab_choice == "⚠️  City Stress Tracker":

    st.title("City Stress Tracker")
    st.markdown(
        "Trader capital depletion, market fragility and "
        "transporter fuel responses across Phase 1 and "
        "Phase 2 scenarios."
    )

    # ── Phase 1 capital heatmap ───────────────────────────────
    st.subheader(
        "Trader capital — Phase 1 monthly heatmap"
    )
    if (not p1_steps.empty and
            "capital" in p1_steps.columns):
        t_df = p1_steps[
            p1_steps["breed"] == "TraderAgent"
        ].copy()
        cap_ht = (
            t_df
            .groupby(["date", "home_city"])["capital"]
            .mean()
            .reset_index()
            .pivot(
                index="home_city",
                columns="date",
                values="capital",
            )
        )
        cap_ht.columns = [
            c.strftime("%b %y") for c in cap_ht.columns
        ]
        fig_ht = px.imshow(
            cap_ht,
            color_continuous_scale=[
                [0.0,  "#B71C1C"],
                [0.70, "#F9A825"],
                [0.90, "#2E7D32"],
                [1.0,  "#1B5E20"],
            ],
            zmin=300_000,
            zmax=600_000,
            title=(
                "Mean trader capital per city per month (₦)  "
                "— red = stressed, green = growing"
            ),
            aspect="auto",
        )
        fig_ht.update_layout(
            height=280,
            coloraxis_colorbar=dict(
                title="₦", tickformat=",d"
            ),
        )
        st.plotly_chart(fig_ht, use_container_width=True)

    # ── Phase 2 capital comparison ────────────────────────────
    st.subheader(
        "Capital change from ₦500k — December 2026 by scenario"
    )
    cap_rows = []
    for scenario in ["baseline", "mpr_tighten", "fx_shock"]:
        sdf = D.get(f"p2_{scenario}_steps", pd.DataFrame())
        if sdf.empty:
            continue
        t_s = sdf[sdf["breed"] == "TraderAgent"]
        if t_s.empty:
            continue
        last = t_s["date"].max()
        ld   = t_s[t_s["date"] == last]
        for city in CITIES:
            cd = ld[ld["home_city"] == city]
            if cd.empty:
                continue
            cap_rows.append({
                "Scenario": SCENARIO_LABELS[scenario],
                "City":     city,
                "Capital":  cd["capital"].mean(),
                "Delta":    cd["capital"].mean() - 500_000,
            })

    if cap_rows:
        cap_df  = pd.DataFrame(cap_rows)
        fig_cp2 = px.bar(
            cap_df,
            x="City", y="Delta",
            color="Scenario",
            barmode="group",
            title=(
                "Capital change from ₦500k starting capital "
                "(positive = growth, negative = depletion)"
            ),
            color_discrete_map={
                SCENARIO_LABELS["baseline"]:    "#1565C0",
                SCENARIO_LABELS["mpr_tighten"]: "#E65100",
                SCENARIO_LABELS["fx_shock"]:    "#6A1B9A",
            },
        )
        fig_cp2.add_hline(
            y=0, line_color="black", line_width=1.5
        )
        fig_cp2.update_layout(
            height=400,
            yaxis_title="Capital change from ₦500k (₦)",
        )
        st.plotly_chart(fig_cp2, use_container_width=True)

    # ── Transporter fuel response ─────────────────────────────
    st.divider()
    st.subheader("🚌 Transporter responses to fuel costs")
    st.markdown(
        "Transporter agents choose between operate, suspend, "
        "increase_fare and switch_route each month. "
        "Under fuel shock scenarios PMS rises ₦100 — "
        "transporters must respond."
    )

    # Phase 1 transporter actions
    if not p1_steps.empty:
        tr_df = p1_steps[
            p1_steps["breed"] == "TransporterAgent"
        ].copy()
        if (not tr_df.empty and
                "action" in tr_df.columns):
            tr_df["action_name"] = (
                tr_df["action"]
                .map(TRANSPORTER_ACTIONS)
                .fillna("unknown")
            )
            tr_dist = (
                tr_df
                .groupby(["home_city", "action_name"])
                .size()
                .reset_index(name="count")
            )
            tr_dist["pct"] = (
                tr_dist
                .groupby("home_city")["count"]
                .transform(lambda x: x / x.sum() * 100)
            )
            fig_tr1 = px.bar(
                tr_dist,
                x="pct", y="home_city",
                color="action_name",
                orientation="h",
                title=(
                    "Transporter action distribution Phase 1 "
                    "— Onitsha and Kano predominantly operate, "
                    "others switch routes"
                ),
                color_discrete_map={
                    "operate":       "#2E7D32",
                    "suspend":       "#B71C1C",
                    "increase_fare": "#E65100",
                    "switch_route":  "#1565C0",
                },
            )
            fig_tr1.update_layout(height=300)
            st.plotly_chart(
                fig_tr1, use_container_width=True
            )

    # Phase 2 transporter comparison across scenarios
    st.subheader(
        "Transporter action shifts under fuel shock"
    )
    tr_scen_rows = []
    for scenario in ["baseline", "mpr_tighten", "fx_shock"]:
        sdf = D.get(f"p2_{scenario}_steps", pd.DataFrame())
        if sdf.empty:
            continue
        trs = sdf[sdf["breed"] == "TransporterAgent"].copy()
        if trs.empty:
            continue
        if "action" in trs.columns:
            trs["action_name"] = (
                trs["action"]
                .map(TRANSPORTER_ACTIONS)
                .fillna("unknown")
            )
            ac = (
                trs["action_name"]
                .value_counts(normalize=True) * 100
            )
            tr_scen_rows.append({
                "Scenario":         SCENARIO_LABELS[scenario],
                "Operate %":        f"{ac.get('operate', 0):.0f}%",
                "Increase fare %":  f"{ac.get('increase_fare', 0):.0f}%",
                "Switch route %":   f"{ac.get('switch_route', 0):.0f}%",
                "Suspend %":        f"{ac.get('suspend', 0):.0f}%",
            })

    if tr_scen_rows:
        st.dataframe(
            pd.DataFrame(tr_scen_rows)
            .set_index("Scenario"),
            use_container_width=True,
        )
        st.caption(
            "Under MPR Tighten and FX Shock fuel costs rise "
            "₦100 per litre. Expect increase_fare % to rise "
            "and operate % to fall as routes become less "
            "profitable without fare adjustments."
        )

    # Transporter cumulative reward
    if (not p1_steps.empty and
            "cum_reward" in p1_steps.columns):
        tr_rew = p1_steps[
            p1_steps["breed"] == "TransporterAgent"
        ]
        if not tr_rew.empty:
            last_d  = tr_rew["date"].max()
            rew_df  = (
                tr_rew[tr_rew["date"] == last_d]
                .groupby("home_city")["cum_reward"]
                .mean()
                .reset_index()
                .sort_values("cum_reward")
            )
            fig_trew = px.bar(
                rew_df,
                x="cum_reward",
                y="home_city",
                orientation="h",
                title=(
                    "Transporter cumulative reward by city "
                    "(negative = fuel costs exceed fares)"
                ),
                color="cum_reward",
                color_continuous_scale=[
                    [0.0,  "#B71C1C"],
                    [0.5,  "#F9A825"],
                    [1.0,  "#2E7D32"],
                ],
            )
            fig_trew.add_vline(
                x=0, line_color="black", line_width=1.5
            )
            fig_trew.update_layout(
                height=300,
                xaxis_title="Cumulative reward (₦)",
                showlegend=False,
            )
            st.plotly_chart(
                fig_trew, use_container_width=True
            )

    # ── Fragility table ───────────────────────────────────────
    st.divider()
    st.subheader("City fragility summary — from TDA Stage 3")

    frag = pd.DataFrame({
        "City":             CITIES,
        "TDA stress %":     [66.3, 100.0, 67.5,
                             85.0, 72.1,  33.8],
        "H1 persistence":   [0.122, 0.000, 0.074,
                             0.096, 0.009, 0.274],
        "Network hub":      [
            "Veg_Oil", "Chicken",
            "Tinned_Tomatoes", "Chicken",
            "Salt", "Rice",
        ],
        "Hub betweenness":  [0.043, 0.304, 0.552,
                             0.430, 0.201, 0.086],
        "Interpretation":   [
            "Capital city — moderate stress",
            "100% stressed — most fragile",
            "Price hub — thin margins",
            "High FX exposure — import dependent",
            "Trading hub — dominant sourcing city",
            "Oil economy — most resilient",
        ],
    })

    def _highlight_stress(row):
        v = row["TDA stress %"]
        if v >= 90:
            return ["background-color: #FFCDD2"] * len(row)
        elif v >= 70:
            return ["background-color: #FFF9C4"] * len(row)
        else:
            return ["background-color: #C8E6C9"] * len(row)

    st.dataframe(
        frag.style.apply(_highlight_stress, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Bauchi: 100% stressed months, H1_persistence=0 — "
        "no stable cyclical dynamics, never returned to "
        "pre-2020 price attractor. "
        "Port Harcourt: lowest stress, highest H1 persistence — "
        "oil economy premium provides structural stability."
    )
