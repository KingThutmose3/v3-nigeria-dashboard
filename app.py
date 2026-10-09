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
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


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
    st.subheader("Trader capital trajectory by city")
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
                    "Seller captures ~90% of market surplus  \n"
                    "(E&K intermediary prediction confirmed)"
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
                    "45% of transactions are away trades  \n"
                    "(trader sourced at city A, sold at city B)"
                ),
                color_discrete_map={
                    "Away trade":  "#2E7D32",
                    "Local trade": "#1565C0",
                },
            )
            fig_aw.update_layout(height=320)
            st.plotly_chart(fig_aw, use_container_width=True)

    # ── Banking agent actions ─────────────────────────────────
    st.subheader(
        "Banking agent credit policy — Lagos ease_credit "
        "is the key emergent finding"
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
                "Banking agent policy — Lagos independently "
                "learned ease_credit (no other city did)"
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
    st.subheader("Trader WoLF status — winning vs losing")
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
            title="% Traders in winning WoLF status by city",
            color_discrete_map=CITY_COLORS,
        )
        fig_wf.add_hline(
            y=50, line_dash="dash",
            line_color="grey",
            annotation_text="50% threshold",
        )
        fig_wf.update_layout(
            height=340,
            yaxis_title="% winning",
        )
        st.plotly_chart(fig_wf, use_container_width=True)


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

    # ── Price trajectory selector ─────────────────────────────
    st.subheader("Forecast price trajectories")
    ind_choice = st.selectbox(
        "Select indicator",
        ["Tomatoes_Fresh", "PMS_Price",
         "Bus_Intercity", "Veg_Oil", "Rice"],
        index=0,
    )

    fig_pr = go.Figure()
    line_styles = {
        "baseline":    dict(dash="solid", width=3),
        "mpr_tighten": dict(dash="dash",  width=2),
        "fx_shock":    dict(dash="dot",   width=2),
    }

    for scenario in ["baseline", "mpr_tighten", "fx_shock"]:
        pdf = D.get(f"p2_{scenario}_prices", pd.DataFrame())
        if pdf.empty or ind_choice not in pdf.columns:
            continue
        monthly = (
            pdf.groupby("date")[ind_choice]
            .mean()
            .reset_index()
        )
        fig_pr.add_trace(go.Scatter(
            x=monthly["date"],
            y=monthly[ind_choice],
            mode="lines+markers",
            name=SCENARIO_LABELS[scenario],
            line=dict(
                color=SCENARIO_COLORS[scenario],
                **line_styles[scenario],
            ),
            hovertemplate=(
                f"{SCENARIO_LABELS[scenario]}<br>"
                f"%{{x|%b %Y}}: ₦%{{y:,.0f}}"
                f"<extra></extra>"
            ),
        ))

    fig_pr.update_layout(
        title=f"{ind_choice} — Forecast Jul–Dec 2026",
        xaxis_title="Month",
        yaxis_title="Price (₦)",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # ── Fuel and transport section ────────────────────────────
    st.divider()
    st.subheader("⛽ Fuel and transport cost impact")
    st.markdown(
        "PMS_Price +₦100 and Bus_Intercity +₦60 applied in "
        "MPR Tighten and FX Shock on top of LSTM forecast. "
        "Higher bus fares reduce net spread signal, making "
        "inter-city travel less profitable."
    )

    # PMS and Bus side-by-side trajectory
    fig_ft = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "PMS_Price Jul–Dec 2026",
            "Bus_Intercity Jul–Dec 2026",
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
            monthly = (
                pdf.groupby("date")[ind]
                .mean()
                .reset_index()
            )
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

    # Fuel vs migration metrics
    fuel_rows = []
    for scenario in ["baseline", "mpr_tighten", "fx_shock"]:
        pdf  = D.get(f"p2_{scenario}_prices", pd.DataFrame())
        mdf  = D.get(f"p2_{scenario}_migrations", pd.DataFrame())
        tdf  = D.get(f"p2_{scenario}_transactions", pd.DataFrame())
        if pdf.empty:
            continue
        dec_df  = pdf[pdf["date"] == pdf["date"].max()]
        avg_pms = (
            dec_df["PMS_Price"].mean()
            if "PMS_Price" in dec_df.columns else 0
        )
        avg_bus = (
            dec_df["Bus_Intercity"].mean()
            if "Bus_Intercity" in dec_df.columns else 0
        )
        away_pct = (
            tdf["is_away_trade"].mean() * 100
            if (not tdf.empty and
                "is_away_trade" in tdf.columns) else 0
        )
        fuel_rows.append({
            "Scenario":           SCENARIO_LABELS[scenario],
            "PMS Dec 2026 (₦)":   f"₦{avg_pms:,.0f}",
            "Bus Dec 2026 (₦)":   f"₦{avg_bus:,.0f}",
            "Total migrations":   len(mdf),
            "Total transactions": len(tdf),
            "Away trades %":      f"{away_pct:.0f}%",
        })

    if fuel_rows:
        st.dataframe(
            pd.DataFrame(fuel_rows).set_index("Scenario"),
            use_container_width=True,
        )

    # ── Scenario migration maps ───────────────────────────────
    st.divider()
    st.subheader(
        "Migration corridors by scenario — Jul–Dec 2026"
    )

    map_c1, map_c2, map_c3 = st.columns(3)
    map_cols_map = {
        "baseline":    map_c1,
        "mpr_tighten": map_c2,
        "fx_shock":    map_c3,
    }

    for scenario, mcol in map_cols_map.items():
        with mcol:
            st.markdown(
                f"**{SCENARIO_LABELS[scenario]}**  \n"
                f"*{SCENARIO_DESC[scenario]}*"
            )
            mdf = D.get(
                f"p2_{scenario}_migrations", pd.DataFrame()
            )

            s_map = folium.Map(
                location=[9.0, 8.0],
                zoom_start=5,
                tiles="CartoDB positron",
            )

            for city, (lat, lon) in CITY_COORDS.items():
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    color="white",
                    weight=1.5,
                    fill=True,
                    fill_color=CITY_COLORS[city],
                    fill_opacity=0.9,
                    tooltip=city,
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
                        weight=2 + (n / mx) * 6,
                        color=SCENARIO_COLORS[scenario],
                        opacity=0.75,
                        tooltip=f"{src}→{dst}: {n}",
                    ).add_to(s_map)

            st_folium(
                s_map, width=290, height=300,
                key=f"map_{scenario}",
            )
            st.caption(
                f"Movements: {len(mdf):,}  |  "
                f"Transactions: "
                f"{len(D.get(f'p2_{scenario}_transactions', pd.DataFrame())):,}"
            )

    # ── Surplus comparison ────────────────────────────────────
    st.divider()
    st.subheader("Transaction surplus comparison")
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
