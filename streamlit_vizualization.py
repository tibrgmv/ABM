import time
import math
from collections import deque

import pandas as pd
import streamlit as st
import altair as alt


from model import ABMModel


st.set_page_config(page_title="ABM LOB Live", layout="wide", initial_sidebar_state="expanded")


def _agent_palette(agent_type):
    m = {
        "MarketMaker": "#7C3AED",
        "Fundamentalist": "#10B981",
        "Chartist": "#F59E0B",
        "NoiseTrader": "#60A5FA",
        "MetaOrder": "#EF4444",
        "Unknown": "#9CA3AF",
    }
    return m.get(agent_type, "#9CA3AF")

def _finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _sanitize_model_numbers(model):
    if not _finite(getattr(model, "current_price", None)):
        try:
            model.current_price = float(model.market.book.mid_price(model.market.mid))
        except Exception:
            model.current_price = float(getattr(model.market, "mid", 1.0))

    if not _finite(getattr(model, "fundamental_price", None)):
        model.fundamental_price = float(model.current_price)

    if not _finite(getattr(model.market, "mid", None)):
        model.market.mid = float(model.current_price)

    if not _finite(getattr(model, "dt", None)) or float(model.dt) <= 0:
        model.dt = 1.0

    if not _finite(getattr(model, "tick_size", None)) or float(model.tick_size) <= 0:
        model.tick_size = 0.1



def _side_color(side):
    return "#22C55E" if side == "buy" else "#EF4444"


def _fmt_float(x, nd=6):
    try:
        if x is None:
            return "—"
        if not pd.isna(x):
            return f"{float(x):.{nd}f}"
    except Exception:
        pass
    return "—"


def _fmt_int(x):
    try:
        if x is None:
            return "—"
        return str(int(x))
    except Exception:
        return "—"


def _build_agent_labels(model):
    labels = {}
    types = {}
    for a in getattr(model, "agents_list", []):
        uid = getattr(a, "unique_id", None)
        if uid is None:
            continue
        t = a.__class__.__name__
        labels[int(uid)] = f"{t} #{int(uid)}"
        types[int(uid)] = t
    labels[-777] = "MetaOrder (-777)"
    types[-777] = "MetaOrder"
    return labels, types


def _attach_instrumentation(model, state):
    market = model.market
    book = market.book

    if state.get("instrumented", False):
        return

    agent_labels, agent_types = _build_agent_labels(model)
    state["agent_labels"] = agent_labels
    state["agent_types"] = agent_types

    state["events"] = deque(maxlen=int(state.get("events_maxlen", 4000)))
    state["trades"] = deque(maxlen=int(state.get("trades_maxlen", 2000)))

    state["seq"] = 0
    state["last_book_t"] = None
    state["seq_in_step"] = 0

    orig_place_limit = market.place_limit
    orig_place_market = market.place_market
    orig_cancel = market.cancel
    orig_record_trade = book._record_trade

    def _now_stamp():
        bt = int(book.t)
        if state["last_book_t"] is None or state["last_book_t"] != bt:
            state["last_book_t"] = bt
            state["seq_in_step"] = 0
        state["seq_in_step"] += 1
        state["seq"] += 1
        return bt, state["seq_in_step"], state["seq"]

    def place_limit(agent_id, side, price, qty, ttl=1):
        bt, k, seq = _now_stamp()
        oid = orig_place_limit(agent_id, side, price, qty, ttl=ttl)
        state["events"].appendleft(
            {
                "t": bt,
                "k": k,
                "seq": seq,
                "etype": "LIMIT",
                "agent_id": int(agent_id),
                "agent_type": state["agent_types"].get(int(agent_id), "Unknown"),
                "side": "buy" if side == "buy" else "sell",
                "qty": int(qty),
                "price": float(price),
                "oid": oid if oid is not None else "",
                "ttl": int(ttl),
            }
        )
        return oid

    def place_market(agent_id, side, qty):
        bt, k, seq = _now_stamp()
        oid = orig_place_market(agent_id, side, qty)
        state["events"].appendleft(
            {
                "t": bt,
                "k": k,
                "seq": seq,
                "etype": "MARKET",
                "agent_id": int(agent_id),
                "agent_type": state["agent_types"].get(int(agent_id), "Unknown"),
                "side": "buy" if side == "buy" else "sell",
                "qty": int(qty),
                "price": float("nan"),
                "oid": oid if oid is not None else "",
                "ttl": "",
            }
        )
        return oid

    def cancel(order_id):
        bt, k, seq = _now_stamp()
        o = book.orders.get(order_id)
        ok = orig_cancel(order_id)
        if o is not None:
            px = float(o.price_ticks) * float(book.tick_size) if o.price_ticks is not None else float("nan")
            state["events"].appendleft(
                {
                    "t": bt,
                    "k": k,
                    "seq": seq,
                    "etype": "CANCEL",
                    "agent_id": int(o.agent_id),
                    "agent_type": state["agent_types"].get(int(o.agent_id), "Unknown"),
                    "side": "buy" if o.side == "buy" else "sell",
                    "qty": int(o.qty),
                    "price": px,
                    "oid": int(order_id),
                    "ttl": "",
                }
            )
        else:
            state["events"].appendleft(
                {
                    "t": bt,
                    "k": k,
                    "seq": seq,
                    "etype": "CANCEL",
                    "agent_id": "",
                    "agent_type": "Unknown",
                    "side": "",
                    "qty": "",
                    "price": float("nan"),
                    "oid": int(order_id),
                    "ttl": "",
                }
            )
        return ok

    def _record_trade(price_ticks, qty, passive_agent_id, aggressive_agent_id):
        orig_record_trade(price_ticks, qty, passive_agent_id, aggressive_agent_id)
        bt, k, seq = _now_stamp()
        px = float(price_ticks) * float(book.tick_size)
        state["trades"].appendleft(
            {
                "t": bt,
                "k": k,
                "seq": seq,
                "price": px,
                "qty": int(qty),
                "passive_id": int(passive_agent_id),
                "passive_type": state["agent_types"].get(int(passive_agent_id), "Unknown"),
                "aggr_id": int(aggressive_agent_id),
                "aggr_type": state["agent_types"].get(int(aggressive_agent_id), "Unknown"),
            }
        )
        state["events"].appendleft(
            {
                "t": bt,
                "k": k,
                "seq": seq,
                "etype": "TRADE",
                "agent_id": int(aggressive_agent_id),
                "agent_type": state["agent_types"].get(int(aggressive_agent_id), "Unknown"),
                "side": "",
                "qty": int(qty),
                "price": px,
                "oid": "",
                "ttl": "",
            }
        )

    market.place_limit = place_limit
    market.place_market = place_market
    market.cancel = cancel
    book._record_trade = _record_trade

    state["instrumented"] = True


def _lob_frame(model, depth):
    bids, asks = model.market.book.snapshot_l2(depth=int(depth))
    tick = float(model.tick_size)

    rows = []
    for p_ticks, q in bids:
        rows.append({"side": "bid", "price": float(p_ticks) * tick, "qty": float(q), "signed_qty": -float(q)})
    for p_ticks, q in asks:
        rows.append({"side": "ask", "price": float(p_ticks) * tick, "qty": float(q), "signed_qty": float(q)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["price", "side"], ascending=[True, True]).reset_index(drop=True)
    return df


def _mid_frame(state, tail=600):
    df = pd.DataFrame(state.get("mid_series", []))
    if df.empty:
        return df
    return df.tail(int(tail)).reset_index(drop=True)


def _events_frame(state, tail=200):
    df = pd.DataFrame(list(state.get("events", [])))
    if df.empty:
        return df
    return df.head(int(tail)).reset_index(drop=True)


def _trades_frame(state, tail=200):
    df = pd.DataFrame(list(state.get("trades", [])))
    if df.empty:
        return df
    return df.head(int(tail)).reset_index(drop=True)


def _render_header():
    st.markdown(
        """
        <style>
          .app-title { font-size: 28px; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 8px 0; }
          .subtle { color: rgba(120,120,120,0.95); margin-top: -6px; }
          .chip { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; margin-right: 6px; }
          .chip-ok { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.35); }
          .chip-bad { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.35); }
          .chip-warn { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.35); }
          .panel { border: 1px solid rgba(120,120,120,0.25); border-radius: 16px; padding: 14px 14px; background: rgba(255,255,255,0.02); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="app-title">ABM LOB Live — Market Simulation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Live order book, mid-price, regime, order flow and trades — instrumented without changing your core code.</div>', unsafe_allow_html=True)


def _render_metrics(model, state):
    bb = model.market.book.best_bid()
    ba = model.market.book.best_ask()
    sp = model.market.spread()
    db, da = model.market.book.depth_at_best()
    imb = (db - da) / max(1.0, db + da)
    vol = float(getattr(model, "volatility", 0.0))
    reg = int(getattr(model, "regime", 0))
    t = int(model.market.book.t)
    trades_step = int(model.market.book.trades_in_step)
    vol_step = float(model.market.book.volume_in_step)

    c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
    c1.metric("t", _fmt_int(t))
    c2.metric("Mid", _fmt_float(model.market.mid, 6))
    c3.metric("Best Bid / Ask", f"{_fmt_float(bb, 6)} / {_fmt_float(ba, 6)}")
    c4.metric("Spread", _fmt_float(sp, 6))
    c5.metric("Depth (best)", f"{_fmt_float(db, 0)} / {_fmt_float(da, 0)}")
    c6.metric("Imbalance", _fmt_float(imb, 4))

    c7, c8, c9, c10, c11, c12 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
    regime_label = "CALM" if reg == 0 else "STRESS"
    c7.metric("Regime", regime_label)
    c8.metric("Realized σ (window=50)", _fmt_float(vol, 6))
    c9.metric("Trades in step", _fmt_int(trades_step))
    c10.metric("Volume in step", _fmt_float(vol_step, 2))
    c11.metric("Orders live", _fmt_int(len(model.market.book.orders)))
    c12.metric("Events logged", _fmt_int(len(state.get("events", []))))


def _render_lob(model, depth):
    df = _lob_frame(model, depth=depth)
    if df.empty:
        st.info("LOB is empty yet — run a few steps.")
        return

    chart = (
        alt.Chart(df)
        .mark_bar(size=10, cornerRadiusEnd=5)
        .encode(
            x=alt.X("signed_qty:Q", title="Depth (Bid negative ← | → Ask positive)"),
            y=alt.Y("price:Q", title="Price", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "side:N",
                scale=alt.Scale(domain=["bid", "ask"], range=["#22C55E", "#EF4444"]),
                legend=alt.Legend(title="Side", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("side:N", title="Side"),
                alt.Tooltip("price:Q", title="Price", format=".6f"),
                alt.Tooltip("qty:Q", title="Qty", format=".0f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_mid_chart(state, tail=600):
    df = _mid_frame(state, tail=tail)
    if df.empty:
        st.info("No mid series yet — start the sim.")
        return

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("t:Q", title="t"),
            y=alt.Y("mid:Q", title="Mid", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("t:Q", format=".0f"), alt.Tooltip("mid:Q", format=".6f")],
        )
        .properties(height=260)
    )

    reg = (
        alt.Chart(df)
        .mark_area(opacity=0.10)
        .encode(
            x="t:Q",
            y=alt.value(0),
            y2=alt.value(1),
            color=alt.Color(
                "regime:N",
                scale=alt.Scale(domain=["CALM", "STRESS"], range=["#10B981", "#EF4444"]),
                legend=None,
            ),
        )
        .transform_calculate(regime="datum.reg==0 ? 'CALM' : 'STRESS'")
        .transform_aggregate(
            reg="max(reg)",
            mid="mean(mid)",
            groupby=["t"],
        )
    )

    st.altair_chart(line, use_container_width=True)


def _render_flow_table(state, max_rows=180):
    df = _events_frame(state, tail=max_rows)
    if df.empty:
        st.info("No order flow yet.")
        return

    df = df.copy()
    df["agent"] = df["agent_id"].apply(lambda x: state["agent_labels"].get(int(x), str(x)) if str(x) != "" else "—")
    df["who"] = df["agent_type"].astype(str)
    df["px"] = df["price"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    df["qty"] = df["qty"].astype(str)

    def row_html(r):
        et = str(r["etype"])
        who = str(r["who"])
        ag = str(r["agent"])
        side = str(r["side"])
        qty = str(r["qty"])
        px = str(r["px"])
        t = str(r["t"])
        k = str(r["k"])

        bg = "rgba(120,120,120,0.10)"
        if et == "TRADE":
            bg = "rgba(245,158,11,0.12)"
        elif et == "MARKET":
            bg = "rgba(96,165,250,0.12)"
        elif et == "LIMIT":
            bg = "rgba(34,197,94,0.10)"
        elif et == "CANCEL":
            bg = "rgba(239,68,68,0.10)"

        chip_color = _agent_palette(who)
        side_color = _side_color(side) if side in ("buy", "sell") else "#9CA3AF"

        side_chip = ""
        if side in ("buy", "sell"):
            side_chip = f'<span class="chip" style="background: rgba(0,0,0,0.08); color:{side_color}; border: 1px solid rgba(120,120,120,0.25);">{side.upper()}</span>'

        px_txt = f" @ {px}" if px not in ("", "nan") else ""
        return f"""
          <div style="display:flex; gap:10px; align-items:center; padding:8px 10px; border-radius:14px; background:{bg}; border: 1px solid rgba(120,120,120,0.20); margin-bottom:8px;">
            <div style="min-width:84px; font-weight:800;">t={t}.{k}</div>
            <div style="min-width:76px; font-weight:800;">{et}</div>
            <div>
              <span class="chip" style="background: rgba(0,0,0,0.06); color:{chip_color}; border: 1px solid rgba(120,120,120,0.25);">{who}</span>
              {side_chip}
              <span style="font-weight:800;">{ag}</span>
              <span style="opacity:0.85;"> — qty {qty}{px_txt}</span>
            </div>
          </div>
        """

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Order Flow (latest first)**")
    html = "".join(row_html(r) for _, r in df.iterrows())
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_trades_table(state, max_rows=140):
    df = _trades_frame(state, tail=max_rows)
    if df.empty:
        st.info("No trades yet.")
        return

    df = df.copy()
    df["passive"] = df["passive_id"].apply(lambda x: state["agent_labels"].get(int(x), str(x)))
    df["aggr"] = df["aggr_id"].apply(lambda x: state["agent_labels"].get(int(x), str(x)))
    df["px"] = df["price"].map(lambda x: f"{float(x):.6f}")
    df["qty"] = df["qty"].map(lambda x: str(int(x)))

    def row_html(r):
        t = str(r["t"])
        k = str(r["k"])
        px = str(r["px"])
        qty = str(r["qty"])
        pa = str(r["passive"])
        ag = str(r["aggr"])
        pat = str(r["passive_type"])
        agt = str(r["aggr_type"])

        return f"""
          <div style="display:flex; gap:10px; align-items:center; padding:8px 10px; border-radius:14px; background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.25); margin-bottom:8px;">
            <div style="min-width:84px; font-weight:800;">t={t}.{k}</div>
            <div style="min-width:120px; font-weight:900;">{px}</div>
            <div style="min-width:90px; font-weight:800;">qty {qty}</div>
            <div style="opacity:0.9;">
              <span class="chip" style="background: rgba(0,0,0,0.06); color:{_agent_palette(pat)}; border: 1px solid rgba(120,120,120,0.25);">{pat}</span>
              <span style="font-weight:800;">{pa}</span>
              <span style="opacity:0.8;"> (passive)</span>
              <span style="margin: 0 8px;">→</span>
              <span class="chip" style="background: rgba(0,0,0,0.06); color:{_agent_palette(agt)}; border: 1px solid rgba(120,120,120,0.25);">{agt}</span>
              <span style="font-weight:800;">{ag}</span>
              <span style="opacity:0.8;"> (aggr)</span>
            </div>
          </div>
        """

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Trades (latest first)**")
    html = "".join(row_html(r) for _, r in df.iterrows())
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _init_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "model" not in st.session_state:
        st.session_state.model = None
    if "instrumented" not in st.session_state:
        st.session_state.instrumented = False
    if "mid_series" not in st.session_state:
        st.session_state.mid_series = []
    if "events_maxlen" not in st.session_state:
        st.session_state.events_maxlen = 4000
    if "trades_maxlen" not in st.session_state:
        st.session_state.trades_maxlen = 2000


def _make_model_from_sidebar():
    cfg = {
        "seed": int(st.session_state.cfg_seed),
        "S0": float(st.session_state.cfg_S0),
        "dt": float(st.session_state.cfg_dt),
        "steps": int(st.session_state.cfg_steps),
        "n_fund": int(st.session_state.cfg_n_fund),
        "n_chart": int(st.session_state.cfg_n_chart),
        "fundamental_price": float(st.session_state.cfg_fundamental_price),
        "n_mm": int(st.session_state.cfg_n_mm),
        "n_noise": int(st.session_state.cfg_n_noise),
        "tick_size": float(st.session_state.cfg_tick_size),
        "base_spread_ticks": int(st.session_state.cfg_base_spread_ticks),
        "mm_size": int(st.session_state.cfg_mm_size),
        "p01": float(st.session_state.cfg_p01),
        "p10": float(st.session_state.cfg_p10),
        "shock_rate": float(st.session_state.cfg_shock_rate),
        "shock_impact": float(st.session_state.cfg_shock_impact),
        "n_events_calm": int(st.session_state.cfg_n_events_calm),
        "n_events_stress": int(st.session_state.cfg_n_events_stress),
        "hawkes_mu": float(st.session_state.cfg_hawkes_mu),
        "hawkes_alpha": float(st.session_state.cfg_hawkes_alpha),
        "hawkes_beta": float(st.session_state.cfg_hawkes_beta),
        "max_events": int(st.session_state.cfg_max_events),
        "mm_react_prob": float(st.session_state.cfg_mm_react_prob),
        "mm_latency_events": int(st.session_state.cfg_mm_latency_events),
        "debug": False,
        "debug_print_every": 0,
        "debug_snapshot_every": 0,
    }
    m = ABMModel(**cfg)
    return m


def _reset_sim():
    st.session_state.model = _make_model_from_sidebar()
    st.session_state.instrumented = False
    st.session_state.mid_series = []
    st.session_state.running = False


def _step_sim(n_steps):
    model = st.session_state.model
    state = st.session_state
    _attach_instrumentation(model, state)

    for _ in range(int(n_steps)):
        _sanitize_model_numbers(model)
        try:
            model.step()
        except ValueError as e:
            if "NaN" in str(e) or "cannot convert float NaN" in str(e):
                st.session_state.running = False
                st.error(f"Симуляция остановлена: NaN внутри model.step(): {e}")
                return
            raise

        st.session_state.mid_series.append(
            {
                "t": float(model.market.book.t),
                "mid": float(model.market.mid),
                "reg": int(getattr(model, "regime", 0)),
            }
        )


def _sidebar():
    with st.sidebar:
        st.markdown("### Controls")

        c1, c2 = st.columns(2)
        if c1.button("▶ Run / Resume", use_container_width=True):
            if st.session_state.model is None:
                _reset_sim()
            st.session_state.running = True
        if c2.button("⏸ Pause", use_container_width=True):
            st.session_state.running = False

        c3, c4 = st.columns(2)
        if c3.button("⟲ Reset", use_container_width=True):
            _reset_sim()
        if c4.button("Step ×1", use_container_width=True):
            if st.session_state.model is None:
                _reset_sim()
            _step_sim(1)

        st.markdown("---")
        st.markdown("### Speed")
        st.session_state.steps_per_refresh = st.slider("Steps per refresh", 1, 50, 5, 1)
        st.session_state.refresh_ms = st.slider("Refresh (ms)", 0, 1500, 120, 10)
        st.session_state.lob_depth = st.slider("LOB depth (levels)", 5, 60, 20, 1)

        st.markdown("---")
        st.markdown("### Model config")

        st.session_state.cfg_seed = st.number_input("seed", value=1, step=1)
        st.session_state.cfg_S0 = st.number_input("S0", value=100.0, step=1.0, format="%.6f")
        st.session_state.cfg_fundamental_price = st.number_input("fundamental_price", value=100.0, step=1.0, format="%.6f")
        st.session_state.cfg_dt = st.number_input("dt", value=1.0, step=0.1, format="%.6f")
        st.session_state.cfg_steps = st.number_input("steps (for run())", value=2000, step=100)

        st.markdown("**Agents**")
        st.session_state.cfg_n_mm = st.number_input("n_mm", value=3, step=1)
        st.session_state.cfg_mm_size = st.number_input("mm_size", value=15, step=1)
        st.session_state.cfg_base_spread_ticks = st.number_input("base_spread_ticks", value=2, step=1)
        st.session_state.cfg_tick_size = st.number_input("tick_size", value=0.01, step=0.01, format="%.6f")

        st.session_state.cfg_n_fund = st.number_input("n_fund", value=5, step=1)
        st.session_state.cfg_n_chart = st.number_input("n_chart", value=5, step=1)
        st.session_state.cfg_n_noise = st.number_input("n_noise", value=50, step=1)

        st.markdown("**Regime / shocks**")
        st.session_state.cfg_p01 = st.number_input("p01", value=0.02, step=0.01, format="%.6f")
        st.session_state.cfg_p10 = st.number_input("p10", value=0.10, step=0.01, format="%.6f")
        st.session_state.cfg_shock_rate = st.number_input("shock_rate", value=0.01, step=0.005, format="%.6f")
        st.session_state.cfg_shock_impact = st.number_input("shock_impact", value=8.0, step=1.0, format="%.6f")
        st.session_state.cfg_n_events_calm = st.number_input("n_events_calm", value=400, step=50)
        st.session_state.cfg_n_events_stress = st.number_input("n_events_stress", value=1200, step=100)

        st.markdown("**Hawkes**")
        st.session_state.cfg_hawkes_mu = st.number_input("hawkes_mu", value=200.0, step=10.0, format="%.6f")
        st.session_state.cfg_hawkes_alpha = st.number_input("hawkes_alpha", value=0.5, step=0.1, format="%.6f")
        st.session_state.cfg_hawkes_beta = st.number_input("hawkes_beta", value=5.0, step=0.5, format="%.6f")
        st.session_state.cfg_max_events = st.number_input("max_events", value=3000, step=100)

        st.markdown("**MM reaction**")
        st.session_state.cfg_mm_react_prob = st.number_input("mm_react_prob", value=0.6, step=0.05, format="%.6f")
        st.session_state.cfg_mm_latency_events = st.number_input("mm_latency_events", value=2, step=1)

        st.markdown("---")
        st.markdown("### Export")
        if st.session_state.model is not None and st.session_state.instrumented:
            ev_df = pd.DataFrame(list(st.session_state.events))
            tr_df = pd.DataFrame(list(st.session_state.trades))
            mid_df = pd.DataFrame(st.session_state.mid_series)

            st.download_button(
                "Download order flow CSV",
                data=ev_df.to_csv(index=False).encode("utf-8"),
                file_name="order_flow.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download trades CSV",
                data=tr_df.to_csv(index=False).encode("utf-8"),
                file_name="trades.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download mid series CSV",
                data=mid_df.to_csv(index=False).encode("utf-8"),
                file_name="mid_series.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### Tips")
        st.write("• If LOB looks thin: increase n_mm or mm_size, or reduce tick_size.")
        st.write("• If too fast: lower steps per refresh or increase refresh ms.")


def main():
    _init_state()
    _render_header()
    _sidebar()

    if st.session_state.model is None:
        st.session_state.model = _make_model_from_sidebar()

    model = st.session_state.model

    if st.session_state.running:
        _step_sim(st.session_state.steps_per_refresh)

    _attach_instrumentation(model, st.session_state)

    st.markdown("")

    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Mid-price (live)**")
        _render_mid_chart(st.session_state, tail=800)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Order Book (L2 snapshot)**")
        _render_lob(model, depth=st.session_state.lob_depth)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        _render_metrics(model, st.session_state)
        st.markdown("")
        _render_flow_table(st.session_state, max_rows=160)
        st.markdown("")
        _render_trades_table(st.session_state, max_rows=120)

    if st.session_state.running:
        ms = int(st.session_state.refresh_ms)
        if ms > 0:
            time.sleep(ms / 1000.0)
        st.rerun()



if __name__ == "__main__":
    main()
