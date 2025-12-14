import numpy as np
from mesa import Model

from agents import pareto_int
from market import LOBMarket
from agents import Fundamentalist, Chartist, MarketMaker, NoiseTrader


class ABMModel(Model):
    def __init__(
        self,
        seed,
        S0,
        dt,
        steps,
        n_fund,
        n_chart,
        fundamental_price,
        impact=0.0,
        omega=0.0,
        alpha_var=0.0,
        beta_var=0.0,
        fund_strength=0.5,
        chart_strength=0.6,
        chart_vol_sens=2.0,
        mom_window=20,
        vol_window=20,
        agent_noise=0.15,
        n_mm=3,
        n_noise=50,
        tick_size=0.01,
        base_spread_ticks=2,
        mm_size=15,
        **cfg
    ):
        super().__init__()
        self.rng = np.random.default_rng(int(seed))
        self.steps_n = int(steps)
        self.dt = float(dt)
        self.n_mm = int(n_mm)

        self.tick_size = float(tick_size)

        self.market = LOBMarket(
            S0=float(S0),
            dt=float(dt),
            seed=int(seed) + 1,
            tick_size=float(tick_size),
            base_spread_ticks=int(base_spread_ticks)
        )

        self.current_price = float(S0)
        self.fundamental_price = float(fundamental_price)
        self.volatility = 0.0

        self.agents_list = []

        # Regime switching parameters ===========================
        self.regime = 0
        self.p01 = float(cfg.get("p01", 0.02))
        self.p10 = float(cfg.get("p10", 0.10))
        self.shock_rate = float(cfg.get("shock_rate", 0.01))
        self.shock_impact = float(cfg.get("shock_impact", 8.0))
        self.n_events_calm = int(cfg.get("n_events_calm", 400))
        self.n_events_stress = int(cfg.get("n_events_stress", 1200))

        self.xi_wr = float(cfg.get("xi_wr", 1.0))
        self.xi_ws = float(cfg.get("xi_ws", 1.0))
        self.xi_wI = float(cfg.get("xi_wI", 1.0))
        self.theta_r = float(cfg.get("theta_r", 3.0))
        self.theta_s = float(cfg.get("theta_s", 3.0 * self.tick_size))
        self.theta_I = float(cfg.get("theta_I", 0.5))

        # MM reaction parameters ================================ 
        self.mm_react_prob = float(cfg.get("mm_react_prob", 0.6))
        self.mm_latency_events = int(cfg.get("mm_latency_events", 2))
        self._mm_pending = 0
        self.mm_react_log = []


        # Metaorder parameters ==================================
        self.meta_left = 0
        self.meta_side = "sell"
        self.meta_intensity = 0

        # Hawkes process parameters ==============================
        self.hawkes_mu = float(cfg.get("hawkes_mu", 200.0))
        self.hawkes_alpha = float(cfg.get("hawkes_alpha", 0.5))
        self.hawkes_beta = float(cfg.get("hawkes_beta", 5.0))
        self.hawkes_H = 0.0
        self.max_events = int(cfg.get("max_events", 3000))

        # debug ==================================================
        self.debug = bool(cfg.get("debug", False))
        self.debug_print_every = int(cfg.get("debug_print_every", 0))
        self.debug_snapshot_every = int(cfg.get("debug_snapshot_every", 0))
        self.debug_l2_depth = int(cfg.get("debug_l2_depth", 10))
        self.n_events_log = []
        self.mid_micro_log = []
        self.Nn_log = []
        self.lambda_log = []
        self.lambda_reg_log = []
        self.hawkes_H_log = []
        self.hawkes_cap_hit_log = []
        self.bb_none_log = []
        self.ba_none_log = []
        self.order_count_log = []
        self.bid_levels_log = []
        self.ask_levels_log = []
        self.crossed_log = []
        self.n_limit_log = []
        self.n_market_log = []
        self.n_cancel_log = []
        self.n_expire_log = []
        self.meta_active_log = []
        self.meta_left_log = []
        self.meta_intensity_log = []
        self.meta_side_log = []
        self.l2_snapshots = []
        self.regime_log = []
        self.spread_log = []
        self.depth_bid_log = []
        self.depth_ask_log = []
        self.imbalance_log = []
        self.trade_count_log = []
        self.volume_log = []
        self.mm_requotes_log = []
        self.mm_requotes_in_step = 0


        # create all agents =====================
        uid = 0
        for _ in range(int(n_mm)):
            self.agents_list.append(MarketMaker(uid, self, base_spread_ticks=base_spread_ticks, size=mm_size, ttl=5, ttl_jitter=5))
            uid += 1

        for _ in range(int(n_fund)):
            self.agents_list.append(Fundamentalist(uid, self, strength=fund_strength, noise_scale=agent_noise))
            uid += 1

        for _ in range(int(n_chart)):
            self.agents_list.append(Chartist(uid, self, strength=chart_strength, vol_sens=chart_vol_sens, mom_window=mom_window, vol_window=vol_window, noise_scale=agent_noise))
            uid += 1

        for _ in range(int(n_noise)):
            self.agents_list.append(NoiseTrader(uid, self, prob=0.5, sell_bias=0.7))
            uid += 1

    def update_regime(self):
        last_r = self.market.log_returns[-1] if len(self.market.log_returns) else 0.0
        bb = self.market.book.best_bid()
        ba = self.market.book.best_ask()

        spread = float(ba - bb) if (bb is not None and ba is not None) else 0.0
        depth_bid, depth_ask = self.market.book.depth_at_best()

        # imbalance 
        imb = (depth_bid - depth_ask) / max(1.0, depth_bid + depth_ask)

        # computing stress indicator E_t = w_r · {rt < −θ_r} + w_s · {s_t > θ_s} + w_I · {|I_t| > θ_I }
        r_flag = 1.0 if last_r < -self.theta_r * self.market.realized_sigma(window=50) * self.dt**0.5 else 0.0
        s_flag = 1.0 if spread > self.theta_s else 0.0
        I_flag = 1.0 if abs(imb) > self.theta_I else 0.0

        shock_trigger = self.xi_wr * r_flag + self.xi_ws * s_flag + self.xi_wI * I_flag

        if self.regime == 0:
            # calm -> stress
            p = min(1.0, self.p01 * (1.0 + shock_trigger))
            if float(self.rng.uniform()) < p:
                self.regime = 1
        else:
            # stress -> calm
            p = self.p10
            if float(self.rng.uniform()) < p:
                self.regime = 0


    def step(self):
        self.update_regime()
        self.maybe_shock()
        self.mm_requotes_in_step = 0

        # determine number of events this step (Hawkes process)
        base_scale = self.n_events_calm if self.regime == 0 else self.n_events_stress
        Lambda = self.hawkes_mu + self.hawkes_alpha * self.hawkes_H
        Lambda_reg = Lambda * (base_scale / max(1.0, self.n_events_calm))

        # poisson distribution to get number of events
        Nn = int(self.rng.poisson(Lambda_reg * self.dt))
        n_events = min(self.max_events, max(1, Nn))

        # debug ================================================
        self.n_events_log.append(int(n_events))
        self.Nn_log.append(int(Nn))
        self.lambda_log.append(float(Lambda))
        self.lambda_reg_log.append(float(Lambda_reg))
        self.hawkes_H_log.append(float(self.hawkes_H))
        self.hawkes_cap_hit_log.append(int(Nn >= self.max_events))
        # ======================================================


        # market makers react at the beginning of the step (to fill the LOB)
        for i in range(self.n_mm):
            self.agents_list[i].step()


        mm_reacts_this_step = 0
        for k in range(n_events):
            a = self.agents_list[int(self.rng.integers(self.n_mm, len(self.agents_list)))]
            a.step()

            trade_now = self.market.book.trade_happened
            if trade_now:
                self.market.mark_to_market()
                self.current_price = float(self.market.mid)

                # --- micro mid log  ---------------------
                bb = self.market.book.best_bid()
                ba = self.market.book.best_ask()
                t_micro = float(self.market.book.t) + (k + 1) / (n_events + 1)

                if bb is not None and ba is not None:
                    mid_q = 0.5 * (float(bb) + float(ba))
                else:
                    mid_q = float(self.market.mid)

                self.mid_micro_log.append((t_micro, mid_q))
                # ------------------------------------------


                # market makers may react to the trade
                if self._mm_pending <= 0 and float(self.rng.uniform()) < self.mm_react_prob:
                    self._mm_pending = self.mm_latency_events
                else:
                    self._mm_pending = min(self._mm_pending, 1)

                self.market.book.trade_happened = False


            # process pending MM reactions
            if self._mm_pending > 0:
                self._mm_pending -= 1
                if self._mm_pending == 0:
                    for i in range(self.n_mm):
                        self.agents_list[i].step()
                    mm_reacts_this_step += 1

        self.mm_react_log.append(int(mm_reacts_this_step))

        # debug ===============================================
        if self.market.book.t == 200:
            bb = self.market.book.best_bid()
            ba = self.market.book.best_ask()
            sp = self.market.spread()
            db, da = self.market.book.depth_at_best()
            print(f't={self.market.book.t}  bb={bb}  ba={ba}  sp={sp}  db={db}  da={da}  trades_step={self.market.book.trades_in_step}')


        self.market.end_step()

        # debug ==============================================
        self.current_price = float(self.market.mid)
        self.volatility = float(self.market.realized_sigma(window=50))

        bb = self.market.book.best_bid()
        ba = self.market.book.best_ask()
        spread = float(ba - bb) if (bb is not None and ba is not None) else float("nan")
        depth_bid, depth_ask = self.market.book.depth_at_best()
        imb = (depth_bid - depth_ask) / max(1.0, depth_bid + depth_ask)
        
        self.regime_log.append(int(self.regime))
        self.spread_log.append(spread)
        self.depth_bid_log.append(float(depth_bid))
        self.depth_ask_log.append(float(depth_ask))
        self.imbalance_log.append(float(imb))
        self.trade_count_log.append(int(self.market.book.trades_in_step))
        self.volume_log.append(float(self.market.book.volume_in_step))

        self.bb_none_log.append(int(bb is None))
        self.ba_none_log.append(int(ba is None))
        self.order_count_log.append(int(len(self.market.book.orders)))
        self.bid_levels_log.append(int(len(self.market.book.bid_prices)))
        self.ask_levels_log.append(int(len(self.market.book.ask_prices)))
        self.crossed_log.append(int(self.market.book.crossed_in_step > 0))

        self.mm_requotes_log.append(int(self.mm_requotes_in_step))

        self.n_limit_log.append(int(self.market.book.n_limit_in_step))
        self.n_market_log.append(int(self.market.book.n_market_in_step))
        self.n_cancel_log.append(int(self.market.book.n_cancel_in_step))
        self.n_expire_log.append(int(self.market.book.n_expire_in_step))

        self.meta_active_log.append(int(self.meta_left > 0))
        self.meta_left_log.append(int(self.meta_left))
        self.meta_intensity_log.append(int(self.meta_intensity))
        self.meta_side_log.append(1 if self.meta_side == "buy" else -1)

        if self.debug_snapshot_every > 0 and (self.market.book.t % self.debug_snapshot_every == 0):
            bids, asks = self.market.book.snapshot_l2(depth=self.debug_l2_depth)
            self.l2_snapshots.append((int(self.market.book.t), float(self.current_price), bids, asks))

        if self.debug_print_every > 0 and (self.market.book.t % self.debug_print_every == 0):
            print(
                "t", self.market.book.t,
                "reg", self.regime,
                "mid", round(self.current_price, 4),
                "spread", round(self.spread_log[-1], 4),
                "bb_none/ba_none", self.bb_none_log[-1], self.ba_none_log[-1],
                "orders", self.order_count_log[-1],
                "events", n_events,
                "limits/mkt/cxl/exp", self.n_limit_log[-1], self.n_market_log[-1], self.n_cancel_log[-1], self.n_expire_log[-1],
                "mm_requote", self.mm_requotes_log[-1],
                "crossed", self.crossed_log[-1]
            )
        # ============================================================

        self.market.book.reset_step_counters()

        # update Hawkes process intensity memory
        self.hawkes_H = np.exp(-self.hawkes_beta * self.dt) * self.hawkes_H + float(n_events)


    def maybe_shock(self):
        if self.meta_left <= 0: # if there is no shock now
            if float(self.rng.uniform()) >= self.shock_rate:
                return
            
            # start new shock (choose random number of steps and intensity)
            self.meta_left = int(self.rng.integers(20, 120)) * (2 if self.regime == 1 else 1)
            self.meta_side = "sell" if float(self.rng.uniform()) < 0.6 else "buy"
            self.meta_intensity = int(self.rng.integers(3, 15)) * (3 if self.regime == 1 else 1)
            self.regime = 1

        # execute shock orders (Pareto size distribution = heavy tails)
        for _ in range(self.meta_intensity):
            q = int(pareto_int(self.rng, 5.0, 1.2, 20000))
            self.market.place_market(agent_id=-777, side=self.meta_side, qty=q)

        self.meta_left -= 1


    def run(self):
        for _ in range(self.steps_n):
            self.step()
        return np.array(self.market.prices, dtype=float)
