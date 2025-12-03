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

        self.regime = 0
        self.p01 = float(cfg.get("p01", 0.02))
        self.p10 = float(cfg.get("p10", 0.10))
        self.shock_rate = float(cfg.get("shock_rate", 0.01))
        self.shock_impact = float(cfg.get("shock_impact", 8.0))
        self.n_events_calm = int(cfg.get("n_events_calm", 400))
        self.n_events_stress = int(cfg.get("n_events_stress", 1200))

        self.regime_log = []
        self.spread_log = []
        self.depth_bid_log = []
        self.depth_ask_log = []
        self.imbalance_log = []
        self.trade_count_log = []
        self.volume_log = []

        self.meta_left = 0
        self.meta_side = "sell"
        self.meta_intensity = 0



        uid = 0
        for _ in range(int(n_mm)):
            self.agents_list.append(MarketMaker(uid, self, base_spread_ticks=base_spread_ticks, size=mm_size, ttl=5))
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
        shock_trigger = 1.0 if last_r < -3.0 * self.market.realized_sigma(window=50) * self.dt**0.5 else 0.0

        if self.regime == 0:
            p = min(1.0, self.p01 * (1.0 + 5.0 * shock_trigger))
            if float(self.rng.uniform()) < p:
                self.regime = 1
        else:
            p = self.p10
            if float(self.rng.uniform()) < p:
                self.regime = 0


    def step(self):
        self.update_regime()
        self.maybe_shock()

        n_events = self.n_events_stress if self.regime == 1 else self.n_events_calm

        for _ in range(n_events):
            a = self.agents_list[int(self.rng.integers(0, len(self.agents_list)))]
            a.step()

        if self.market.book.t == 200:
            bb = self.market.book.best_bid()
            ba = self.market.book.best_ask()
            sp = self.market.spread()
            db, da = self.market.book.depth_at_best()
            print("t=", self.market.book.t, "bb/ba/sp=", bb, ba, sp, "depth=", db, da, "trades_step=", self.market.book.trades_in_step)


        self.market.end_step()
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

        self.market.book.reset_step_counters()


    def maybe_shock(self):
        if self.meta_left <= 0:
            if float(self.rng.uniform()) >= self.shock_rate:
                return
            self.meta_left = int(self.rng.integers(20, 120)) * (2 if self.regime == 1 else 1)
            self.meta_side = "sell" if float(self.rng.uniform()) < 0.6 else "buy"
            self.meta_intensity = int(self.rng.integers(3, 15)) * (3 if self.regime == 1 else 1)
            self.regime = 1

        for _ in range(self.meta_intensity):
            q = int(pareto_int(self.rng, 5.0, 1.2, 20000))
            self.market.place_market(agent_id=-777, side=self.meta_side, qty=q)

        self.meta_left -= 1




    def run(self):
        for _ in range(self.steps_n):
            self.step()
        return np.array(self.market.prices, dtype=float)
