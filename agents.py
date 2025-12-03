import numpy as np
from mesa import Agent

def pareto_int(rng, xm, alpha, cap):
    u = float(rng.uniform())
    x = xm / (u ** (1.0 / alpha))
    return int(min(cap, max(1.0, x)))


class Fundamentalist(Agent):
    def __init__(self, uid, model, strength, noise_scale, max_qty=5):
        super().__init__(model)
        self.strength = float(strength)
        self.noise_scale = float(noise_scale)
        self.max_qty = int(max_qty)

    def step(self):
        S = float(self.model.current_price)
        F = float(self.model.fundamental_price)
        eps = float(self.model.rng.normal(0.0, 1.0))
        signal = self.strength * (F - S) / max(S, 1e-12) + self.noise_scale * eps
        qty = int(min(self.max_qty, max(1, round(abs(signal) * self.max_qty))))
        side = "buy" if signal > 0 else "sell"
        self.model.market.place_market(self.unique_id, side, qty)


class Chartist(Agent):
    def __init__(self, uid, model, strength, vol_sens, mom_window, vol_window, noise_scale, max_qty=5):
        super().__init__(model)
        self.strength = float(strength)
        self.vol_sens = float(vol_sens)
        self.mom_window = int(mom_window)
        self.vol_window = int(vol_window)
        self.noise_scale = float(noise_scale)
        self.max_qty = int(max_qty)
        self.uid = uid

    def step(self):
        mom = float(self.model.market.recent_momentum(self.mom_window))
        v = float(self.model.market.recent_vol(self.vol_window))
        eps = float(self.model.rng.normal(0.0, 1.0))
        signal = self.strength * mom * (1.0 + self.vol_sens * v) + self.noise_scale * eps
        qty = int(min(self.max_qty, max(1, round(abs(signal) * self.max_qty))))
        side = "buy" if signal > 0 else "sell"
        self.model.market.place_market(self.unique_id, side, qty)



class NoiseTrader(Agent):
    def __init__(self, uid, model, prob=0.2, sell_bias=0.5, xm=1.0, alpha=1.6, cap=200, limit_offset_ticks=2, ttl=20):
        super().__init__(model)
        self.prob = float(prob)
        self.sell_bias = float(sell_bias)
        self.xm = float(xm)
        self.alpha = float(alpha)
        self.cap = int(cap)
        self.limit_offset_ticks = int(limit_offset_ticks)
        self.ttl = int(ttl)

    def step(self):
        if float(self.model.rng.uniform()) > self.prob:
            return

        reg = int(self.model.regime)
        sell_bias = min(0.95, self.sell_bias + 0.20 * reg)
        side = "sell" if float(self.model.rng.uniform()) < sell_bias else "buy"

        qty = pareto_int(self.model.rng, self.xm, self.alpha, self.cap)

        mid = float(self.model.current_price)
        tick = float(self.model.tick_size)
        off = float(self.limit_offset_ticks + reg) * tick

        limit_prob = 0.85 if reg == 0 else 0.35
        if float(self.model.rng.uniform()) < limit_prob:
            px = mid - off if side == "buy" else mid + off
            self.model.market.place_limit(self.unique_id, side, px, qty, ttl=self.ttl)
        else:
            self.model.market.place_market(self.unique_id, side, qty)




class MarketMaker(Agent):
    def __init__(self, uid, model, base_spread_ticks=2, size=10, ttl=10, levels=5):
        super().__init__(model)
        self.base_spread_ticks = int(base_spread_ticks)
        self.size = int(size)
        self.ttl = int(ttl)
        self.levels = int(levels)
        self.live = []

    def step(self):
        for oid in self.live:
            self.model.market.cancel(oid)
        self.live.clear()

        reg = int(self.model.regime)
        mid = self.model.market.book.mid_price(self.model.market.mid)

        tick = float(self.model.tick_size)

        spread = self.base_spread_ticks + (3 if reg == 1 else 0)
        levels = max(1, self.levels - (2 if reg == 1 else 0))
        base_size = max(1, self.size // (2 if reg == 1 else 1))

        m = mid / tick
        bid0 = int(m) - spread
        ask0 = int(m) + spread

        for L in range(levels):
            q = max(1, base_size // (L + 1))
            bid = (bid0 - L) * tick
            ask = (ask0 + L) * tick
            oidb = self.model.market.place_limit(self.unique_id, "buy", bid, q, ttl=self.ttl)
            oida = self.model.market.place_limit(self.unique_id, "sell", ask, q, ttl=self.ttl)
            if oidb is not None:
                self.live.append(oidb)
            if oida is not None:
                self.live.append(oida)
