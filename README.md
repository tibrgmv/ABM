# Time-Fractional Black–Scholes in an Agent-Based LOB Market

This project builds an **agent-based market (ABM)** with a stylized **limit order book (LOB)** that generates realistic price dynamics. The key idea of the project is to simulate trading data similar to real markets without the classical assumtion of log-normality, with heavy tails, volatility clustering, different regimes (calm/stress), self-exiting (Hawkes-based) processes and other featues. 

We introduce 4 types of traders:

1) market makers
2) fundamentalists
3) momentum traders
4) noisy traders

who act accordingly to current situation on the market, current regime (calm/stress), recent volatility and other factors. 


On top of the simulated market, it compares **classical Black–Scholes (BS)** vs **time-fractional Black–Scholes (TFBS)** option pricing.

---

## Key idea

Classical BS is **memoryless** (Markov diffusion). TFBS introduces **temporal nonlocality (“memory”)** controlled by **α ∈ (0, 1]**:
- **α = 1** → recovers BS
- **α < 1** → stronger memory / persistence effects


---

## Project Tree 

````

.
├── agents.py                   # Agent classes & decision rules
├── market.py                   # LOB engine: tick-based prices, FIFO queues per level, matching + cancels/expirations
├── model.py                    # ABMModel orchestration: macro steps + micro-events, regime logic, shocks/meta-orders, logging
├── analysis.py                 # Metrics + hypothesis evaluation
├── visualization.py            # Plot helpers
├── streamlit_vizualization.py  # Streamlit dashboard for interactive exploration
├── main_h1_h4.ipynb            # Runner notebook for Hypotheses H1 & H4 (pricing + mixed dealer effects)
├── main_h2_h3.ipynb            # Runner notebook for Hypotheses H2 & H3 (bias + where TFBS gains are largest)
├── requirements.txt           
└── README.md                

````

---

## Market model (ABM + LOB)

### LOB mechanics
- Discrete **tick grid**, best bid/ask → mid-price and spread
- **Price–time priority** (FIFO within a level)
- **Limit orders** add liquidity, **market orders** consume liquidity
- **Cancellations** + **TTL expiration buckets** (orders can expire after a fixed lifetime)


---

## Hypotheses (H1–H4) and headline findings

- **H1 (TFBS beats BS in a memory+tail market)**  
  TFBS shows **consistently lower pricing error** vs BS for put options; the improvement is stable across replications.

- **H2 (TFBS has smaller pricing bias)**  
  Both models show negative signed bias vs the reference, but TFBS is **closer to zero** and **significantly** closer contract-by-contract.

- **H3 (TFBS gains concentrate in longer maturities / extreme strikes)**  
  TFBS outperforms BS **more often than not** overall; the **maturity effect is supported**.

- **H4 (mixed dealer beliefs reshape IV and liquidity)**  
  The most robust effect is **liquidity**: as TFBS dealer share increases, **tail (OTM put) bid–ask widths widen**. Smile/skew effects exist but are less stable than the liquidity channel.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
````

---

## How to run

### Notebooks

* `main_h1_h4.ipynb` — test of hypothesis 1, 4
* `main_h2_h3.ipynb` — test of hypothesis 2, 3 

### Streamlit

```bash
streamlit run streamlit_vizualization.py
```


