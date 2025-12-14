# IFT6162 Final Project — Hybrid Wind + Battery Dispatch (RMC)

This repository contains a reproducible implementation of the **Stochastic Control / Regression Monte Carlo (RMC)** method from:

- Thiha Aung & Mike Ludkovski, *Optimal Dispatch of Hybrid Renewable–Battery Storage Resources: A Stochastic Control Approach* (IEEE CDC 2024)

It includes two main notebooks:

- **IV-A**: a *toy stationary exemple* wind generation case.
- **IV-B**: a *a data-calibrated non-stationary* wind generation case driven by **day-ahead (DA) forecasts**.

---

## Repository Layout

- [IV-A.ipynb](IV-A.ipynb): Section IV-A — stationary toy model + RMC training + evaluation plots.
- [IV-B.ipynb](IV-B.ipynb): Section IV-B — non-stationary model driven by DA forecasts + adaptive designs + evaluation bands.
- Data files (committed):
  - [readdata.py](readdata.py): utilities to read/merge the original `.h5` files into clean CSVs.
  - [wind_forecast_time_value_raw_mean_2018.csv](wind_forecast_time_value_raw_mean_2018.csv) (hourly DA forecast mean)
  - `Site_*.h5` (raw sources)

---

## Data (IV-B)

- **File**: [wind_forecast_time_value_raw_mean_2018.csv](wind_forecast_time_value_raw_mean_2018.csv)
- **Columns**:
  - `forecast_time`: hourly timestamps (UTC)
  - `forecast_value`: forecast mean (MW)

If you need to regenerate CSVs from the original `.h5` sources, see [readdata.py](readdata.py).

---

## Environment / Reproducibility

### Python version

Recommended: **Python 3.10+** (the notebooks use modern type hints).

### Install dependencies

```bash
pip install -r requirements.txt
```

### Reproducibility notes (what to expect)

- The RMC algorithm relies on Monte Carlo simulation and GP regression. Results will vary slightly with:
  - random seeds,
  - `Nloc` (design sites) and `Nrep` (replications per site),
  - Gaussian Process hyperparameter optimization.
- For **quick runs**, reduce `Nloc`/`Nrep` (both notebooks already include this guidance).
- For **paper-like quality**, increase `Nloc`/`Nrep` (expect longer runtime).

---

## How To Run

### IV-A (stationary toy model)

1. Open [IV-A.ipynb](IV-A.ipynb)
2. Run cells top-to-bottom.
3. Expected outputs:
   - A trained time-indexed control policy (GP regressors)
  - Multi-path trajectory plots ($X_k$, $B_k$, $I_k$, $O_k$).
  - An estimated total cost / value estimate printed to stdout (e.g., `Estimated value function V(0, ...)`).
  - A Figure-3-like control map (heatmap) of the learned policy at $t=0$.

Key IV-A assumptions:

- $m_k \equiv m$ constant mean reversion level
- $M_k \equiv M$ constant target
- stationary additive-noise dynamics:

$$
X_{k+1}=X_k+\alpha(m-X_k)\Delta t+\sigma\,Z_k\sqrt{\Delta t}
$$

### IV-B (non-stationary, DA-driven)

1. Open [IV-B.ipynb](IV-B.ipynb)
2. Ensure [wind_forecast_time_value_raw_mean_2018.csv] dataset's path is correct.
3. Choose:
   - `DAY_TO_RUN` (any day available in the CSV)
   - `TARGET_MODE` in `{mean, forecast}`
4. Run cells top-to-bottom.

Expected outputs:

- Plots of $m_k$ and $M_k$ on a 15-minute grid
- Pilot-simulation envelope used to build adaptive designs $\mathcal{D}_k$
- RMC training logs
- Confidence bands comparing $X_k$ and firmed output $O_k$ vs target $M_k$
- A single-trajectory comparison (same $X_k$, compare Greedy baseline vs Opt policy)

IV-B dynamics (nonnegative, state-dependent volatility):

$$
X_{k+1}=\left|X_k+\alpha(m_k-X_k)\Delta t+\sigma\sqrt{X_k}\,Z_k\sqrt{\Delta t}\right|
$$

---

## What To Expect (Does it work?)

In IV-B, after training:

- The firmed output $O_k = X_k - B_k$ should track $M_k$ **tighter** than a simple greedy/myopic baseline.
- The printed “% improvement” (relative to the baseline loss) should typically be **positive** on many days, though it can vary by day and by hyperparameters.

If the improvement is noisy:

- Increase `Nloc_B` / `Nrep_B`,
- Fix random seeds,
- Reduce GP hyperparameter restarts for speed (at the cost of fit quality).

---

## Notes / Limitations

- This is an educational reproduction: runtime/accuracy tradeoffs are exposed in notebook parameters.

