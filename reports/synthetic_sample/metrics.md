# Results — `synthetic_sample`

Data 2024-01-02 → 2024-06-28; out-of-sample 2024-03-01 → 2024-06-28, 4 walk-forward folds (expanding, test 1 mo), seeds [0, 1]. Values are mean ± std across seeds.

## Primary (toy) trade stream

- trades: 674 total, 472 out-of-sample
- OOS win rate 26.7%; avg R gross -0.166, cost 0.030 R, net -0.197 R
- time-exit rate 11.9%; median hold 119 min

## Prediction quality (pooled OOS)

| metric | meta-model | base-rate predictor |
|---|---|---|
| ROC-AUC | 0.5999 ± 0.0010 | 0.5 (cost-rule score: 0.6052; logistic regression: 0.5792) |
| Brier | 0.1915 ± 0.0004 | 0.1966 ± 0.0000 |
| Log-loss | 0.5698 ± 0.0013 | 0.5824 ± 0.0000 |
| Brier skill score | 0.0256 ± 0.0020 | 0 |

## Trading impact (same trades, Fusion Zero costs)

| policy | trades | avg risk | win rate | avg R net | risk-weighted R | total return | CAGR | max DD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed 1% risk | 472 ± 0 | 1.0% ± 0.0% | 26.7% ± 0.0% | -0.197 ± 0.000 | -0.197 ± 0.000 | -62.5% ± 0.0% | -95.1% ± 0.0% | 68.8% ± 0.0% | -4.72 ± 0.00 | -5.16 ± 0.00 | -1.38 ± 0.00 |
| ATR-scaled risk | 472 ± 0 | 0.9% ± 0.0% | 26.7% ± 0.0% | -0.197 ± 0.000 | -0.256 ± 0.000 | -69.2% ± 0.0% | -97.3% ± 0.0% | 73.4% ± 0.0% | -5.43 ± 0.00 | -5.80 ± 0.00 | -1.33 ± 0.00 |
| Cost-rule filter (no ML) | 155 ± 0 | 1.0% ± 0.0% | 36.1% ± 0.0% | -0.047 ± 0.000 | -0.047 ± 0.000 | -7.9% ± 0.0% | -22.4% ± 0.0% | 21.5% ± 0.0% | -0.83 ± 0.00 | -1.19 ± 0.00 | -1.04 ± 0.00 |
| Meta-model filter | 232 ± 47 | 1.0% ± 0.0% | 30.9% ± 0.5% | -0.103 ± 0.022 | -0.103 ± 0.022 | -23.2% ± 7.8% | -54.7% ± 14.0% | 37.7% ± 5.8% | -1.79 ± 0.52 | -2.19 ± 0.61 | -1.44 ± 0.15 |
| Meta-model ¼-Kelly | 184 ± 1 | 1.0% ± 0.0% | 34.4% ± 2.0% | -0.011 ± 0.065 | 0.043 ± 0.065 | 6.0% ± 12.0% | 22.0% ± 41.6% | 21.3% ± 5.6% | 0.60 ± 0.85 | 0.99 ± 1.41 | 1.34 ± 2.30 |

## Sharpe differences (moving-block bootstrap of daily returns)

| policy | vs | seed | ΔSharpe | 95% CI | share of resamples ≤ 0 |
|---|---|---|---|---|---|
| ATR-scaled risk | Fixed 1% risk | 0 | -0.73 | [-1.59, +0.23] | 0.93 |
| Cost-rule filter (no ML) | Fixed 1% risk | 0 | +4.04 | [+0.80, +7.21] | 0.01 |
| Meta-model filter | Fixed 1% risk | 0 | +3.24 | [+0.68, +6.19] | 0.01 |
| Meta-model filter | Fixed 1% risk | 1 | +2.53 | [+0.07, +4.79] | 0.02 |
| Meta-model ¼-Kelly | Fixed 1% risk | 0 | +6.08 | [+3.03, +9.29] | 0.00 |
| Meta-model ¼-Kelly | Fixed 1% risk | 1 | +4.87 | [+1.80, +7.87] | 0.00 |
| Meta-model filter | Cost-rule filter (no ML) | 0 | -0.80 | [-4.26, +3.51] | 0.61 |
| Meta-model filter | Cost-rule filter (no ML) | 1 | -1.52 | [-4.60, +1.65] | 0.82 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 0 | +2.04 | [-0.13, +5.32] | 0.04 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 1 | +0.83 | [-2.35, +3.29] | 0.29 |

## Per-seed

| seed | auc | brier_skill | meta_filter.trades | meta_filter.total_return | meta_filter.max_drawdown | meta_filter.sharpe | kelly.trades | kelly.total_return | kelly.max_drawdown | kelly.sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6006 | 0.0241 | 198 | -0.1762 | 0.3354 | -1.4179 | 184 | 0.1448 | 0.1735 | 1.2032 |
| 1 | 0.5991 | 0.0270 | 265 | -0.2872 | 0.4181 | -2.1590 | 185 | -0.0247 | 0.2523 | -0.0024 |

## Primary stream by year (OOS)

| decision_time | trades | win_rate | avg_r_gross | avg_cost_r | avg_r_net |
|---|---|---|---|---|---|
| 2024 | 472 | 0.2669 | -0.1665 | 0.0300 | -0.1965 |

## Per-fold (seed 0)

| fold | test_start | test_end | n_train | n_test | train_base_rate | test_base_rate | auc | threshold | n_rounds | max_depth | min_data_in_leaf | kept_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2024-03-01 | 2024-04-01 | 196 | 106 | 0.3010 | 0.2547 | 0.5696 | 0.2400 | 1 | 3 | 50 | 1.0000 |
| 1 | 2024-04-01 | 2024-05-01 | 308 | 134 | 0.2825 | 0.2239 | 0.5580 | 0.3000 | 12 | 3 | 20 | 0.2463 |
| 2 | 2024-05-01 | 2024-06-01 | 437 | 113 | 0.2632 | 0.3009 | 0.6374 | 0.3400 | 28 | 3 | 20 | 0.1327 |
| 3 | 2024-06-01 | 2024-07-01 | 548 | 119 | 0.2737 | 0.2941 | 0.6362 | 0.3000 | 21 | 3 | 50 | 0.3697 |

## SHAP mean |value| (seed 0, OOS)

| feature | mean_abs_shap |
|---|---|
| stoch_k | 0.0697 |
| hour_sin | 0.0640 |
| spread_atr | 0.0636 |
| range_atr | 0.0525 |
| stoch_diff | 0.0427 |
| atr_pct | 0.0418 |
| upper_wick_ratio | 0.0299 |
| wick_to_range_ratio | 0.0292 |
| bb_pct_b | 0.0258 |
| minute_sin | 0.0206 |
| clv | 0.0198 |
| adx | 0.0146 |
| stoch_d | 0.0133 |
| stop_atr | 0.0118 |
| minute_cos | 0.0104 |
| lower_wick_ratio | 0.0083 |
| hour_cos | 0.0035 |
| side | 0.0000 |
