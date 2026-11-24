# Results — `synthetic_sample`

Data 2024-01-02 → 2024-06-28; out-of-sample 2024-03-01 → 2024-06-28, 4 walk-forward folds (expanding, test 1 mo), seeds [0, 1]. Values are mean ± std across seeds.

## Primary (toy) trade stream

- trades: 674 total, 472 out-of-sample
- OOS win rate 26.7%; avg R gross -0.166, cost 0.030 R, net -0.197 R
- time-exit rate 11.9%; median hold 119 min

## Prediction quality (pooled OOS)

| metric | meta-model | base-rate predictor |
|---|---|---|
| ROC-AUC | 0.5999 ± 0.0010 | 0.5 (cost-rule score: 0.6052) |
| Brier | 0.1915 ± 0.0004 | 0.1966 ± 0.0000 |
| Log-loss | 0.5698 ± 0.0013 | 0.5824 ± 0.0000 |
| Brier skill score | 0.0256 ± 0.0020 | 0 |

## Trading impact (same trades, Fusion Zero costs)

| policy | trades | win rate | avg R net | total return | CAGR | max DD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|---|---|---|
| Fixed 1% risk | 470 ± 0 | 26.8% ± 0.0% | -0.195 ± 0.000 | -61.4% ± 0.0% | -94.6% ± 0.0% | 67.6% ± 0.0% | -4.68 ± 0.00 | -5.12 ± 0.00 | -1.40 ± 0.00 |
| ATR-scaled risk | 452 ± 0 | 26.1% ± 0.0% | -0.194 ± 0.000 | -67.9% ± 0.0% | -96.9% ± 0.0% | 72.3% ± 0.0% | -5.30 ± 0.00 | -5.70 ± 0.00 | -1.34 ± 0.00 |
| Cost-rule filter (no ML) | 155 ± 0 | 36.1% ± 0.0% | -0.047 ± 0.000 | -6.5% ± 0.0% | -18.7% ± 0.0% | 21.0% ± 0.0% | -0.68 ± 0.00 | -0.98 ± 0.00 | -0.89 ± 0.00 |
| Meta-model filter | 232 ± 47 | 30.9% ± 0.5% | -0.103 ± 0.022 | -22.3% ± 8.1% | -53.1% ± 14.7% | 37.4% ± 6.2% | -1.73 ± 0.56 | -2.12 ± 0.65 | -1.41 ± 0.16 |
| Meta-model ¼-Kelly | 178 ± 4 | 34.5% ± 1.9% | -0.004 ± 0.056 | 7.0% ± 12.6% | 25.7% ± 44.6% | 20.8% ± 5.8% | 0.67 ± 0.90 | 1.12 ± 1.51 | 1.60 ± 2.59 |

## Sharpe differences (moving-block bootstrap of daily returns)

| policy | vs | seed | ΔSharpe | 95% CI | share of resamples ≤ 0 |
|---|---|---|---|---|---|
| ATR-scaled risk | Fixed 1% risk | 0 | -0.63 | [-1.58, +0.43] | 0.89 |
| Cost-rule filter (no ML) | Fixed 1% risk | 0 | +4.16 | [+0.92, +7.29] | 0.01 |
| Meta-model filter | Fixed 1% risk | 0 | +3.28 | [+0.67, +6.32] | 0.01 |
| Meta-model filter | Fixed 1% risk | 1 | +2.52 | [-0.10, +4.98] | 0.04 |
| Meta-model ¼-Kelly | Fixed 1% risk | 0 | +6.15 | [+3.01, +9.48] | 0.00 |
| Meta-model ¼-Kelly | Fixed 1% risk | 1 | +4.87 | [+1.79, +7.98] | 0.00 |
| Meta-model filter | Cost-rule filter (no ML) | 0 | -0.87 | [-4.35, +3.48] | 0.61 |
| Meta-model filter | Cost-rule filter (no ML) | 1 | -1.64 | [-4.77, +1.58] | 0.83 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 0 | +1.99 | [-0.22, +5.27] | 0.04 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 1 | +0.72 | [-2.45, +3.29] | 0.32 |

## Per-seed

| seed | auc | brier_skill | meta_filter.trades | meta_filter.total_return | meta_filter.max_drawdown | meta_filter.sharpe | kelly.trades | kelly.total_return | kelly.max_drawdown | kelly.sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6006 | 0.0241 | 198 | -0.1661 | 0.3303 | -1.3356 | 176 | 0.1588 | 0.1667 | 1.3068 |
| 1 | 0.5991 | 0.0270 | 265 | -0.2801 | 0.4183 | -2.1286 | 181 | -0.0195 | 0.2492 | 0.0346 |

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
