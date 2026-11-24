# Results — `xauusd_histdata_2019_2026_spread_x2`

Data 2019-01-02 → 2026-11-13; out-of-sample 2021-01-01 → 2026-11-13, 12 walk-forward folds (expanding, test 6 mo), seeds [0, 1, 2, 3, 4]. Values are mean ± std across seeds.

## Primary (toy) trade stream

- trades: 10,713 total, 7,932 out-of-sample
- OOS win rate 28.3%; avg R gross -0.077, cost 0.012 R, net -0.089 R
- time-exit rate 6.5%; median hold 150 min

## Prediction quality (pooled OOS)

| metric | meta-model | base-rate predictor |
|---|---|---|
| ROC-AUC | 0.5371 ± 0.0014 | 0.5 (cost-rule score: 0.5538; logistic regression: 0.5378) |
| Brier | 0.2027 ± 0.0001 | 0.2033 ± 0.0000 |
| Log-loss | 0.5952 ± 0.0002 | 0.5967 ± 0.0000 |
| Brier skill score | 0.0030 ± 0.0003 | 0 |

## Trading impact (same trades, Fusion Zero costs)

| policy | trades | avg risk | win rate | avg R net | risk-weighted R | total return | CAGR | max DD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed 1% risk | 7,932 ± 0 | 1.0% ± 0.0% | 28.3% ± 0.0% | -0.089 ± 0.000 | -0.089 ± 0.000 | -100.0% ± 0.0% | -74.1% ± 0.0% | 100.0% ± 0.0% | -2.01 ± 0.00 | -2.55 ± 0.00 | -0.74 ± 0.00 |
| ATR-scaled risk | 7,932 ± 0 | 1.1% ± 0.0% | 28.3% ± 0.0% | -0.089 ± 0.000 | -0.118 ± 0.000 | -100.0% ± 0.0% | -85.3% ± 0.0% | 100.0% ± 0.0% | -2.52 ± 0.00 | -3.09 ± 0.00 | -0.85 ± 0.00 |
| Cost-rule filter (no ML) | 3,870 ± 0 | 1.0% ± 0.0% | 31.6% ± 0.0% | -0.013 ± 0.000 | -0.013 ± 0.000 | -60.0% ± 0.0% | -14.5% ± 0.0% | 85.0% ± 0.0% | -0.19 ± 0.00 | -0.29 ± 0.00 | -0.17 ± 0.00 |
| Meta-model filter | 4,568 ± 77 | 1.0% ± 0.0% | 29.8% ± 0.3% | -0.053 ± 0.011 | -0.053 ± 0.011 | -94.2% ± 2.7% | -39.3% ± 5.1% | 96.8% ± 1.3% | -0.93 ± 0.18 | -1.29 ± 0.24 | -0.41 ± 0.05 |
| Meta-model ¼-Kelly | 1,855 ± 84 | 0.7% ± 0.0% | 31.3% ± 0.2% | -0.034 ± 0.006 | -0.046 ± 0.019 | -50.6% ± 12.0% | -11.6% ± 3.7% | 69.6% ± 5.2% | -0.43 ± 0.19 | -0.66 ± 0.28 | -0.17 ± 0.04 |

## Sharpe differences (moving-block bootstrap of daily returns)

| policy | vs | seed | ΔSharpe | 95% CI | share of resamples ≤ 0 |
|---|---|---|---|---|---|
| ATR-scaled risk | Fixed 1% risk | 0 | -0.50 | [-0.79, -0.20] | 1.00 |
| Cost-rule filter (no ML) | Fixed 1% risk | 0 | +1.82 | [+1.21, +2.44] | 0.00 |
| Meta-model filter | Fixed 1% risk | 0 | +1.30 | [+0.78, +1.87] | 0.00 |
| Meta-model filter | Fixed 1% risk | 1 | +0.84 | [+0.26, +1.40] | 0.00 |
| Meta-model filter | Fixed 1% risk | 2 | +1.01 | [+0.43, +1.59] | 0.00 |
| Meta-model filter | Fixed 1% risk | 3 | +1.24 | [+0.72, +1.79] | 0.00 |
| Meta-model filter | Fixed 1% risk | 4 | +1.05 | [+0.53, +1.61] | 0.00 |
| Meta-model ¼-Kelly | Fixed 1% risk | 0 | +1.61 | [+0.69, +2.47] | 0.00 |
| Meta-model ¼-Kelly | Fixed 1% risk | 1 | +1.45 | [+0.45, +2.36] | 0.00 |
| Meta-model ¼-Kelly | Fixed 1% risk | 2 | +1.67 | [+0.77, +2.55] | 0.00 |
| Meta-model ¼-Kelly | Fixed 1% risk | 3 | +1.36 | [+0.40, +2.20] | 0.00 |
| Meta-model ¼-Kelly | Fixed 1% risk | 4 | +1.84 | [+0.96, +2.73] | 0.00 |
| Meta-model filter | Cost-rule filter (no ML) | 0 | -0.52 | [-1.10, +0.06] | 0.96 |
| Meta-model filter | Cost-rule filter (no ML) | 1 | -0.98 | [-1.55, -0.37] | 1.00 |
| Meta-model filter | Cost-rule filter (no ML) | 2 | -0.81 | [-1.45, -0.21] | 1.00 |
| Meta-model filter | Cost-rule filter (no ML) | 3 | -0.58 | [-1.11, -0.05] | 0.99 |
| Meta-model filter | Cost-rule filter (no ML) | 4 | -0.77 | [-1.35, -0.14] | 0.99 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 0 | -0.21 | [-0.97, +0.54] | 0.73 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 1 | -0.37 | [-1.25, +0.42] | 0.81 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 2 | -0.15 | [-0.87, +0.56] | 0.64 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 3 | -0.46 | [-1.31, +0.31] | 0.90 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 4 | +0.02 | [-0.68, +0.80] | 0.48 |

## Per-seed

| seed | auc | brier_skill | meta_filter.trades | meta_filter.total_return | meta_filter.max_drawdown | meta_filter.sharpe | kelly.trades | kelly.total_return | kelly.max_drawdown | kelly.sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5377 | 0.0033 | 4592 | -0.9085 | 0.9497 | -0.7168 | 1808 | -0.4817 | 0.6507 | -0.4095 |
| 1 | 0.5356 | 0.0026 | 4531 | -0.9720 | 0.9794 | -1.1709 | 1759 | -0.5698 | 0.7347 | -0.5670 |
| 2 | 0.5361 | 0.0027 | 4632 | -0.9580 | 0.9789 | -1.0002 | 1958 | -0.4882 | 0.6642 | -0.3476 |
| 3 | 0.5370 | 0.0031 | 4451 | -0.9179 | 0.9575 | -0.7701 | 1926 | -0.6565 | 0.7679 | -0.6578 |
| 4 | 0.5392 | 0.0033 | 4633 | -0.9545 | 0.9723 | -0.9679 | 1826 | -0.3341 | 0.6619 | -0.1730 |

## Primary stream by year (OOS)

| decision_time | trades | win_rate | avg_r_gross | avg_cost_r | avg_r_net |
|---|---|---|---|---|---|
| 2021 | 1385 | 0.2606 | -0.1492 | 0.0150 | -0.1642 |
| 2022 | 1339 | 0.2733 | -0.1051 | 0.0145 | -0.1196 |
| 2023 | 1244 | 0.2548 | -0.1944 | 0.0164 | -0.2108 |
| 2024 | 1368 | 0.2851 | -0.0723 | 0.0122 | -0.0845 |
| 2025 | 1385 | 0.3170 | 0.0455 | 0.0070 | 0.0385 |
| 2026 | 1211 | 0.3072 | 0.0087 | 0.0035 | 0.0051 |

## Per-fold (seed 0)

| fold | test_start | test_end | n_train | n_test | train_base_rate | test_base_rate | auc | threshold | n_rounds | max_depth | min_data_in_leaf | kept_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2021-01-01 | 2021-07-01 | 2774 | 708 | 0.2495 | 0.2528 | 0.5282 | 0.2300 | 15 | 3 | 400 | 0.9520 |
| 1 | 2021-07-01 | 2022-01-01 | 3473 | 677 | 0.2511 | 0.2688 | 0.5807 | 0.2400 | 40 | 3 | 100 | 0.6411 |
| 2 | 2022-01-01 | 2022-07-01 | 4162 | 673 | 0.2535 | 0.2704 | 0.5362 | 0.2500 | 24 | 5 | 400 | 0.5275 |
| 3 | 2022-07-01 | 2023-01-01 | 4829 | 666 | 0.2560 | 0.2763 | 0.5581 | 0.2700 | 35 | 3 | 100 | 0.2673 |
| 4 | 2023-01-01 | 2023-07-01 | 5498 | 551 | 0.2585 | 0.2722 | 0.5026 | 0.2600 | 34 | 5 | 100 | 0.4392 |
| 5 | 2023-07-01 | 2024-01-01 | 6042 | 693 | 0.2597 | 0.2410 | 0.5123 | 0.2700 | 36 | 5 | 100 | 0.2424 |
| 6 | 2024-01-01 | 2024-07-01 | 6749 | 672 | 0.2578 | 0.2783 | 0.5403 | 0.2700 | 59 | 5 | 400 | 0.3229 |
| 7 | 2024-07-01 | 2025-01-01 | 7421 | 696 | 0.2597 | 0.2917 | 0.5336 | 0.2600 | 39 | 3 | 400 | 0.4799 |
| 8 | 2025-01-01 | 2025-07-01 | 8107 | 687 | 0.2625 | 0.3231 | 0.4973 | 0.2600 | 95 | 3 | 400 | 0.5924 |
| 9 | 2025-07-01 | 2026-01-01 | 8799 | 698 | 0.2670 | 0.3109 | 0.5357 | 0.2600 | 53 | 3 | 100 | 0.7135 |
| 10 | 2026-01-01 | 2026-07-01 | 9494 | 679 | 0.2704 | 0.3284 | 0.4908 | 0.2700 | 33 | 5 | 400 | 0.8675 |
| 11 | 2026-07-01 | 2027-01-01 | 10169 | 532 | 0.2741 | 0.2801 | 0.5229 | 0.2600 | 32 | 5 | 400 | 0.9323 |

## SHAP mean |value| (seed 0, OOS)

| feature | mean_abs_shap |
|---|---|
| atr_pct | 0.0735 |
| stop_atr | 0.0438 |
| range_atr | 0.0302 |
| hour_cos | 0.0235 |
| upper_wick_ratio | 0.0208 |
| stoch_d | 0.0150 |
| wick_to_range_ratio | 0.0144 |
| clv | 0.0142 |
| stoch_diff | 0.0090 |
| minute_sin | 0.0089 |
| bb_pct_b | 0.0085 |
| adx | 0.0082 |
| lower_wick_ratio | 0.0080 |
| stoch_k | 0.0078 |
| hour_sin | 0.0052 |
| minute_cos | 0.0046 |
| side | 0.0012 |
