# Results — `xauusd_histdata_2019_2026`

Data 2019-01-02 → 2026-11-13; out-of-sample 2021-01-01 → 2026-11-13, 12 walk-forward folds (expanding, test 6 mo), seeds [0, 1, 2, 3, 4]. Values are mean ± std across seeds.

## Primary (toy) trade stream

- trades: 10,813 total, 7,969 out-of-sample
- OOS win rate 29.8%; avg R gross -0.024, cost 0.012 R, net -0.037 R
- time-exit rate 6.4%; median hold 149 min

## Prediction quality (pooled OOS)

| metric | meta-model | base-rate predictor |
|---|---|---|
| ROC-AUC | 0.5186 ± 0.0028 | 0.5 (cost-rule score: 0.5369; logistic regression: 0.5237) |
| Brier | 0.2094 ± 0.0001 | 0.2094 ± 0.0000 |
| Log-loss | 0.6097 ± 0.0003 | 0.6097 ± 0.0000 |
| Brier skill score | 0.0001 ± 0.0007 | 0 |

## Trading impact (same trades, Fusion Zero costs)

| policy | trades | avg risk | win rate | avg R net | risk-weighted R | total return | CAGR | max DD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed 1% risk | 7,969 ± 0 | 1.0% ± 0.0% | 29.8% ± 0.0% | -0.037 ± 0.000 | -0.037 ± 0.000 | -97.8% ± 0.0% | -47.8% ± 0.0% | 99.2% ± 0.0% | -0.81 ± 0.00 | -1.08 ± 0.00 | -0.48 ± 0.00 |
| ATR-scaled risk | 7,969 ± 0 | 1.1% ± 0.0% | 29.8% ± 0.0% | -0.037 ± 0.000 | -0.059 ± 0.000 | -99.8% ± 0.0% | -65.9% ± 0.0% | 99.9% ± 0.0% | -1.23 ± 0.00 | -1.60 ± 0.00 | -0.66 ± 0.00 |
| Cost-rule filter (no ML) | 3,931 ± 0 | 1.0% ± 0.0% | 32.0% ± 0.0% | 0.009 ± 0.000 | 0.009 ± 0.000 | -7.1% ± 0.0% | -1.2% ± 0.0% | 77.9% ± 0.0% | 0.16 ± 0.00 | 0.24 ± 0.00 | -0.02 ± 0.00 |
| Meta-model filter | 4,911 ± 277 | 1.0% ± 0.0% | 30.4% ± 0.3% | -0.023 ± 0.008 | -0.023 ± 0.008 | -79.5% ± 9.8% | -24.6% ± 6.2% | 91.2% ± 4.2% | -0.40 ± 0.15 | -0.56 ± 0.21 | -0.27 ± 0.06 |
| Meta-model ¼-Kelly | 3,036 ± 63 | 0.7% ± 0.1% | 31.1% ± 0.3% | -0.011 ± 0.008 | 0.003 ± 0.010 | -18.2% ± 19.0% | -3.7% ± 3.8% | 72.5% ± 5.6% | 0.04 ± 0.12 | 0.07 ± 0.18 | -0.05 ± 0.05 |

## Sharpe differences (moving-block bootstrap of daily returns)

| policy | vs | seed | ΔSharpe | 95% CI | share of resamples ≤ 0 |
|---|---|---|---|---|---|
| ATR-scaled risk | Fixed 1% risk | 0 | -0.42 | [-0.70, -0.14] | 1.00 |
| Cost-rule filter (no ML) | Fixed 1% risk | 0 | +0.97 | [+0.33, +1.58] | 0.00 |
| Meta-model filter | Fixed 1% risk | 0 | +0.26 | [-0.33, +0.82] | 0.21 |
| Meta-model filter | Fixed 1% risk | 1 | +0.61 | [+0.05, +1.18] | 0.01 |
| Meta-model filter | Fixed 1% risk | 2 | +0.46 | [-0.05, +1.00] | 0.04 |
| Meta-model filter | Fixed 1% risk | 3 | +0.26 | [-0.25, +0.79] | 0.16 |
| Meta-model filter | Fixed 1% risk | 4 | +0.48 | [-0.10, +1.06] | 0.05 |
| Meta-model ¼-Kelly | Fixed 1% risk | 0 | +0.72 | [-0.05, +1.49] | 0.03 |
| Meta-model ¼-Kelly | Fixed 1% risk | 1 | +0.76 | [-0.04, +1.50] | 0.03 |
| Meta-model ¼-Kelly | Fixed 1% risk | 2 | +0.94 | [+0.18, +1.74] | 0.01 |
| Meta-model ¼-Kelly | Fixed 1% risk | 3 | +0.99 | [+0.20, +1.75] | 0.01 |
| Meta-model ¼-Kelly | Fixed 1% risk | 4 | +0.86 | [+0.05, +1.73] | 0.02 |
| Meta-model filter | Cost-rule filter (no ML) | 0 | -0.71 | [-1.27, -0.16] | 0.99 |
| Meta-model filter | Cost-rule filter (no ML) | 1 | -0.36 | [-0.93, +0.25] | 0.89 |
| Meta-model filter | Cost-rule filter (no ML) | 2 | -0.52 | [-1.05, +0.02] | 0.97 |
| Meta-model filter | Cost-rule filter (no ML) | 3 | -0.72 | [-1.27, -0.14] | 0.99 |
| Meta-model filter | Cost-rule filter (no ML) | 4 | -0.50 | [-1.09, +0.14] | 0.93 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 0 | -0.25 | [-0.93, +0.45] | 0.76 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 1 | -0.21 | [-0.95, +0.45] | 0.73 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 2 | -0.03 | [-0.71, +0.62] | 0.52 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 3 | +0.02 | [-0.66, +0.67] | 0.46 |
| Meta-model ¼-Kelly | Cost-rule filter (no ML) | 4 | -0.11 | [-0.74, +0.57] | 0.60 |

## Per-seed

| seed | auc | brier_skill | meta_filter.trades | meta_filter.total_return | meta_filter.max_drawdown | meta_filter.sharpe | kelly.trades | kelly.total_return | kelly.max_drawdown | kelly.sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5138 | -0.0009 | 4894 | -0.8754 | 0.9473 | -0.5478 | 3037 | -0.4022 | 0.7799 | -0.0887 |
| 1 | 0.5186 | 0.0001 | 4634 | -0.6483 | 0.8431 | -0.1965 | 3100 | -0.2994 | 0.7431 | -0.0513 |
| 2 | 0.5210 | 0.0008 | 4736 | -0.7774 | 0.9102 | -0.3558 | 3081 | -0.0565 | 0.6715 | 0.1294 |
| 3 | 0.5200 | 0.0006 | 5357 | -0.8930 | 0.9461 | -0.5526 | 2938 | 0.0726 | 0.6584 | 0.1828 |
| 4 | 0.5196 | 0.0001 | 4935 | -0.7798 | 0.9140 | -0.3350 | 3025 | -0.2247 | 0.7706 | 0.0520 |

## Primary stream by year (OOS)

| decision_time | trades | win_rate | avg_r_gross | avg_cost_r | avg_r_net |
|---|---|---|---|---|---|
| 2021 | 1393 | 0.2800 | -0.0792 | 0.0160 | -0.0953 |
| 2022 | 1349 | 0.3010 | -0.0083 | 0.0157 | -0.0240 |
| 2023 | 1258 | 0.2719 | -0.1393 | 0.0179 | -0.1572 |
| 2024 | 1373 | 0.2986 | -0.0193 | 0.0129 | -0.0322 |
| 2025 | 1385 | 0.3235 | 0.0717 | 0.0072 | 0.0645 |
| 2026 | 1211 | 0.3121 | 0.0243 | 0.0036 | 0.0208 |

## Per-fold (seed 0)

| fold | test_start | test_end | n_train | n_test | train_base_rate | test_base_rate | auc | threshold | n_rounds | max_depth | min_data_in_leaf | kept_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2021-01-01 | 2021-07-01 | 2837 | 709 | 0.2711 | 0.2680 | 0.4930 | 0.2700 | 25 | 5 | 100 | 0.5261 |
| 1 | 2021-07-01 | 2022-01-01 | 3537 | 684 | 0.2714 | 0.2924 | 0.5338 | 0.2700 | 10 | 3 | 100 | 0.6038 |
| 2 | 2022-01-01 | 2022-07-01 | 4233 | 677 | 0.2743 | 0.2821 | 0.5304 | 0.2700 | 34 | 5 | 400 | 0.6056 |
| 3 | 2022-07-01 | 2023-01-01 | 4904 | 672 | 0.2753 | 0.3199 | 0.5240 | 0.2900 | 42 | 3 | 100 | 0.2470 |
| 4 | 2023-01-01 | 2023-07-01 | 5579 | 554 | 0.2807 | 0.2834 | 0.5103 | 0.2800 | 78 | 3 | 400 | 0.5108 |
| 5 | 2023-07-01 | 2024-01-01 | 6126 | 704 | 0.2811 | 0.2628 | 0.4890 | 0.2800 | 47 | 5 | 400 | 0.4446 |
| 6 | 2024-01-01 | 2024-07-01 | 6844 | 674 | 0.2792 | 0.2982 | 0.5167 | 0.2800 | 43 | 5 | 400 | 0.4688 |
| 7 | 2024-07-01 | 2025-01-01 | 7518 | 699 | 0.2809 | 0.2990 | 0.5445 | 0.2700 | 3 | 3 | 400 | 1.0000 |
| 8 | 2025-01-01 | 2025-07-01 | 8207 | 687 | 0.2826 | 0.3275 | 0.4832 | 0.2800 | 184 | 3 | 400 | 0.5735 |
| 9 | 2025-07-01 | 2026-01-01 | 8899 | 698 | 0.2858 | 0.3195 | 0.5372 | 0.2800 | 96 | 5 | 400 | 0.6433 |
| 10 | 2026-01-01 | 2026-07-01 | 9594 | 679 | 0.2884 | 0.3314 | 0.4870 | 0.2900 | 33 | 5 | 400 | 0.8203 |
| 11 | 2026-07-01 | 2027-01-01 | 10269 | 532 | 0.2911 | 0.2876 | 0.5063 | 0.2700 | 19 | 5 | 400 | 0.9793 |

## SHAP mean |value| (seed 0, OOS)

| feature | mean_abs_shap |
|---|---|
| atr_pct | 0.0515 |
| stop_atr | 0.0329 |
| range_atr | 0.0261 |
| upper_wick_ratio | 0.0226 |
| wick_to_range_ratio | 0.0198 |
| stoch_d | 0.0185 |
| clv | 0.0149 |
| hour_cos | 0.0135 |
| bb_pct_b | 0.0129 |
| adx | 0.0116 |
| minute_sin | 0.0106 |
| stoch_diff | 0.0093 |
| lower_wick_ratio | 0.0086 |
| minute_cos | 0.0079 |
| hour_sin | 0.0065 |
| stoch_k | 0.0049 |
| side | 0.0007 |
