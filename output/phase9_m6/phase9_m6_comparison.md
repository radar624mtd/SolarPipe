# Phase 9 M6 — Full Backtest Report

Generated: 2026-04-10T21:14:40.678414+00:00

Config: `configs/phase8_live_eval.yaml`  |  Events: 71  |  OMNI subset: 15  |  Fallbacks: 56


## Headline numbers (MAE, transit hours)

| Metric | Phase 8 | Phase 9 density | Phase 9 static ctrl |
|---|---:|---:|---:|
| MAE on OMNI-covered subset (n=15) | 10.80h | 4.42h | 6.56h |
| MAE on 71-event set (effective; fallbacks use Phase 8) | 12.33h | 10.98h | — |
| RMSE subset | — | 6.84h | — |
| Bias subset | — | -3.52h | — |

## Ship criterion (PHASE9_SPEC §7.1)

- Aggregate 71-event MAE ≤ 12.33h? **PASS** (10.98h)

## OMNI-covered subset (Phase 9 density is scored here)

| Launch (UTC) | Obs h | P8 pred | P8 err | P9 den pred | P9 den err | Δ (P9−P8) | term | shock |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-01-01T19:35:28 | 73.1 | 63.68 | -9.40 | 65.81 | -7.27 | -2.13 | target_reached | - |
| 2026-01-05T08:49:04 | 68.2 | 63.52 | -4.63 | 66.13 | -2.02 | -2.62 | target_reached | - |
| 2026-01-08T05:52:00 | 61.7 | 40.41 | -21.31 | 63.74 | +2.03 | -19.28 | target_reached | - |
| 2026-01-10T20:35:12 | 65.1 | 40.75 | -24.37 | 54.24 | -10.87 | -13.49 | target_reached | - |
| 2026-01-10T20:48:00 | 64.9 | 51.54 | -13.38 | 60.11 | -4.80 | -8.58 | target_reached | - |
| 2026-01-11T05:11:28 | 56.5 | 57.15 | +0.63 | 56.86 | +0.34 | -0.30 | target_reached | - |
| 2026-01-17T23:23:44 | 43.5 | 58.23 | +14.71 | 45.00 | +1.48 | -13.23 | shock_detected | Y |
| 2026-01-18T08:36:16 | 34.3 | 61.05 | +26.73 | 36.00 | +1.68 | -25.05 | shock_detected | Y |
| 2026-01-18T18:08:00 | 24.8 | 43.52 | +18.76 | 26.00 | +1.23 | -17.52 | shock_detected | Y |
| 2026-03-10T04:48:00 | 74.3 | 77.68 | +3.38 | 71.54 | -2.76 | -0.63 | target_reached | - |
| 2026-03-17T05:47:44 | 67.7 | 64.24 | -3.47 | 62.07 | -5.65 | +2.18 | target_reached | - |
| 2026-03-18T09:23:12 | 58.9 | 58.78 | -0.12 | 55.55 | -3.35 | +3.23 | target_reached | - |
| 2026-03-22T16:23:28 | 61.5 | 64.57 | +3.07 | 60.27 | -1.24 | -1.83 | target_reached | - |
| 2026-03-22T23:23:44 | 54.5 | 65.03 | +10.54 | 53.88 | -0.61 | -9.94 | target_reached | - |
| 2026-03-30T03:24:48 | 56.1 | 48.62 | -7.47 | 35.08 | -21.00 | +13.54 | target_reached | - |

## Fall-back events (not scored against Phase 9; use Phase 8 baseline)

| Launch (UTC) | Obs h | P8 err | density cov | termination | reason |
|---|---:|---:|---:|---|---|
| 2026-01-01T03:52:32 | 88.8 | +0.76 | 1.00 | timeout | timeout_72h |
| 2026-01-01T23:10:56 | 149.8 | -7.72 | 1.00 | timeout | timeout_72h |
| 2026-01-03T07:47:12 | 117.2 | -4.43 | 1.00 | timeout | timeout_72h |
| 2026-01-04T01:25:20 | 99.5 | -10.82 | 1.00 | timeout | timeout_72h |
| 2026-01-06T01:59:28 | 51.0 | -1.05 | 1.00 | timeout | timeout_72h |
| 2026-01-07T11:35:28 | 80.0 | -2.96 | 1.00 | timeout | timeout_72h |
| 2026-01-08T02:59:12 | 130.7 | -14.23 | 1.00 | timeout | timeout_72h |
| 2026-01-08T15:47:12 | 51.8 | -4.73 | 1.00 | timeout | timeout_72h |
| 2026-01-08T16:59:44 | 50.6 | -6.34 | 1.00 | timeout | timeout_72h |
| 2026-01-08T17:12:32 | 50.4 | -8.80 | 1.00 | timeout | timeout_72h |
| 2026-01-09T08:38:24 | 101.1 | +3.82 | 1.00 | timeout | timeout_72h |
| 2026-01-09T15:36:32 | 154.1 | -21.16 | 1.00 | timeout | timeout_72h |
| 2026-01-10T02:22:56 | 17.2 | +23.15 | 1.00 | timeout | timeout_72h |
| 2026-01-10T16:12:48 | 218.7 | -16.50 | 1.00 | timeout | timeout_72h |
| 2026-01-11T21:30:40 | 100.2 | -6.77 | 1.00 | timeout | timeout_72h |
| 2026-01-12T01:44:32 | 36.0 | +21.27 | 1.00 | timeout | timeout_72h |
| 2026-01-12T02:59:12 | 34.7 | +26.42 | 1.00 | timeout | timeout_72h |
| 2026-01-14T08:23:28 | 41.3 | +14.90 | 1.00 | timeout | timeout_72h |
| 2026-01-29T12:45:52 | 145.6 | -15.99 | 0.74 | timeout | low_coverage(0.74) |
| 2026-01-31T14:11:12 | 96.1 | +6.58 | 1.00 | timeout | timeout_72h |
| 2026-02-02T00:46:56 | 61.5 | +5.66 | 1.00 | timeout | timeout_72h |
| 2026-02-11T03:48:16 | 87.8 | -0.63 | 1.00 | timeout | timeout_72h |
| 2026-02-11T10:08:00 | 81.5 | +5.54 | 1.00 | timeout | timeout_72h |
| 2026-02-13T09:38:08 | 34.0 | +19.08 | 1.00 | timeout | timeout_72h |
| 2026-02-16T05:00:48 | 113.6 | -18.50 | 0.99 | timeout | timeout_72h |
| 2026-02-16T14:09:04 | 104.5 | -8.62 | 0.99 | timeout | timeout_72h |
| 2026-02-17T11:22:40 | 99.0 | -0.33 | 0.99 | timeout | timeout_72h |
| 2026-02-18T00:23:28 | 86.0 | +1.60 | 0.99 | timeout | timeout_72h |
| 2026-02-18T03:48:16 | 82.6 | +3.18 | 0.99 | timeout | timeout_72h |
| 2026-03-10T02:25:04 | 76.7 | +0.65 | 1.00 | timeout | timeout_72h |
| 2026-03-13T06:24:00 | 181.9 | -18.22 | 1.00 | timeout | timeout_72h |
| 2026-03-13T12:48:00 | 156.7 | -5.96 | 1.00 | timeout | timeout_72h |
| 2026-03-16T13:26:24 | 84.1 | -4.10 | 1.00 | timeout | timeout_72h |
| 2026-03-16T15:23:44 | 206.5 | -23.24 | 1.00 | timeout | timeout_72h |
| 2026-03-17T05:07:12 | 68.4 | +27.20 | 1.00 | timeout | timeout_72h |
| 2026-03-17T06:38:56 | 66.9 | +19.41 | 1.00 | timeout | timeout_72h |
| 2026-03-17T08:23:28 | 65.1 | +29.08 | 1.00 | timeout | timeout_72h |
| 2026-03-17T10:52:48 | 62.6 | +75.30 | 1.00 | timeout | timeout_72h |
| 2026-03-17T16:53:20 | 75.4 | +3.68 | 1.00 | timeout | timeout_72h |
| 2026-03-18T19:35:28 | 154.3 | -8.22 | 1.00 | timeout | timeout_72h |
| 2026-03-21T08:23:28 | 93.5 | -13.01 | 1.00 | timeout | timeout_72h |
| 2026-03-25T06:47:28 | 172.7 | -21.25 | 1.00 | timeout | timeout_72h |
| 2026-03-25T15:47:12 | 163.7 | -26.17 | 1.00 | timeout | timeout_72h |
| 2026-03-26T21:24:16 | 134.1 | -22.79 | 1.00 | timeout | timeout_72h |
| 2026-03-27T23:13:04 | 108.3 | -12.58 | 1.00 | timeout | timeout_72h |
| 2026-03-28T01:25:20 | 106.0 | -4.05 | 1.00 | timeout | timeout_72h |
| 2026-03-28T17:08:16 | 141.9 | -17.31 | 1.00 | timeout | timeout_72h |
| 2026-03-28T23:47:12 | 83.7 | -5.97 | 1.00 | timeout | timeout_72h |
| 2026-03-29T05:11:28 | 78.3 | -0.39 | 1.00 | timeout | timeout_72h |
| 2026-03-29T11:12:00 | 72.3 | +7.24 | 1.00 | timeout | timeout_72h |
| 2026-03-30T21:15:44 | 38.2 | +18.29 | 1.00 | timeout | timeout_72h |
| 2026-03-30T21:30:40 | 38.0 | +18.70 | 1.00 | timeout | timeout_72h |
| 2026-04-01T00:23:28 | 62.6 | +14.74 | 1.00 | timeout | timeout_72h |
| 2026-04-01T02:35:44 | 60.4 | +15.65 | 1.00 | timeout | timeout_72h |
| 2026-04-01T05:09:20 | 57.9 | +10.30 | 1.00 | timeout | timeout_72h |
| 2026-04-01T23:45:04 | 39.3 | +8.48 | 1.00 | timeout | timeout_72h |

## Subset tally

Phase 9 density vs Phase 8 on 15 subset events: **12 wins, 3 losses, 0 ties**.
