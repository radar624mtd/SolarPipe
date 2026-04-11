# PINN v1 Training Results

Generated: 2026-04-11T04:25:30.957887+00:00
Mode: residual
Train: 1884 events | Holdout: 90 events (2026)

## Holdout Performance

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| Physics-only (ODE) | 20.15h | — | — |
| Phase 8 baseline | 12.33h | — | — |
| PINN v1 (this) | 9.05h | 12.89h | +1.19h |

CV MAE: 11.09h ± 1.76h (walk-forward, 5 folds)

## CCMC 4-Event Comparison

| Event | Truth | Physics | PINN v1 | Error | Ph8 |
|---|---:|---:|---:|---:|---:|
| Jan-18 X1.9 | 24.77h | 37.24h | 45.52h | +20.75h | 43.52h |
| Mar-18 SIDC | 58.90h | 55.94h | 56.21h | -2.69h | 58.78h |
| Mar-30 | 56.08h | 39.47h | 47.31h | -8.77h | 48.62h |
| Apr-01 | 39.28h | 48.50h | 44.94h | +5.66h | 47.77h |

CCMC MAE: **9.47h** (Ph8: 8.57h)

## Top 20 Feature Importances

| Feature | Importance |
|---|---:|
| cme_longitude | 1792 |
| preceding_cme_angular_sep_min | 1662 |
| preceding_cme_speed_mean | 1321 |
| preceding_cme_speed_max | 1266 |
| cdaw_2nd_speed_20rs | 1253 |
| cdaw_angular_width_deg | 1248 |
| sw_bz_ambient | 1243 |
| delta_v_kms | 1182 |
| omni_24h_speed_mean | 1162 |
| omni_24h_bz_std | 1160 |
| cme_latitude | 1160 |
| omni_24h_bx_mean | 1156 |
| omni_24h_bz_sigma | 1147 |
| omni_24h_by_mean | 1123 |
| cdaw_2nd_speed_init | 1122 |
| cdaw_2nd_speed_final | 1117 |
| omni_48h_speed_gradient | 1115 |
| cdaw_linear_speed_kms | 1094 |
| cme_speed_kms | 1085 |
| omni_24h_ae_max | 1050 |
