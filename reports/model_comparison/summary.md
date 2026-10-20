# Model comparison -- held-out test period

Trained on 1,630,556 rows; test 572,200 cell-days, base rate 8.22%.

## Season coverage of each split

| split | season | days | rows | fires | fire_rate |
|---|---|---|---|---|---|
| train | Summer (Dec-Feb) | 150 | 429150 | 19366 | 0.0451 |
| train | Autumn (Mar-May) | 184 | 526424 | 38142 | 0.0725 |
| train | Winter (Jun-Aug) | 177 | 506397 | 38628 | 0.0763 |
| train | Spring (Sep-Nov) | 91 | 260351 | 23563 | 0.0905 |
| val | Summer (Dec-Feb) | 90 | 257490 | 10630 | 0.0413 |
| val | Autumn (Mar-May) | 13 | 37193 | 832 | 0.0224 |
| val | Winter (Jun-Aug) | 7 | 20027 | 2052 | 0.1025 |
| val | Spring (Sep-Nov) | 91 | 260351 | 24841 | 0.0954 |
| test | Summer (Dec-Feb) | 0 | 0 | 0 |  |
| test | Autumn (Mar-May) | 79 | 226019 | 16647 | 0.0737 |
| test | Winter (Jun-Aug) | 92 | 263212 | 22911 | 0.087 |
| test | Spring (Sep-Nov) | 29 | 82969 | 7470 | 0.09 |

## Test metrics (threshold tuned on validation)

| model | features | roc_auc | pr_auc | brier | threshold | precision | recall | f1 | precision_top_5pct | sanity_pass | sanity_median_spread | train_minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ffdi_only | 1 | 0.531 | 0.0874 | 0.0753 | 0.0742 | 0.0871 | 0.8901 | 0.1586 | 0.1258 | 0/15 | 0.066 | 0.1 |
| logistic | 27 | 0.7824 | 0.2399 | 0.0685 | 0.1788 | 0.2297 | 0.5631 | 0.3263 | 0.325 | 0/15 | 0.0112 | 0.2 |
| random_forest | 27 | 0.881 | 0.4382 | 0.0576 | 0.2063 | 0.3572 | 0.6439 | 0.4595 | 0.5256 | 0/15 | 0.2575 | 0.7 |
| lgbm | 27 | 0.9021 | 0.4851 | 0.0546 | 0.2232 | 0.4094 | 0.629 | 0.496 | 0.5651 | 0/15 | 0.2932 | 0.2 |
| lgbm_no_coords | 25 | 0.8237 | 0.3077 | 0.0652 | 0.1752 | 0.2502 | 0.6315 | 0.3584 | 0.3973 | 5/15 | 0.032 | 0.3 |
| lgbm_monotone | 27 | 0.9006 | 0.4774 | 0.055 | 0.2199 | 0.4065 | 0.629 | 0.4939 | 0.5584 | 15/15 | 0.4568 | 0.2 |

## Test ROC-AUC by state

| state | rows | fires | fire_rate | ffdi_only | lgbm | lgbm_monotone | lgbm_no_coords | logistic | random_forest |
|---|---|---|---|---|---|---|---|---|---|
| Australian Capital Territory | 200 | 24 | 0.12 | 0.5 | 0.7951 | 0.7731 | 0.7532 | 0.6741 | 0.8003 |
| Tasmania | 7400 | 803 | 0.1085 | 0.5002 | 0.8297 | 0.8327 | 0.7761 | 0.6651 | 0.7712 |
| Victoria | 19800 | 2685 | 0.1356 | 0.4998 | 0.8072 | 0.8015 | 0.7134 | 0.5574 | 0.7657 |
| New South Wales | 69800 | 10844 | 0.1554 | 0.5159 | 0.8431 | 0.8423 | 0.761 | 0.7595 | 0.8323 |
| Queensland | 131000 | 11568 | 0.0883 | 0.5573 | 0.8833 | 0.8798 | 0.7998 | 0.7925 | 0.8665 |
| South Australia | 68800 | 700 | 0.0102 | 0.5414 | 0.8641 | 0.8926 | 0.7723 | 0.6364 | 0.8402 |
| Northern Territory | 93800 | 11343 | 0.1209 | 0.5034 | 0.89 | 0.887 | 0.835 | 0.8219 | 0.87 |
| Western Australia | 180200 | 9061 | 0.0503 | 0.5273 | 0.9201 | 0.9196 | 0.8154 | 0.7327 | 0.8913 |
| Offshore / unassigned | 1200 | 0 | 0.0 |  |  |  |  |  |  |

## Risk bands on the test period

| model | band | cell_days | fires | from_prob | share_of_cell_days | observed_fire_rate | share_of_all_fires |
|---|---|---|---|---|---|---|---|
| ffdi_only | Low | 848 | 38 | 0.0 | 0.0015 | 0.0448 | 0.0008 |
| ffdi_only | Moderate | 90519 | 5131 | 0.037 | 0.1582 | 0.0567 | 0.1091 |
| ffdi_only | High | 480833 | 41859 | 0.074 | 0.8403 | 0.0871 | 0.8901 |
| ffdi_only | Extreme | 0 | 0 | 0.537 | 0.0 |  | 0.0 |
| logistic | Low | 375668 | 11813 | 0.0 | 0.6565 | 0.0314 | 0.2512 |
| logistic | Moderate | 81234 | 8733 | 0.089 | 0.142 | 0.1075 | 0.1857 |
| logistic | High | 115243 | 26456 | 0.179 | 0.2014 | 0.2296 | 0.5626 |
| logistic | Extreme | 55 | 26 | 0.589 | 0.0001 | 0.4727 | 0.0006 |
| random_forest | Low | 439843 | 10033 | 0.0 | 0.7687 | 0.0228 | 0.2133 |
| random_forest | Moderate | 47576 | 6712 | 0.103 | 0.0831 | 0.1411 | 0.1427 |
| random_forest | High | 77077 | 24923 | 0.206 | 0.1347 | 0.3234 | 0.53 |
| random_forest | Extreme | 7704 | 5360 | 0.603 | 0.0135 | 0.6957 | 0.114 |
| lgbm | Low | 448404 | 8912 | 0.0 | 0.7836 | 0.0199 | 0.1895 |
| lgbm | Moderate | 51535 | 8535 | 0.112 | 0.0901 | 0.1656 | 0.1815 |
| lgbm | High | 64958 | 24060 | 0.223 | 0.1135 | 0.3704 | 0.5116 |
| lgbm | Extreme | 7303 | 5521 | 0.612 | 0.0128 | 0.756 | 0.1174 |
| lgbm_no_coords | Low | 362754 | 8166 | 0.0 | 0.634 | 0.0225 | 0.1736 |
| lgbm_no_coords | Moderate | 90750 | 9162 | 0.088 | 0.1586 | 0.101 | 0.1948 |
| lgbm_no_coords | High | 117758 | 29082 | 0.175 | 0.2058 | 0.247 | 0.6184 |
| lgbm_no_coords | Extreme | 938 | 618 | 0.588 | 0.0016 | 0.6588 | 0.0131 |
| lgbm_monotone | Low | 455630 | 9869 | 0.0 | 0.7963 | 0.0217 | 0.2099 |
| lgbm_monotone | Moderate | 43800 | 7577 | 0.11 | 0.0765 | 0.173 | 0.1611 |
| lgbm_monotone | High | 64487 | 23508 | 0.22 | 0.1127 | 0.3645 | 0.4999 |
| lgbm_monotone | Extreme | 8283 | 6074 | 0.61 | 0.0145 | 0.7333 | 0.1292 |

## Sanity check (probability must fall from very high to wet)

| place | month | ffdi_only | lgbm | lgbm_monotone | lgbm_no_coords | logistic | random_forest |
|---|---|---|---|---|---|---|---|
| Blue Mountains, NSW | Jan | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| Blue Mountains, NSW | May | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| Blue Mountains, NSW | Sep | FAIL | FAIL | pass | pass | FAIL | FAIL |
| Central Victoria | Jan | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| Central Victoria | May | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| Central Victoria | Sep | FAIL | FAIL | pass | pass | FAIL | FAIL |
| SE Queensland | Jan | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| SE Queensland | May | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| SE Queensland | Sep | FAIL | FAIL | pass | pass | FAIL | FAIL |
| South-west WA | Jan | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| South-west WA | May | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| South-west WA | Sep | FAIL | FAIL | pass | pass | FAIL | FAIL |
| Top End, NT | Jan | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| Top End, NT | May | FAIL | FAIL | pass | FAIL | FAIL | FAIL |
| Top End, NT | Sep | FAIL | FAIL | pass | pass | FAIL | FAIL |
