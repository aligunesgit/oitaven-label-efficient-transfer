# Cost-sensitive active regional adaptation pilot

Revised class-supported spatial protocol: original test blocks frozen,256-pixel non-test blocks with32-pixel buffers. Non-test benchmark labels are used to construct class-supported subsets; acquisition never sees unacquired labels. Controller/development samples are enriched to at least20 per class; development metrics are inverse-inclusion weighted. The rejected geometric pilot is diagnostic only.

Development evaluation only; reserved test never opened. Five-band local spectral/spatial features, source-only standardization, CORAL fit on source training and unlabeled adaptation pool, multinomial logistic classifier.

Source class-balanced subsampling assumes labeled source availability. Target controller set costs700 labels, additional to acquisition budgets. Development evaluation uses2000 further labels. Zero labels are unannotated. Sampling is limited to a geometric grid within available reference support.

Hypothetical fixed miss costs [3,3,3,1,1,3,1] are used for every reported method and decision. They are not measured ecological losses. Fixed coefficients alter training/acquisition emphasis on water and artificial surfaces; adaptive coefficients can vary per class. Class balance and feature transforms are shared. Probability estimates are not claimed to be calibrated.

Fixed/Adaptive Entropy and Risk variants separate acquisition from fitting effects. Adaptive coefficients use only the disjoint700-label controller set, with minimum10 examples per class. Initial query IDs are matched across methods; initial fitted models differ where cost weights differ.

| Method | Acquired labels | Seeds | Macro-F1 | Balanced accuracy | Fixed risk | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_entropy | 105 | 5 | 0.3513 | 0.5290 | 0.7749 | 0.4183 |
| adaptive_entropy | 260 | 5 | 0.3613 | 0.5451 | 0.7615 | 0.4402 |
| adaptive_entropy | 523 | 5 | 0.3606 | 0.5627 | 0.7868 | 0.4237 |
| adaptive_entropy | 1046 | 5 | 0.3676 | 0.5672 | 0.7699 | 0.4335 |
| adaptive_risk | 105 | 5 | 0.3513 | 0.5290 | 0.7749 | 0.4183 |
| adaptive_risk | 260 | 5 | 0.3537 | 0.5329 | 0.7925 | 0.3904 |
| adaptive_risk | 523 | 5 | 0.3658 | 0.5443 | 0.7474 | 0.4091 |
| adaptive_risk | 1046 | 5 | 0.3890 | 0.5734 | 0.6646 | 0.4509 |
| budgetmatched_entropy | 105 | 5 | 0.3673 | 0.6074 | 0.7252 | 0.3525 |
| budgetmatched_entropy | 260 | 5 | 0.3742 | 0.6114 | 0.7066 | 0.3705 |
| budgetmatched_entropy | 523 | 5 | 0.3759 | 0.6162 | 0.7064 | 0.3706 |
| budgetmatched_entropy | 1046 | 5 | 0.3719 | 0.6195 | 0.7122 | 0.3630 |
| budgetmatched_risk | 105 | 5 | 0.3673 | 0.6074 | 0.7252 | 0.3525 |
| budgetmatched_risk | 260 | 5 | 0.3564 | 0.6057 | 0.7566 | 0.3188 |
| budgetmatched_risk | 523 | 5 | 0.3475 | 0.6128 | 0.7803 | 0.2831 |
| budgetmatched_risk | 1046 | 5 | 0.3408 | 0.6128 | 0.7965 | 0.2641 |
| fixed_entropy | 105 | 5 | 0.3025 | 0.4838 | 0.8247 | 0.3207 |
| fixed_entropy | 260 | 5 | 0.3004 | 0.4653 | 0.8461 | 0.3214 |
| fixed_entropy | 523 | 5 | 0.2950 | 0.4579 | 0.8807 | 0.3048 |
| fixed_entropy | 1046 | 5 | 0.2840 | 0.4629 | 0.9240 | 0.2745 |
| fixed_random | 105 | 5 | 0.3025 | 0.4838 | 0.8247 | 0.3207 |
| fixed_random | 260 | 5 | 0.3226 | 0.5245 | 0.7854 | 0.3321 |
| fixed_random | 523 | 5 | 0.3274 | 0.5232 | 0.7649 | 0.3469 |
| fixed_random | 1046 | 5 | 0.3378 | 0.5367 | 0.7456 | 0.3565 |
| fixed_risk1 | 105 | 5 | 0.3513 | 0.5290 | 0.7749 | 0.4183 |
| fixed_risk1 | 260 | 5 | 0.3481 | 0.5182 | 0.7651 | 0.4156 |
| fixed_risk1 | 523 | 5 | 0.3369 | 0.4953 | 0.7908 | 0.3650 |
| fixed_risk1 | 1046 | 5 | 0.3405 | 0.5330 | 0.7615 | 0.3655 |
| fixed_risk3 | 105 | 5 | 0.3025 | 0.4838 | 0.8247 | 0.3207 |
| fixed_risk3 | 260 | 5 | 0.2905 | 0.4843 | 0.8618 | 0.2683 |
| fixed_risk3 | 523 | 5 | 0.2890 | 0.4763 | 0.8636 | 0.2418 |
| fixed_risk3 | 1046 | 5 | 0.3001 | 0.5225 | 0.8455 | 0.2472 |
| fixed_risk5 | 105 | 5 | 0.2459 | 0.4447 | 0.8584 | 0.2639 |
| fixed_risk5 | 260 | 5 | 0.2390 | 0.4479 | 0.9147 | 0.2028 |
| fixed_risk5 | 523 | 5 | 0.2514 | 0.4514 | 0.8995 | 0.1940 |
| fixed_risk5 | 1046 | 5 | 0.2836 | 0.4980 | 0.8725 | 0.2131 |
| fixed_risk7 | 105 | 5 | 0.2003 | 0.4154 | 0.8976 | 0.2207 |
| fixed_risk7 | 260 | 5 | 0.2045 | 0.4251 | 0.9372 | 0.1726 |
| fixed_risk7 | 523 | 5 | 0.2224 | 0.4285 | 0.9180 | 0.1728 |
| fixed_risk7 | 1046 | 5 | 0.2638 | 0.4659 | 0.8905 | 0.1948 |

Single-scene pilot; seeds share one spatial split. Buffer/scale sensitivity and independent-scene confirmation remain necessary. No LLM is used and no publication-ready superiority is claimed.
