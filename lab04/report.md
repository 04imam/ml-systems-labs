# Lab 4: Data Pipeline, Validation and Feature Engineering

## 1. Objective

The objective of this lab was to develop a reproducible machine
learning pipeline, inspect dataset quality, prevent data leakage,
evaluate a baseline model, and assess the impact of feature engineering.

## 2. Dataset

The experiment uses historical US demographic and economic data
with income-category labels.

The dataset contains demographic and employment-related features,
including sensitive attributes such as race and sex.

The labels represent historical income categories and should not
be interpreted as measures of individual ability or merit.

## 3. Data Quality Assessment

The following results were recorded:

| Quality check | Result |
|---|---:|
| Exact duplicate records in raw data | 52 |
| Exact duplicates removed from training | 24 |
| Feature groups with conflicting labels | 5 |
| Records manually reviewed | 50 |
| Impossible values in reviewed sample | 0 |
| Missing cells in reviewed sample | 5 |
| Records containing those missing cells | 3 |

Schema validation and missing-value results are documented in
`results/validation_report.json`.

The manual review assessed plausibility and consistency, not the
factual correctness of individual income labels.

## 4. Preprocessing Pipeline

The preprocessing pipeline included:

1. Stratified 70/30 training/testing split before transformation.
2. Median imputation for numerical features.
3. Most-frequent imputation for categorical features.
4. Train-only 1st/99th percentile clipping.
5. One-hot encoding for categorical variables.
6. Standard scaling for numerical features.
7. Exact duplicate removal from training data only.

The split was performed before fitting transformations to help
prevent information leakage from the test set.

## 5. Feature Engineering

Two additional features were introduced:

**Capital net**

`capital_net = capital-gain - capital-loss`

**Hours per age**

`hours_per_age = hours-per-week / (age - 16)`

These features were evaluated against the baseline model.

## 6. Model Comparison

| Metric | Baseline | Engineered features |
|---|---:|---:|
| Accuracy | 0.8540 | 0.8578 |
| F1 score | 0.6601 | 0.6696 |

The engineered-feature model improved accuracy by 0.0038,
equivalent to 0.38 percentage points.

The F1 score increased by 0.0095.

The results suggest that the two engineered features provided
a modest improvement on the evaluated dataset.

Detailed results are stored in
`results/model_comparison.json`.

## 7. Manual Label Quality Review

A reproducible random sample of 50 records was reviewed.

Results:

- No impossible values identified.
- No education inconsistencies identified.
- No invalid income-label categories identified.
- Five missing cells found across three records.

The identified missing values were permitted and handled
during preprocessing.

The sample is documented in
`results/quality_sample_50.csv`.

## 8. Bias, Privacy and Limitations

The dataset reflects historical US demographic and
economic conditions.

Income outcomes may contain structural inequalities,
and model performance may not generalize to current populations.

Although the dataset is publicly distributed, demographic
and employment-related information warrants careful handling.

Automated checks cannot establish the factual correctness
of every label.

The model should not be used to make consequential decisions
about individuals without appropriate safeguards and
independent validation.

## 9. Reproducibility

Run the complete experiment from the Lab 4 directory:

```powershell
python src/pipeline.py
```

The experiment's supporting artifacts include:

- `results/validation_report.json`
- `results/label_quality_report.json`
- `results/model_comparison.json`
- `results/quality_sample_50.csv`
- `results/sha256_hashes.json`

Processed-data hashes support verification of generated artifacts.

## 10. Conclusion

Lab 4 demonstrated the development of a reproducible machine
learning preprocessing and evaluation pipeline.

The implementation incorporated data validation, leakage-aware
preprocessing, feature engineering, model comparison, and
manual quality assessment.

Feature engineering improved accuracy from 85.40% to 85.78%
and F1 from 0.6601 to 0.6696.

The experiment also highlighted the importance of dataset
quality, reproducibility, privacy, and bias awareness in
machine learning systems.
