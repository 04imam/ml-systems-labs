# Lab 1: Environment and First System Measurements

## 1. Goal

The goal of this lab was to create a reproducible Python environment, train two baseline machine learning models, measure their system cost, and evaluate where the models could be deployed.

## 2. Method

The Breast Cancer Wisconsin dataset from scikit-learn was used.

The dataset was divided into:
- 70% training data
- 30% testing data

The split used stratification and random_state=42.

Two baseline models were trained:

1. Logistic Regression
   - max_iter=1000
   - random_state=42

2. Random Forest
   - n_estimators=100
   - random_state=42

Training time was measured after one warm-up run and repeated five times. The median training time was reported.

Single-sample inference was repeated 100 times and the median inference time was reported.

Model size was measured after saving the models using joblib.

Memory usage was measured using psutil.

## 3. Results

### Model Accuracy

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.9415 |
| Random Forest | 0.9357 |

### System Measurements

| Model | Median Training Time (ms) | Median Inference Time (ms) | Model Size (KB) | Training Memory (MB) | Inference Memory (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 257.7420 | 0.030750 | 1.03 | 266.97 | 266.97 |
| Random Forest | 99.0412 | 1.416900 | 284.07 | 268.63 | 268.72 |

## 4. Deployment Suitability

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | Suitable | Suitable | Not suitable | Not suitable |
| Random Forest | Suitable | Suitable | Not suitable | Not suitable |

Logistic Regression is suitable for Cloud and Edge deployment because its inference latency is far below the required limits and its model size is very small. However, its measured memory usage is approximately 267 MB, which is slightly above the Mobile budget of 256 MB.

Random Forest is also suitable for Cloud and Edge deployment. Its inference latency is approximately 1.42 ms and its model size is approximately 284 KB. However, its measured memory usage is approximately 269 MB, which exceeds the Mobile memory limit.

Neither model is suitable for TinyML because their measured memory usage is far above the TinyML limit of 256 KB. Random Forest also exceeds the TinyML model-size limit of 100 KB.

## 5. Conclusions

1. Logistic Regression achieved slightly higher accuracy than Random Forest on the test dataset.

2. Random Forest trained faster than Logistic Regression, but Logistic Regression had much lower inference latency and a much smaller model size.

3. Both models are suitable for Cloud and Edge environments based on the measured deployment requirements, but neither model fits the Mobile or TinyML memory budgets.
