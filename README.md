### Background
This repo implements a simple anomaly detection algorithm using the BayesianRidge package.

### Overview
We load data from sklearn.datasets to make a dummy regression problem and inject a single anomaly into the test set.
We then fit a BayesianRidge model on the train set, predict the test set, and calculate the posterior probability of seeing 
data as extreme as what we observe (or more) in the test set with respect to a normal distribution centered at the prediction.

### Results
We can identify the anomaly easily as shown.

```
y_test  y_test_pred  y_test_pred_std    pprobs (posterior probability)
30.0    42.94359     2.745109           0.000001
```
