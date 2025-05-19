import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from scipy import stats

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("initializing")

np.random.seed(42)

logger.info("initializing dummy data")
X, y = make_regression(n_samples=100, n_features=1, n_informative=1, noise=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

logger.info("data shapes after splits: ")
logger.info('overall')
logger.info(f"X: {X.shape}")
logger.info(f"y: {y.shape}")

logger.info('train')
logger.info(f"X_train: {X_train.shape}")
logger.info(f"y_train: {y_train.shape}")

logger.info('test')
logger.info(f"X_test: {X_test.shape}")
logger.info(f"y_test: {y_test.shape}")

logger.info("injecting anomalies")
anomaly_1_X = np.array([1]).reshape(-1,1)
anomaly_1_y = 30

X_test = np.vstack([X_test, anomaly_1_X])
y_test = np.append(y_test, anomaly_1_y)

logger.info("fitting model")
model = BayesianRidge()
fitted = model.fit(X_train, y_train)
y_test_pred, y_test_pred_std = fitted.predict(X_test, return_std=True)

def get_prob_of_point(row):
    """
    Get posterior probability of point being an anomaly
    :param row: row to process
    :return: posterior probability of seeing a value at least as extreme as this one
    """""
    y_test = row['y_test']
    y_test_pred = row['y_test_pred']
    y_test_pred_std = row['y_test_pred_std']

    norm = stats.norm(loc=y_test_pred, scale=y_test_pred_std)

    if y_test > y_test_pred:
        prob = 1 - norm.cdf(y_test)
    else:
        prob = norm.cdf(y_test)

    return prob

# Posterior probabilities
df = pd.DataFrame()
df['y_test'] = y_test
df['y_test_pred'] = y_test_pred
df['y_test_pred_std'] = y_test_pred_std

df['pprobs'] = df.apply(get_prob_of_point, axis=1)
logger.info("probs < 0.05")
logger.info(df[df['pprobs'] < 0.05])


# Plots
logger.info("plotting y pred vs. X test")
fig, ax = plt.subplots()
ax.scatter(X_test, y_test)
ax.scatter(X_test, y_test_pred, c='r')
plt.fill_between(np.reshape(X_test, (-1,)), y_test_pred-y_test_pred_std, y_test_pred+y_test_pred_std, color='red', alpha=0.3, label='95% CI')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Two Scatter Plots on the Same Axes')
#plt.show()
plt.savefig('plots/main_1d.png')
plt.close()

logger.info("plotting error")
fig, ax = plt.subplots()
mse = (y_test - y_test_pred)**2
ax.scatter(X_test, mse)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
#plt.show()
plt.savefig('plots/main_1d-error.png')
plt.close()

