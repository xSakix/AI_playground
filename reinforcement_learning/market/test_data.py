import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
from sklearn.linear_model import Ridge

print('loading data...')
x_orig = np.load('x.npy')
print(x_orig.shape)
y_orig = np.load('y.npy')
print(y_orig.shape)

y = y_orig
x = x_orig
print(x.shape)
print(y.shape)
# x = np.repeat(x, 3, axis=0)
# y_orig = np.repeat(y_orig, 3, axis=0)
# print(x.shape)

print('min:', np.min(np.nan_to_num(x)))
print('max:', np.max(np.nan_to_num(x)))

# counter = 0
# counter_zeros = 0
#
# for xx in x:
#     if np.sum(xx) == 0:
#         counter += 1
#     for y in xx:
#         if y == 0:
#             counter_zeros += 1
#
# print('%d/%d = %f' % (counter, len(x), counter / len(x) * 100.))
# print('%d/%d = %f' % (counter_zeros, len(x) * 200, counter_zeros / (len(x) * 200) * 100))
#
# print(x[0:x_orig.shape[1], 0])

# xx = np.column_stack(tuple(x_orig[0][:, 0][:, i] for i in range(x_orig.shape[3])))
# print(xx.shape)
for i in range(2, x_orig.shape[1]):
    # df = pd.DataFrame(x[:, i])
    # df.fillna(method='bfill', inplace=True)
    # plt.plot(df.as_matrix())
    plt.plot(x[:, i])

plt.legend(['pct', 'lowerb', 'mean', 'median', 'higherb'])
plt.show()


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                 '%d' % int(height),
                 ha='center', va='bottom')


(labels, counts) = np.unique(y, return_counts=True)
rect = plt.bar(labels, counts)
autolabel(rect)
plt.show()

# print(np.std(x))
# s = np.std(x, axis=0)
# z = np.column_stack((np.random.normal(0, s[i], len(x)) for i in range(len(s))))
# x_new = x + z
#
# for i in range(x_orig.shape[3]):
#     df = pd.DataFrame(x_new[0:x_orig.shape[1], i])
#     df.fillna(method='bfill', inplace=True)
#     plt.plot(df.as_matrix())
#
# plt.legend(['lowerb', 'mean', 'higherb'])
# plt.show()

print(np.mean(x, axis=0))
print(np.std(x, axis=0))
# with pm.Model() as model:
#     betas = pm.Normal('betas', mu=0, sd=10, shape=6)
#     mu = pm.Deterministic('mu',1.0 +
#                           betas[0] * x[:, 0] +
#                           betas[1] * x[:, 1] +
#                           betas[2] * x[:, 2] +
#                           betas[3] * x[:, 3] +
#                           betas[4] * x[:, 4] +
#                           betas[5] * x[:, 5])
#
#     sigma = pm.Uniform('sigma', lower=0, upper=np.std(y))
#     action = pm.Normal('actions', mu=mu, sd=sigma, observed=y)
#     trace = pm.sample(1000, tune=1000)
#
# pm.traceplot(trace)
# plt.show()
#
# print(pm.summary(trace, alpha=0.11, varnames=['betas']))
#
# trace_df = pm.trace_to_dataframe(trace, varnames=['betas'])
# print(trace_df.corr().round(2))

reg = Ridge(alpha=1.0)
reg.fit(x, y)
print(reg.coef_)

df_coef = pd.DataFrame(reg.coef_.reshape(-1, len(reg.coef_)),
                       columns=['ror', 'bench', 'pct', 'lowerb', 'mean', 'median', 'higherb'])

print(df_coef.to_string())
