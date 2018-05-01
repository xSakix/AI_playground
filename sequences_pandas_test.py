import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({'a': np.linspace(0.5, 2., num=10)})
print(df)

dfs = []
cols = []
for i in range(0,5):
    cols.append(str(i))
    dfs.append(df.shift(-i))

result = pd.concat(dfs,axis=1)
result.columns = cols
result.dropna(inplace=True)
result = result.ewm(alpha=0.55).mean()

print(result)

plt.plot(df)
plt.plot(result)
plt.show()