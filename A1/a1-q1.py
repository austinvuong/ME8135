import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)  # set for reprod

def stdnormal2(num_samples=10_000):
   return (np.random.normal(0, 1, num_samples) ** 2)

def chi2(K):
   samples = stdnormal2()
   for i in range(K-1):
      samples += stdnormal2()
   return samples
   
K = [1, 2, 3, 10, 100]
for k in K:
   # sns.histplot(chi2(k), kde=True, label=f'K = {k}')
   sns.kdeplot(chi2(k), label=f'K = {k}')

plt.title('Distribution of y from Samples')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()

plt.show()
