import numpy as np
import pandas as pd
np.random.seed(0)
db = pd.read_csv("data.csv")

# Prediction of Whether Cancer is Benign or Malignant 
db_trunc = db['diagnosis']

res = db_trunc.value_counts()
B,M= res.values
print("# Benign: ",B)
print("# Maligant: ",M)
p  = B/(B+M)
print("Probability of cancer  being benign:  (Population Probability) ",p)



# Sample


n=30
sample = db_trunc.sample(n=n)
sample = sample.transform(lambda x : 1 if x=='B' else 0)
X_mean = sample.sum(axis = 0)/(len(sample))
Z = (np.sqrt(n) * (X_mean -1/2))/(1/2)
print("Value of Test Statistc:" ,Z)




for n in range(1,400,5):
    sample = db_trunc.sample(n=n)
    sample = sample.transform(lambda x : 1 if x=='B' else 0)
    X_mean = sample.sum(axis = 0)/(len(sample))

    Z = (np.sqrt(n) * (X_mean -1/2))/(1/2)
    testing_result = "Reject H0" if Z >1.96 else "Failed to Reject H0"
    print(f"n {n} value {Z :.2f}   {testing_result}")
