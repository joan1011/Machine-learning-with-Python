# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
#ranges of indexes from 0 to 7501 i is gonna take values from 0 to 7500.list is created for all the different transctions append function is used to append these different transcations
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset(support=number of transcation of particular product/Total number of transactions)
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
#3*7/7500= 0.0028 min_support 3 times 7 days a week divided by total transactions
# No rule
# Visualising the results put the rules into variable
results = list(rules)