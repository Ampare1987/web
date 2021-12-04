import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
HOUSING_PATH = "D:\\"


def load_hosing_data(housing_path=HOUSING_PATH):
    scv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(scv_path)


housing = load_hosing_data()
print(housing.head())
housing.info()


housing.hist(bins=50, figsize=(20, 15))
# plt.show()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(housing["income_cat"].value_counts() / len(housing))

for set in (strat_test_set, strat_train_set):
    set.drop(["income_cat"],axis=1, inplace=True)

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
# plt.show()
plt.legend()
# Looking for Correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)




