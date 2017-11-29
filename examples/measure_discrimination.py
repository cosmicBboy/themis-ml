"""Example code snippet for measuring discrimination in a dataset.

Below we use the German Credit data to measure the mean difference in "good"
and "bad" loan outcomes among men and women.
"""

from themis_ml.datasets import german_credit
from themis_ml.metrics import mean_difference, normalized_mean_difference

# load the german credit data
df = german_credit(raw=True)

# target variable
# values: 1 = low credit risk, 0 = high credit risk
credit_risk = df["credit_risk"]

# get sex of the individual from the "personal_status_and_sex" column.
# values are:
# - "male_divorced/separated"
# - "female_divorced/separated/married"
# - "male_single"
# - "male_married/widowed"
# - "female_single
s_map = {"male": 0, "female": 1}
sex = df["personal_status_and_sex"].map(lambda x: s_map[x.split("_")[0]])

# get foreign worker status
# 1 = yes, 0 = no
foreign = df["foreign_worker"]

# The mean difference scores below suggest that men and non-foreign workers
# are more likely to have low credit risks compared to women and foreign
# workers respectively.
print("Mean difference scores:")
print("protected class = sex: %s" % mean_difference(credit_risk, sex)[0])
# 0.0748013090229
print("protected class = foreign: %s" %
      mean_difference(credit_risk, foreign)[0])
# 0.199264685246

print("\nNormalized mean difference scores:")
# normalized mean difference
print("protected class = sex: %s" %
      normalized_mean_difference(credit_risk, sex)[0])
# 0.0772946859903
print("protected class = foreign: %s" %
      normalized_mean_difference(credit_risk, foreign)[0])
# 0.63963963964
