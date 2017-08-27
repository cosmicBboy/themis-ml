"""Define data normalization map for German Credit Data.

See here for more details:
https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
"""

import pandas as pd

from .data_types import VariableType, Variable, VariableMap

# A categorical map for variables in the german credit dataset.
# This map is an ordered dictionary where:
# - keys are the column names
# - the index of the tuple is the ordering for each variable in the data file:
#   https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
# - values are None if the variable is numeric or a dictionary if the variable
#   is categorical. This dictionary maps variable codes to human-readable
#   values.
GERMAN_CREDIT_TARGET = "credit_risk"
GERMAN_CREDIT_VARIABLE_MAP = VariableMap([
    Variable(
        "status_of_existing_checking_account",
        VariableType.ORDERED_CATEGORICAL,
        {
            "A11": 1,  # "... < 0 DM"
            "A12": 2,  # "0 <= ... < 200 DM"
            "A13": 3,  # "... >= 200 DM / salary assignments for at least 1
                       # year"
            "A14": 0,  # "no checking account"
        }),
    Variable("duration_in_month", VariableType.NUMERIC),
    Variable(
        "credit_history",
        VariableType.NON_ORDERED_CATEGORICAL,
        {
            "A30": "no_credits_taken/all_credits_paid_back_duly",
            "A31": "all_credits_at_this_bank_paid_back_duly",
            "A32": "existing_credits_paid_back_duly_till_now",
            "A33": "delay_in_paying_off_in_the_past",
            "A34": "critical_account/other_credits_existing_not_at_this_bank"
        }),
    Variable(
        "purpose",
        VariableType.NON_ORDERED_CATEGORICAL,
        {
            "A40": "car_(new)",
            "A41": "car_(used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic_appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "vacation/does_not_exist?",
            "A48": "retraining",
            "A49": "business",
            "A410": "others"
        }),
    Variable("credit_amount", VariableType.NUMERIC),
    Variable(
        "savings_account/bonds",
        VariableType.ORDERED_CATEGORICAL,
        {
            "A61": 1,  # "... < 100 DM"
            "A62": 2,  # "100 <= ... < 500 DM"
            "A63": 3,  # "500 <= ... < 1000 DM"
            "A64": 4,  # ".. >= 1000 DM"
            "A65": 0,  # "unknown/ no savings account"
        }),
    Variable(
        "present_employment_since",
        VariableType.ORDERED_CATEGORICAL,
        {
            "A71": 0,  # "unemployed"
            "A72": 1,  # "... < 1 year"
            "A73": 2,  # "1 <= ... < 4 years"
            "A74": 3,  # "4 <= ... < 7 years"
            "A75": 4,  # ".. >= 7 years"
        }),
    Variable(
        "installment_rate_in_percentage_of_disposable_income",
        VariableType.NUMERIC),
    Variable(
        "personal_status_and_sex",
        VariableType.NON_ORDERED_CATEGORICAL,
        {
            "A91": "male_divorced/separated",
            "A92": "female_divorced/separated/married",
            "A93": "male_single",
            "A94": "male_married/widowed",
            "A95": "female_single"
        }),
    Variable(
        "other_debtors/guarantors",
        VariableType.NON_ORDERED_CATEGORICAL,
        {
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor"
        }),
    Variable(
        "present_residence_since",
        VariableType.NUMERIC),
    Variable(
        "property",
        VariableType.NON_ORDERED_CATEGORICAL,
        {
            "A121": "real_estate",
            "A122": "building_society_savings_agreement/life_insurance",
            "A123": "car_or_other",
            "A124": "unknown/no_property"
        }),
    Variable("age_in_years", VariableType.NUMERIC),
    Variable(
        "other_installment_plans",
        VariableType.NON_ORDERED_CATEGORICAL,
        {
            "A141": "bank",
            "A142": "stores",
            "A143": "none"
        }),
    Variable(
        "housing",
        VariableType.NON_ORDERED_CATEGORICAL,
        {
            "A151": "rent",
            "A152": "own",
            "A153": "for free"
        }),
    Variable("number_of_existing_credits_at_this_bank", VariableType.NUMERIC),
    Variable(
        "job",
        VariableType.ORDERED_CATEGORICAL,
        {
            "A171": 0,  # "unemployed/ unskilled - non-resident"
            "A172": 1,  # "unskilled - resident"
            "A173": 2,  # "skilled employee / official",
            "A174": 3,  # "management/self-employed/highly qualified
                        #  employee/officer"
        }),
    Variable(
        "number_of_people_being_liable_to_provide_maintenance_for",
        VariableType.NUMERIC),
    Variable(
        "telephone",
        VariableType.BINARY,
        {
            "A191": 0,  # "none"
            "A192": 1,  # "yes, registered under the customers name"
        }),
    Variable(
        "foreign_worker",
        VariableType.BINARY,
        {
            "A201": 1,  # yes
            "A202": 0   # no
        }),
    Variable(
        GERMAN_CREDIT_TARGET,
        VariableType.BINARY,
        {
            1: 1,  # good
            2: 0   # bad
        },
        is_target=True),
])


def preprocess_german_credit_data(df):
    """Prepare the German Credit data for modeling.

    :param pd.DataFrame df: column and value-normalized german credit data.
    :returns: dataframe ready for modeling.

    This function converts all categorical variables into dummy variables.
    """
    GERMAN_CREDIT_VARIABLE_MAP.non_ordered_categorical_variables
    return pd.concat([
        df[GERMAN_CREDIT_VARIABLE_MAP.numeric_variables],
        df[GERMAN_CREDIT_VARIABLE_MAP.ordered_categorical_variables],
        df[GERMAN_CREDIT_VARIABLE_MAP.binary_variables],
        pd.get_dummies(
            df[GERMAN_CREDIT_VARIABLE_MAP.non_ordered_categorical_variables]),
        df[GERMAN_CREDIT_VARIABLE_MAP.targets]
    ], axis=1)
