"""Define data normalization map for German Credit Data.

See here for more details:
https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
"""

import pandas as pd

from .data_types import VariableType, Variable, VariableMap, string_cleaner


# A categorical map for variables in the census income "adult" dataset.
# This map is an ordered dictionary where:
# - keys are the column names
# - the index of the tuple is the ordering for each variable in the data file:
#   https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
# - metadata about the how to map codes to human-readable values are found
#   in this document:
#   https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar94.pdf
# - values are None if the variable is numeric or a dictionary if the variable
#   is categorical. This dictionary maps variable codes to human-readable
#   values.
census_income_variable_map = VariableMap([
    Variable(
        "age",
        VariableType.NUMERIC),
    Variable(
        "class_of_worker",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    # The human readable code for this variable is in "major_industry_code"
    # and we'll therefore ignore this variable for modeling purposes
    Variable(
        "detailed_industry_recode",
        VariableType.NON_ORDERED_CATEGORICAL,
        ignore=True),
    # The human readable code for this variable is in "major_occupation_code"
    # and we'll therefore ignore this varaible for modeling purposes
    Variable(
        "detailed_occupation_recode",
        VariableType.NON_ORDERED_CATEGORICAL,
        ignore=True),
    Variable(
        "education",
        VariableType.ORDERED_CATEGORICAL,
        transformer=lambda x: {
            "children": 0,
            "less_than_1st_grade": 1,
            "1st_2nd_3rd_or_4th_grade": 2,
            "5th_or_6th_grade": 3,
            "7th_and_8th_grade": 4,
            "9th_grade": 5,
            "10th_grade": 6,
            "11th_grade": 7,
            "12th_grade_no_diploma": 8,
            "high_school_graduate": 9,
            "some_college_but_no_degree": 10,
            "associates_degree-occup_/vocational": 11,
            "associates_degree-academic_program": 12,
            "bachelors_degree(ba_ab_bs)": 13,
            "masters_degree(ma_ms_meng_med_msw_mba)": 14,
            "prof_school_degree_(md_dds_dvm_llb_jd)": 15,
            "doctorate_degree(phd_edd)": 16,
        }[string_cleaner(x)]),
    Variable(
        "wage_per_hour",
        VariableType.NUMERIC),
    Variable(
        "enroll_in_edu_inst_last_wk",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "marital_stat",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "major_industry_code",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "major_occupation_code",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "race",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "hispanic_origin",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "sex",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "member_of_a_labor_union",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "reason_for_unemployment",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "full_or_part_time_employment_stat",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "capital_gains",
        VariableType.NUMERIC),
    Variable(
        "capital_losses",
        VariableType.NUMERIC),
    Variable(
        "dividends_from_stocks",
        VariableType.NUMERIC),
    Variable(
        "tax_filer_stat",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "region_of_previous_residence",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "state_of_previous_residence",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "detailed_household_and_family_stat",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "detailed_household_summary_in_household",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "instance_weight",
        VariableType.NUMERIC,
        ignore=True),
    Variable(
        "migration_code_change_in_msa",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "migration_code-change_in_reg",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "migration_code-move_within_reg",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "live_in_this_house_1_year_ago",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "migration_prev_res_in_sunbelt",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "num_persons_worked_for_employer",
        VariableType.NUMERIC),
    Variable(
        "family_members_under_18",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "country_of_birth_father",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "country_of_birth_mother",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "country_of_birth_self",
        VariableType.NON_ORDERED_CATEGORICAL
        ,transformer=string_cleaner),
    Variable(
        "citizenship",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    Variable(
        "own_business_or_self_employed",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer={
            0: "not_in_universe",
            1: "yes",
            2: "no",
        }),
    Variable(
        "fill_inc_questionnaire_for_veteran's_admin",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer=string_cleaner),
    # This was a particularly tough variable to find the code mappings.
    # You can find it in page 9-30 in this document:
    # https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar94.pdf
    Variable(
        "veterans_benefits",
        VariableType.NON_ORDERED_CATEGORICAL,
        transformer={
            0: "not_in_universe",
            1: "yes",
            2: "no"
        }),
    Variable(
        "weeks_worked_in_year",
        VariableType.NUMERIC),
    Variable(
        "year",
        VariableType.ORDERED_CATEGORICAL,
        transformer={
            94: 1,
            95: 2}),
    Variable(
        "income_gt_50k",
        VariableType.BINARY,
        transformer={
            " - 50000.": 0,
            " 50000+.": 1
        },
        is_target=True),
])


def preprocess_census_income_data(df):
    """Prepare the Census Income 1994-1995 data for modeling.

    This function converts the target variable "income_gt_50k" into a boolean
    where 1 = income >= 50K and 0 = income < 50K.

    :param pd.DataFrame df: census income data.
    :returns: dataframe ready for modeling.

    This function converts all categorical variables into dummy variables.
    """
    return pd.concat([
        df[census_income_variable_map.numeric_variables],
        df[census_income_variable_map.ordered_categorical_variables],
        df[census_income_variable_map.binary_variables],
        pd.get_dummies(
            df[census_income_variable_map.non_ordered_categorical_variables]),
        df[census_income_variable_map.targets],
        df["dataset_partition"]
    ], axis=1)

