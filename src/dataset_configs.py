"""Dataset configurations for the multi-dataset campaign.

One entry per dataset, declaring the roles of the Standard Fairness Model, the
target state of interest and the discretisation to apply before fitting. The
same structure feeds run_fairmind, build_llm_prompt and the report pipeline, so
adding a dataset means adding an entry here and nothing else.

The binning is not cosmetic. The computation prompt enumerates one line per
(z, w) pair, and the model has been observed to give up past roughly forty of
them: the answer gets truncated, and a more compact output format makes it
worse rather than better. Every configuration below is sized to stay under that
figure, which is what the ``combinations`` note on each entry records.

Roles come from the exploratory work on each dataset and are deliberately kept
to one mediator and one confounder. Two of them would multiply the pairs.
"""

from __future__ import annotations

# Education levels grouped into tiers, so that a sixteen or seventeen level
# confounder does not blow up the prompt on its own. The groupings follow the
# usual reading of the Adult literature: below high school, high school, some
# college, bachelor, graduate.
_ADULT_EDUCATION_TIERS = {
    "Preschool": "<HS", "1st-4th": "<HS", "5th-6th": "<HS", "7th-8th": "<HS",
    "9th": "<HS", "10th": "<HS", "11th": "<HS", "12th": "<HS",
    "HS-grad": "HS-grad",
    "Some-college": "Some-college",
    "Assoc-acdm": "Some-college", "Assoc-voc": "Some-college",
    "Bachelors": "Bachelors",
    "Masters": "Grad", "Prof-school": "Grad", "Doctorate": "Grad",
}

# The same five tiers on the census wording, which spells the levels out. The
# "Children" level has no counterpart in Adult and is kept apart rather than
# folded into "<HS": it marks respondents below working age, so merging it with
# adults who left school early would conflate two different populations.
_CENSUS_EDUCATION_TIERS = {
    "Less than 1st grade": "<HS",
    "1st 2nd 3rd or 4th grade": "<HS",
    "5th or 6th grade": "<HS",
    "7th and 8th grade": "<HS",
    "9th grade": "<HS",
    "10th grade": "<HS",
    "11th grade": "<HS",
    "12th grade no diploma": "<HS",
    "High school graduate": "HS-grad",
    "Some college but no degree": "Some-college",
    "Associates degree-occup /vocational": "Some-college",
    "Associates degree-academic program": "Some-college",
    "Bachelors degree(BA AB BS)": "Bachelors",
    "Masters degree(MA MS MEng MEd MSW MBA)": "Grad",
    "Prof school degree (MD DDS DVM LLB JD)": "Grad",
    "Doctorate degree(PhD EdD)": "Grad",
    "Children": "Children",
}

DATASET_CONFIGS = [
    {
        # Reference dataset, used on its own in the earlier notebooks. The
        # values it produces are pinned by tests/test_benchmark_common.py.
        "dataset_name": "adult",
        "csv_path": "../../data/processed/adult.csv",
        "target_col": "T_income",
        "target_val": ">50K",
        "protected": "S2_gender",
        "x0": "Female",
        "x1": "Male",
        "mediators": ["hours-per-week"],
        "confounders": ["education"],
        "binning": {
            "hours-per-week": {
                "bins": [0, 20, 35, 45, 60, 100],
                "labels": ["<=20", "21-35", "36-45", "46-60", ">60"],
            },
            "education": {"mapping": _ADULT_EDUCATION_TIERS},
        },
        "combinations": 25,
    },
    {
        "dataset_name": "census_income_kdd",
        "csv_path": "../../data/processed/Census_income_kdd.csv",
        "target_col": "T_income_level",
        "target_val": 1,
        "protected": "S1_sex",
        "x0": "Female",
        "x1": "Male",
        "mediators": ["weeks_worked_in_year"],
        "confounders": ["education"],
        "binning": {
            "weeks_worked_in_year": {
                "bins": [-1, 0, 13, 26, 39, 52],
                "labels": ["0", "1-13", "14-26", "27-39", "40-52"],
            },
            "education": {"mapping": _CENSUS_EDUCATION_TIERS},
        },
        "combinations": 30,
    },
    {
        "dataset_name": "bank_marketing",
        "csv_path": "../../data/processed/bank_marketing.csv",
        "target_col": "T_deposit",
        "target_val": "yes",
        "protected": "S2_marital",
        "x0": "single",
        "x1": "married",
        "mediators": ["housing"],
        "confounders": ["education"],
        "binning": {},
        "combinations": 8,
    },
    {
        "dataset_name": "compas_two_year_recid",
        "csv_path": "../../data/processed/compas-scores-two-years.csv",
        "target_col": "T_two_year_recid",
        "target_val": 1,
        "protected": "S1_race",
        "x0": "Caucasian",
        "x1": "African-American",
        "mediators": ["priors_count"],
        "confounders": ["age_cat"],
        "binning": {
            "priors_count": {
                "bins": [-1, 0, 1, 3, 7, 100],
                "labels": ["0", "1", "2-3", "4-7", ">7"],
            },
        },
        "combinations": 15,
    },
    {
        "dataset_name": "diabetes_130",
        "csv_path": "../../data/processed/diabetes_130.csv",
        "target_col": "T_readmitted",
        "target_val": "YES",
        "protected": "S1_gender",
        "x0": "Female",
        "x1": "Male",
        "mediators": ["time_in_hospital"],
        "confounders": ["age"],
        "binning": {
            "time_in_hospital": {
                "bins": [0, 3, 7, 14],
                "labels": ["1-3", "4-7", "8-14"],
            },
        },
        "combinations": 30,
    },
    {
        "dataset_name": "german_credit",
        "csv_path": "../../data/processed/german_credit_complete.csv",
        "target_col": "T_Creditability",
        "target_val": 1,
        "protected": "S2_Sex",
        "x0": "Female",
        "x1": "Male",
        "mediators": ["Length of current employment"],
        "confounders": ["Occupation"],
        "binning": {},
        "combinations": 20,
    },
    {
        "dataset_name": "law_bar_pass",
        "csv_path": "../../data/processed/law_bar_pass_prediction.csv",
        "target_col": "T_bar_passed",
        "target_val": True,
        "protected": "S2_race",
        "x0": "Other",
        "x1": "White",
        "mediators": ["ugpa"],
        "confounders": ["fam_inc"],
        "binning": {
            "ugpa": {
                "bins": [0, 2.5, 3.0, 3.5, 4.0],
                "labels": ["<2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0"],
            },
        },
        "combinations": 20,
    },
    {
        "dataset_name": "oulad_final_result",
        "csv_path": "../../data/processed/studentInfo_OULAD.csv",
        "target_col": "T_final_result",
        "target_val": "Pass",
        "protected": "S1_gender",
        "x0": "F",
        "x1": "M",
        "mediators": ["studied_credits"],
        "confounders": ["highest_education"],
        "binning": {
            "studied_credits": {
                "bins": [0, 60, 90, 120, 180, 700],
                "labels": ["<=60", "61-90", "91-120", "121-180", ">180"],
            },
        },
        "combinations": 25,
    },
    {
        "dataset_name": "student_mat",
        "csv_path": "../../data/processed/student_mat.csv",
        "target_col": "T_grade",
        "target_val": 1,
        "protected": "S2_sex",
        "x0": "F",
        "x1": "M",
        "mediators": ["studytime"],
        "confounders": ["Medu"],
        "binning": {},
        "combinations": 20,
    },
    {
        "dataset_name": "student_por",
        "csv_path": "../../data/processed/student_por.csv",
        "target_col": "T_grade",
        "target_val": 1,
        "protected": "S2_sex",
        "x0": "F",
        "x1": "M",
        "mediators": ["studytime"],
        "confounders": ["Medu"],
        "binning": {},
        "combinations": 20,
    },
]


def config_by_name(name: str) -> dict:
    """Return the configuration with the given dataset name."""
    for config in DATASET_CONFIGS:
        if config["dataset_name"] == name:
            return dict(config)
    known = ", ".join(c["dataset_name"] for c in DATASET_CONFIGS)
    raise KeyError(f"unknown dataset {name!r}. Known: {known}")
