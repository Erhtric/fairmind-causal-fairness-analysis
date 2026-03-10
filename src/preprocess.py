import pandas as pd


def preprocess_adult_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Adult dataset.

    Note:
        - Bins the 'hours-per-week' column into categorical intervals.

    Args:
        df (pd.DataFrame): The input DataFrame containing the Adult dataset.
    Returns:
        pd.DataFrame: The preprocessed DataFrame with binned 'hours-per-week'.
    """
    bins = [0, 20, 40, 60, 80, float("inf")]
    labels = ["0-20", "21-40", "41-60", "61-80", "80+"]
    df["hours-per-week"] = pd.cut(
        df["hours-per-week"], bins=bins, labels=labels, right=False
    )
    return df
