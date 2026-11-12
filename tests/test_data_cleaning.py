import pandas as pd

from src.data_cleaning import clean_data, parse_numeric, parse_rating, standardize_yes_no


def test_rating_parser_handles_valid_and_missing_values():
    values = pd.Series(["4.1/5", " 3.5 /5 ", "NEW", "-", "", "6.0/5", "bad"])
    result = parse_rating(values)
    assert result.iloc[0] == 4.1
    assert result.iloc[1] == 3.5
    assert result.iloc[2:].isna().all()


def test_numeric_parser_removes_commas_and_marks_invalid_values_missing():
    result = parse_numeric(pd.Series(["1,200", "300", "bad", ""]), remove_commas=True)
    assert result.iloc[0] == 1200
    assert result.iloc[1] == 300
    assert result.iloc[2:].isna().all()


def test_yes_no_values_are_standardized_and_unknowns_preserved_as_missing():
    result = standardize_yes_no(pd.Series([" YES ", "no", "unknown", None]))
    assert result.iloc[0] == "Yes"
    assert result.iloc[1] == "No"
    assert result.iloc[2:].isna().all()


def test_clean_data_removes_only_exact_duplicate_and_keeps_missing_rating_and_chain_outlet():
    data = pd.DataFrame({
        "name": ["Chain", "Chain", "Other"],
        "rate": ["4.1/5", "4.1/5", "NEW"],
        "approx_cost(for two people)": ["1,200", "1,200", "300"],
        "votes": ["5", "5", "0"],
        "online_order": ["YES", "YES", "No"],
        "book_table": ["No", "No", "Yes"],
        "location": [" Koramangala ", " Koramangala ", "Indiranagar"],
        "cuisines": ["Thai", "Thai", "Cafe"],
        "rest_type": ["Dining", "Dining", "Cafe"],
        "outlet_id": [1, 1, 2],
    })
    cleaned, report = clean_data(data)
    assert len(cleaned) == 2
    assert report["duplicates_removed"] == 1
    assert cleaned["name"].tolist() == ["Chain", "Other"]
    assert pd.isna(cleaned.iloc[1]["rate"])
    assert cleaned.loc[0, "approx_cost(for two people)"] == 1200
    assert cleaned.loc[0, "online_order"] == "Yes"
    assert report["missing_ratings_after_cleaning"] == 1
