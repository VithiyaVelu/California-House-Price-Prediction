import pandas as pd

from preprocessing import add_features


def test_add_features_creates_all_expected_columns():
    df = pd.DataFrame([
        {
            "MedInc": 4.0,
            "HouseAge": 20,
            "AveRooms": 5.0,
            "AveBedrms": 1.1,
            "Population": 1000,
            "AveOccup": 2.5,
            "Latitude": 34.0,
            "Longitude": -118.0,
        }
    ])

    result = add_features(df)

    assert "rooms_per_person" in result.columns
    assert "bedrooms_per_room" in result.columns
    assert "population_per_household" in result.columns
    assert "population_per_room" in result.columns
    assert "medinc_houseage_interaction" in result.columns

    assert result.loc[0, "rooms_per_person"] == 5.0 / 2.5
    assert result.loc[0, "bedrooms_per_room"] == 1.1 / 5.0
    assert result.loc[0, "population_per_household"] == 1000 / 20
    assert result.loc[0, "population_per_room"] == 1000 / 5.0
    assert result.loc[0, "medinc_houseage_interaction"] == 4.0 * 20
