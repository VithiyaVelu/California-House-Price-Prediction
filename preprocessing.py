import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"]
    df["population_per_household"] = df["Population"] / df["AveOccup"]
    # New features for better accuracy
    df["population_per_room"] = df["Population"] / df["AveRooms"]
    df["medinc_houseage_interaction"] = df["MedInc"] * df["HouseAge"]
    return df

