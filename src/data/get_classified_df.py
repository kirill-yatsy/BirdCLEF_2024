import yaml
import pandas as pd
import glob
import os

from src.config import CONFIG


def get_classified_df( save: bool = False) -> pd.DataFrame:
    mapper = pd.read_csv("data/processed/mapper.csv")
    # create df with x = path to audio file, y = class
    def create_dataframe(path: str) -> pd.DataFrame: 
        paths = glob.glob(f"{path}/**/*.ogg")
        return pd.DataFrame(
            data={"x": paths, "species": [x.split("/")[-2] for x in paths]}
        ) 

    df = pd.DataFrame()

    birdclefs = CONFIG.datasets.birdclefs
    for year in birdclefs:
        path = birdclefs[year]
        new_df = create_dataframe(path)
        df = pd.concat([df, new_df])

    df["species"] = df["species"].astype("category")
    # df["y"] = None
    # df = pd.merge(df, mapper, on="species", how="left") 
     
    df["y"] = df["species"].cat.codes
  

    if save:
        if CONFIG.train.fine_tune:
            df.to_csv("data/processed/files_fine_tune.csv")
        df.to_csv("data/processed/files.csv")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = get_classified_df(save=True)
    ssss = df.groupby("y", as_index=True).agg({"species": "first"}).reset_index()
    # # # # select only species and y columns
    classes = ssss[["species", "y"]] 
    classes.to_csv("data/processed/fine_tune_mapper.csv", index=False)
    # print(classes[classes["y"] == 687])
    # print(df.head())
    # old_df = pd.read_csv("data/processed/files_old.csv")[]
    # new_df = pd.read_csv("data/processed/files.csv")
    # # compare the two dataframes
    # print(old_df.equals(new_df))
    # # find differences
    # print(old_df[~old_df.isin(new_df)].dropna())
