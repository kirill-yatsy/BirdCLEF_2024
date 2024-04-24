import yaml
import pandas as pd
import glob
import os

from src.config import ConfigHolder


def get_classified_df(save: bool = False) -> pd.DataFrame:
    # create df with x = path to audio file, y = class
    def create_dataframe(path: str) -> pd.DataFrame:
        class_names = sorted(os.listdir(path))
        class_names_dic = {class_names[i]: i for i in range(0, len(class_names))}
        paths = glob.glob(f"{path}/**/*.ogg")
        # paths = [x.replace("../../", "") for x in paths]

        df = pd.DataFrame(
            data={"x": paths, "species": [x.split("/")[-2] for x in paths]}
        )
        df["species"] = df["species"].astype("category")
        df["y"] = df["species"].map(class_names_dic)
        return df

    df = pd.DataFrame()

    birdclefs = ConfigHolder.config.data.splitted_datasets.birdclefs
    for year in birdclefs:
        path = birdclefs[year]
        new_df = create_dataframe(path)
        df = pd.concat([df, new_df])

    if save:
        df.to_parquet("data/processed/files.parquet")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = get_classified_df(save=True)
    print(df.head())
