# BirdCLEF_2024

Download data:
```bash
kaggle competitions download -c birdclef-2024

unzip birdclef-2024.zip -d data/birdclef-2024

kaggle competitions download -c birdclef-2023

unzip birdclef-2023.zip -d data/birdclef-2023

kaggle competitions download -c birdclef-2022

unzip birdclef-2022.zip -d data/birdclef-2022

kaggle competitions download -c birdclef-2021

unzip birdclef-2021.zip -d data/birdclef-2021

```


## Notebooks

- notebooks/data/create_files_df.ipynb - collect classified files and creates a dataframe where x is path to file and y is label
