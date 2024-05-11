def get_class_mapper(df):
    mapper = df.groupby("species").agg({"y": "first"}).reset_index()
    mapper = mapper.set_index("species")
    mapper = mapper.sort_values("y")

    # convert to dictionary
    mapper = mapper.to_dict()["y"]
