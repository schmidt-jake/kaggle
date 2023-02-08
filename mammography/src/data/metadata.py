from argparse import ArgumentParser

import pandas as pd

# from sklearn.model_selection import train_test_split


def fix_dtypes(meta: pd.DataFrame) -> pd.DataFrame:
    for col in meta.columns:
        try:
            meta[col] = pd.to_numeric(meta[col], downcast="unsigned")
        except (ValueError, TypeError):
            continue
    for col in ["laterality", "density"]:
        if col in meta.columns:
            meta[col] = meta[col].astype("category")
    return meta


def get_breast_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    # only select the standard CC and MLO views
    views = meta[meta["view"].isin(["CC", "MLO"])]
    columns = views.columns.drop(["view", "image_id"])
    breasts = pd.pivot_table(
        views,
        index=columns.to_list(),
        columns="view",
        values="image_id",
        aggfunc=list,
    )
    breasts.reset_index(inplace=True)
    if breasts.isna().any().any():
        raise RuntimeError("Got NaNs in metadata!")
    if set(meta["prediction_id"]) != set(breasts["prediction_id"]):
        raise RuntimeError("Missing prediction IDs in metadata!")
    breasts.sort_values("prediction_id", inplace=True)
    breasts = fix_dtypes(breasts)
    print(breasts.info())
    return breasts


def main(input_filepath: str, output_filepath: str) -> None:
    meta = pd.read_csv(input_filepath)
    if "predicton_id" not in meta.columns:
        meta["prediction_id"] = meta["patient_id"] + "_" + meta["laterality"]
    # meta.query("patient_id != 27770", inplace=True)
    # meta.query("image_id != 1942326353", inplace=True)
    # meta.query("implant == 0", inplace=True)
    breasts = get_breast_metadata(meta)
    breasts.to_pickle(output_filepath)

    # hold-out by:
    # - machine_id
    # - patient_id

    # stratify by:
    # - cancer
    # - site_id

    # stratify = breasts.loc[:, ["site_id", "cancer", "density", "age"]]
    # stratify["age"] = pd.cut(stratify["age"], bins=4)

    # train_ix, val_ix = train_test_split(breasts.index, test_size=0.2, random_state=42, stratify=stratify)
    # train_breasts = breasts.loc[train_ix]
    # val_breasts = breasts.loc[val_ix]

    # print(train_breasts.info())
    # print(val_breasts.info())

    # train_breasts.to_pickle("gs://rnsa-kaggle/data/png/train.pickle")
    # val_breasts.to_pickle("gs://rnsa-kaggle/data/png/val.pickle")

    # train_breasts.to_pickle("mammography/data/png/train.pickle")
    # val_breasts.to_pickle("mammography/data/png/val.pickle")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_filepath", type=str)
    parser.add_argument("output_filepath", type=str)
    args = parser.parse_args()
    main(**vars(args))
