import json
import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split


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
    def agg(df: pd.DataFrame) -> pd.Series:
        return df[["machine_id", "image_id"]].to_dict("records")

    # only select the standard CC and MLO views
    views = meta.loc[meta["view"].isin(["CC", "MLO"]), :]
    breast_meta = views.groupby(["prediction_id", "view"]).apply(agg).unstack("view").reset_index("prediction_id")

    breast_meta = breast_meta.merge(
        views.drop(["image_id", "machine_id", "view"], axis=1).drop_duplicates("prediction_id"),
        on="prediction_id",
        how="left",
        validate="1:m",
    )
    assert not breast_meta.duplicated("prediction_id").any()
    return breast_meta


def main(input_filepath: str, output_filepath: str) -> None:
    meta = pd.read_csv(input_filepath)
    meta.dropna(subset="age", inplace=True)
    if "prediction_id" not in meta.columns:
        meta["prediction_id"] = meta["patient_id"].astype(str) + "_" + meta["laterality"]
    meta.query("implant == 0", inplace=True)
    meta = meta.loc[~meta["image_id"].isin(list(json.load(open("bad_images.json")).values()))]
    breasts = get_breast_metadata(meta)

    # hold-out by:
    # - machine_id
    # - patient_id

    # stratify by:
    # - cancer
    # - site_id

    stratify = breasts.loc[:, ["site_id", "cancer", "age"]]
    stratify["age"] = pd.cut(stratify["age"], bins=3)

    train_ix, val_ix = train_test_split(breasts.index, test_size=0.2, random_state=42, stratify=stratify)
    train_breasts = breasts.loc[train_ix]
    val_breasts = breasts.loc[val_ix]

    assert set(train_breasts["prediction_id"]).isdisjoint(set(val_breasts["prediction_id"]))

    # train_breasts.to_pickle("gs://rnsa-kaggle/data/png/train.pickle")
    # val_breasts.to_pickle("gs://rnsa-kaggle/data/png/val.pickle")

    train_breasts.to_json(os.path.join(output_filepath, "train.json"), orient="records")
    val_breasts.to_json(os.path.join(output_filepath, "val.json"), orient="records")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_filepath", type=str)
    parser.add_argument("output_filepath", type=str)
    args = parser.parse_args()
    os.makedirs(args.output_filepath, exist_ok=True)
    main(**vars(args))
