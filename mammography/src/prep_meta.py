import pandas as pd
from sklearn.model_selection import train_test_split


def fix_dtypes(meta: pd.DataFrame) -> pd.DataFrame:
    for col in meta.columns:
        try:
            meta[col] = pd.to_numeric(meta[col], downcast="unsigned")
        except (ValueError, TypeError):
            continue
    meta = meta.astype({"laterality": "category", "density": "category"})
    return meta


def main() -> None:
    meta = pd.read_csv("mammography/data/raw/train.csv")
    meta.query("patient_id != 27770", inplace=True)
    meta.query("image_id != 1942326353", inplace=True)
    meta.query("implant == 0", inplace=True)
    # only select the standard CC and MLO views
    meta = meta[meta["view"].isin(["CC", "MLO"])]
    breasts = pd.pivot_table(
        meta,
        index=["patient_id", "laterality", "cancer", "age", "machine_id", "site_id", "BIRADS", "density", "implant"],
        columns="view",
        values="image_id",
        aggfunc=list,
    )
    breasts.dropna(inplace=True)
    breasts.reset_index(inplace=True)
    breasts = fix_dtypes(breasts)

    print(breasts.info())

    # hold-out by:
    # - machine_id
    # - patient_id

    # stratify by:
    # - cancer
    # - site_id

    stratify = breasts.loc[:, ["site_id", "cancer", "density", "age"]]
    stratify["age"] = pd.cut(stratify["age"], bins=4)

    train_ix, val_ix = train_test_split(breasts.index, test_size=0.2, random_state=42, stratify=stratify)
    train_breasts = breasts.loc[train_ix]
    val_breasts = breasts.loc[val_ix]

    print(train_breasts.info())
    print(val_breasts.info())

    train_breasts.to_pickle("gs://rnsa-kaggle/data/png/train.pickle")
    val_breasts.to_pickle("gs://rnsa-kaggle/data/png/val.pickle")

    train_breasts.to_pickle("mammography/data/png/train.pickle")
    val_breasts.to_pickle("mammography/data/png/val.pickle")


if __name__ == "__main__":
    main()
