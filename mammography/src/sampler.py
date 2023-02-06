import pandas as pd
from torch.utils.data import WeightedRandomSampler


class BreastSampler(WeightedRandomSampler):
    def __init__(self, image_meta: pd.DataFrame) -> None:
        self.image_meta = image_meta
        self.breasts = self.image_meta.drop_duplicates(["patient_id", "laterality"])
        class_weights = {0: 1, 1: 3}
        super().__init__(
            weights=self.breasts["cancer"].map(class_weights.get), num_samples=len(self.breasts), replacement=True
        )

    def __iter__(self):
        for i in super().__iter__():
            breast = self.breasts.iloc[i]
            images = self.image_meta.loc[
                (self.image_meta["patient_id"] == breast["patient_id"])
                & (self.image_meta["laterality"] == breast["laterality"]),
                "image_id",
            ]
            yield images.to_list()
