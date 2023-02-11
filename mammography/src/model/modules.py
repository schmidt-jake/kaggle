from typing import Dict

import torch
from torchvision.transforms.functional import convert_image_dtype


class ImageFuser(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, neck: torch.nn.Module) -> None:
        """
        Concatenates the feature maps produced by `feature_extractor`, pools and flattens them,
        then feeds it into `neck`.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.neck = neck
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=4)

    def forward(self, *imgs: torch.Tensor) -> torch.Tensor:
        """
        Inputs should be tensors with shape = `(N, C, H, W)` and dtype `uint`.
        """
        features = [
            self.pool(
                self.feature_extractor(
                    convert_image_dtype(img[:, :3, :, :], dtype=torch.half if img.is_cuda else torch.float).expand(
                        -1, 3, -1, -1
                    )
                )
            )
            for img in imgs
        ]
        features = torch.cat(features, dim=1)
        return self.neck(features)


class Network(torch.nn.Module):
    def __init__(self, image_fuser: ImageFuser, neck: torch.nn.Module) -> None:
        super().__init__()
        self.image_fuser = image_fuser
        self.neck = neck
        self.density_predictor = torch.nn.LazyLinear(out_features=1)

    def forward(self, age: torch.Tensor, **imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        img_features: torch.Tensor = self.image_fuser(*imgs.values())
        density_logit: torch.Tensor = self.density_predictor(img_features)
        cancer_logit: torch.Tensor = self.neck(torch.cat([img_features, age, density_logit.sigmoid()], dim=1))
        return {"cancer": cancer_logit.squeeze(dim=1), "density": density_logit.squeeze(dim=1)}
