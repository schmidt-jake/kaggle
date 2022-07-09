import torch
from torchvision.models.feature_extraction import create_feature_extractor


class Normalizer(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalizes an input image.

        Parameters
        ----------
        img : torch.Tensor
            The input image minibatch, NHWC.

        Returns
        -------
        torch.Tensor
            The normalized image batch, NCHW.
        """
        img = img.clone()
        _min = img.amin(dim=(1, 2), keepdim=True)
        _max = img.amax(dim=(1, 2), keepdim=True)
        img = (img - _min) / (_max - _min)
        img = img.permute(0, 3, 1, 2)
        return img


class FeatureExtractor(torch.jit.ScriptModule):
    def __init__(self, backbone: torch.nn.Module, feature_layer: str = "flatten"):
        super().__init__()
        self.feature_layer = feature_layer
        self.backbone = create_feature_extractor(backbone, return_nodes=[self.feature_layer])

    @torch.jit.script_method
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.backbone(img)[self.feature_layer]


class Classifier(torch.jit.ScriptModule):
    def __init__(self, initial_logit_bias: float, in_features: int):
        super().__init__()
        self.logit = torch.nn.Linear(in_features=in_features, out_features=1)
        self.logit.bias = torch.nn.Parameter(
            torch.tensor(initial_logit_bias, requires_grad=True),
            requires_grad=True,
        )

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit(x)


class Model(torch.jit.ScriptModule):
    def __init__(self, normalizer: Normalizer, feature_extractor: FeatureExtractor, classifier: Classifier):
        super().__init__()
        self.normalizer = normalizer
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        # self.bn = torch.nn.BatchNorm1d(num_features=self.feature_extractor)
        self.activation = torch.nn.ReLU(inplace=True)

    @torch.jit.script_method
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.normalizer(img)
        x = self.feature_extractor(x)
        x = self.activation(x)
        logit = self.classifier(x)
        return logit


class Loss(torch.jit.ScriptModule):
    def __init__(self, pos_weight: float):
        super().__init__()
        self.bce_logit_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    @torch.jit.script_method
    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.bce_logit_loss(input=logit, target=label)
