from logging import getLogger
from typing import Optional

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger as _WandbLogger
from pytorch_lightning.utilities.logger import _scan_checkpoints

logger = getLogger(__name__)


def use_reference_artifact(artifact_name: str, artifact_type: str, filename: str, use_as: Optional[str] = None) -> str:
    artifact: wandb.Artifact = wandb.use_artifact(artifact_or_name=artifact_name, type=artifact_type, use_as=use_as)
    filepath = artifact.get_path(name=filename).ref_target()
    logger.info(f"Using artifact {artifact_name}/{filename} from {filepath}...")
    return filepath


class WandbLogger(_WandbLogger):
    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        for time, path, score, tag in _scan_checkpoints(checkpoint_callback, self._logged_model_time):
            artifact = wandb.Artifact(
                name=f"ckpt-{self.experiment.id}",
                type="checkpoint",
                metadata={checkpoint_callback.monitor: score.item()},
            )
            artifact.add_reference(uri=f"file://{path}", name="checkpoint.pickle")
            self.experiment.log_artifact(artifact, aliases=[tag])
            logger.info(f"Logged artifact {artifact.name}:{tag} from {path}")
            self._logged_model_time[path] = time
