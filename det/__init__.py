import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from engine.model import Model
from nn.tasks import DetectionModel
from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator


class YOLO_SL(Model):
    """
    YOLO-SL.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": DetectionTrainer,
                "validator": DetectionValidator,
                "predictor": DetectionPredictor,
            },
        }
