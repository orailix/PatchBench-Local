
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import (
    InferenceContext,
    PreparedSteeringModel,
    SteeringArtifacts,
    SteeringModel,
    TrainingContext,
)


class UnsteeredSteeringModel(SteeringModel):
    """Baseline model without steering."""

    method_name = "unsteered"

    def train(self, context: TrainingContext) -> SteeringArtifacts:
        """No training is needed for the baseline model."""
        return SteeringArtifacts()

    def prepare_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        context: InferenceContext,
    ) -> PreparedSteeringModel:
        """Return the base model unchanged."""
        return PreparedSteeringModel(model=model, tokenizer=tokenizer)
