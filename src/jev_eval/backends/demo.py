"""Deterministic invented responses. These are not JEV predictions."""

from jev_eval.data import fingerprint
from jev_eval.tasks import NormalizedResponse


class DemoBackend:
    mode = "demo"

    def predict(self, state, model, task, generation):
        predictions = {
            question: int(fingerprint([state.study_id, model.name, question])[:8], 16) % 2 == 0
            for question in task.questions
        }
        return NormalizedResponse(
            predictions=predictions,
            model_revision="synthetic-demo-v1",
            raw={"demo": True, "predictions": predictions},
        )

    def close(self):
        pass
