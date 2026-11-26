"""Typed decision questions and normalized answers shared by inference backends."""

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, StrictBool, StrictStr, ValidationError, model_validator

from jev_eval.config import PrerequisiteError, Settings
from jev_eval.data import fingerprint, load_json

PRIMARY_DOMAINS = ("primary_research", "animal_study", "in_vivo_experiment", "depression_model")
HYBRID_QUESTIONS = (*PRIMARY_DOMAINS, "control_group", "eligible_outcomes", "overall_eligibility")


def primary_question(project, task):
    identifier = project.evaluation.task.primary_question or project.scoring.question
    if identifier not in task.questions or identifier != project.scoring.question:
        raise PrerequisiteError(
            "Configure the primary question explicitly and match the scoring question."
        )
    return identifier


def valid_batch_task(project, task):
    """Legacy binary remains one NOUL; hybrid needs explicit seven-question opt-in."""
    if project.evaluation.task.approach == "binary":
        return len(task.questions) == 1 and all(q.type == "noul" for q in task.questions.values())
    if (
        set(task.questions) != set(HYBRID_QUESTIONS)
        or any(q.type != "noul" for q in task.questions.values())
        or getattr(project.backend, "max_questions", 1) != 7
        or primary_question(project, task) != "overall_eligibility"
    ):
        return False
    reviewed = validate_task_template(project.root / "prompts/reviewed/v1/binary_noul.json")
    return task.questions["overall_eligibility"] == reviewed.questions["retain_for_fulltext"]


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number, not a boolean or string.")
    return value


Probability = Annotated[float, BeforeValidator(_number), Field(ge=0, le=1, allow_inf_nan=False)]
NonnegativeNumber = Annotated[float, BeforeValidator(_number), Field(ge=0, allow_inf_nan=False)]


class NoulQuestion(Settings):
    type: Literal["noul"]
    instructions: StrictStr = Field(min_length=1)
    criteria: dict[str, StrictStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_criteria(self):
        if self.criteria and set(self.criteria) != {"true", "false"}:
            raise ValueError("NOUL criteria must define both true and false.")
        return self


class ChoiceOption(Settings):
    name: StrictStr = Field(min_length=1)
    description: StrictStr | None = None


class ChoiceQuestion(Settings):
    type: Literal["choice"]
    instructions: StrictStr = Field(min_length=1)
    options: list[ChoiceOption] = Field(min_length=2)

    @model_validator(mode="after")
    def unique_options(self):
        if len({option.name for option in self.options}) != len(self.options):
            raise ValueError("Choice option names must be unique.")
        return self


class ScoreQuestion(Settings):
    type: Literal["score"]
    instructions: StrictStr = Field(min_length=1)
    levels: list[StrictStr] = Field(min_length=2)

    @model_validator(mode="after")
    def valid_levels(self):
        if any(not level.strip() for level in self.levels):
            raise ValueError("Score levels must describe each ordered level.")
        return self


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class TaskDefinition(Settings):
    questions: dict[str, Question] = Field(min_length=1)

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump())


class ChoiceAnswer(Settings):
    choice: StrictStr
    confidence: Probability
    probabilities: dict[str, Probability]


class ScoreAnswer(Settings):
    score: NonnegativeNumber
    normalized: Probability
    confidence: Probability
    probabilities: dict[str, Probability]
    legend: dict[str, StrictStr]


class NormalizedResponse(Settings):
    predictions: dict[str, StrictBool] = Field(default_factory=dict)
    probabilities: dict[str, Probability] = Field(default_factory=dict)
    choices: dict[str, ChoiceAnswer] = Field(default_factory=dict)
    scores: dict[str, ScoreAnswer] = Field(default_factory=dict)
    request_id: StrictStr | None = None
    returned_model: StrictStr | None = None
    model_revision: StrictStr | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict | None = None


def validate_task_template(path: Path) -> TaskDefinition:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise PrerequisiteError("Task template must be a JSON object.")
    try:
        task = TaskDefinition.model_validate({"questions": raw.get("questions")})
    except ValidationError:
        raise PrerequisiteError(
            "Task needs named NOUL, choice, or score questions with valid instructions and options."
        ) from None
    return task


def load_task(project) -> TaskDefinition:
    path = project.evaluation.task.prompt_file
    if path is None:
        if project.workflow.execution.mode != "demo":
            raise PrerequisiteError("Set task.prompt_file before hosted inference.")
        path = "prompts/drafts/binary_noul.json"
    return validate_task_template(project.root / path)


def validate_response(response: NormalizedResponse, task: TaskDefinition) -> NormalizedResponse:
    nouls = {key for key, question in task.questions.items() if question.type == "noul"}
    choices = {key for key, question in task.questions.items() if question.type == "choice"}
    scores = {key for key, question in task.questions.items() if question.type == "score"}
    # Preserve compatibility with existing Boolean demo/generic hosted answers.
    boolean_only = bool(response.predictions) and not response.probabilities
    answered_nouls = set(response.predictions) if boolean_only else set(response.probabilities)
    if (
        answered_nouls != nouls
        or set(response.choices) != choices
        or set(response.scores) != scores
        or not set(response.predictions) <= nouls
    ):
        raise PrerequisiteError("Response question IDs do not match the selected task.")
    for key, answer in response.choices.items():
        expected = {option.name for option in task.questions[key].options}
        if set(answer.probabilities) != expected or answer.choice not in expected:
            raise PrerequisiteError("Choice probabilities do not match the submitted options.")
        _validate_distribution(answer.probabilities)
        if answer.probabilities[answer.choice] + 1e-5 < max(answer.probabilities.values()):
            raise PrerequisiteError("Selected choice is not a highest-probability option.")
    for key, answer in response.scores.items():
        levels = task.questions[key].levels
        expected = {str(index): level for index, level in enumerate(levels)}
        if set(answer.probabilities) != set(expected) or answer.legend != expected:
            raise PrerequisiteError(
                "Score probabilities or legend do not match the submitted levels."
            )
        if answer.score > len(levels) - 1:
            raise PrerequisiteError("Score exceeds the top level index.")
        _validate_distribution(answer.probabilities)
    return response


def _validate_distribution(probabilities):
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-5):
        raise PrerequisiteError("Choice/score probabilities must sum to one (tolerance 0.00001).")
