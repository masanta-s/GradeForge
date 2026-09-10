"""Teacher-configurable diagram marking for a question.

Defaults follow the plan's weightage form: labels 40 %, structure 35 %, completeness 25 %.
When a component can't be measured for a sheet (no required labels, no vision model and no
reference drawing, ...), the remaining weights are rescaled to sum to 1, so the teacher's
relative priorities still hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

COMPONENTS = ("label", "structure", "completeness")


@dataclass
class DiagramSpec:
    marks: float                                    # marks for the diagram part of the question
    required_labels: list[str] = field(default_factory=list)
    description: str = ""                           # what a correct drawing shows (for the VLM)
    label_weight: float = 0.40
    structure_weight: float = 0.35
    completeness_weight: float = 0.25
    reference_image: str | None = None              # optional path to a reference drawing

    @property
    def weights(self) -> dict[str, float]:
        return {"label": self.label_weight, "structure": self.structure_weight,
                "completeness": self.completeness_weight}

    def normalized_weights(self, available: set[str]) -> dict[str, float]:
        usable = {k: w for k, w in self.weights.items() if k in available and w > 0}
        total = sum(usable.values())
        return {k: w / total for k, w in usable.items()} if total else {}

    def validate(self, question_id: str, max_marks: float) -> list[str]:
        problems = []
        if not 0 < self.marks <= max_marks:
            problems.append(f"Q{question_id}: diagram marks must be in (0, {max_marks:g}]")
        if any(w < 0 for w in self.weights.values()):
            problems.append(f"Q{question_id}: diagram weights cannot be negative")
        elif abs(sum(self.weights.values()) - 1.0) > 0.01:
            problems.append(f"Q{question_id}: diagram weights must add up to 100 %")
        if self.label_weight > 0 and not self.required_labels:
            problems.append(f"Q{question_id}: label weight is set but no required labels are listed")
        return problems
