"""Orchestrator: answer key + a student's segmented answers -> a graded paper.

Routes MCQs to the MCQ grader, written answers to the hybrid subjective grader, and mixed
"choose and justify" questions to both (option -> option_marks, justification -> the rest), then
collects *reasons* for teacher review — low-confidence OCR, ambiguous MCQ marks, LLM/local
disagreement, text the segmenter couldn't place — so the review screen can say why.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from src.grading.answer_key import AnswerKey, Question
from src.grading.mcq_grader import MCQResult, grade_mcq, split_mixed_answer
from src.grading.structured_output import CompletionFn
from src.grading.subjective_grader import GradedExample, SubjectiveGrader, SubjectiveResult
from src.ocr.answer_segmenter import SegmentationResult

_UNCERTAIN = "AI grade uncertain (model disagreement or unreliable output)"


def _mcq_reasons(result: MCQResult) -> list[str]:
    return {
        "ambiguous": ["more than one option marked"],
        "unreadable": ["could not read the chosen option"],
        "text_match": ["option matched from written text, not a letter"],
    }.get(result.method, [])


def _option_feedback(question: Question, result: MCQResult) -> str:
    return "Correct option." if result.correct else f"Correct option: {question.correct_option}."


@dataclass(frozen=True)
class MixedResult:
    """A 'choose and justify' question: the option choice and the justification, graded separately."""
    choice: MCQResult
    justification: SubjectiveResult


@dataclass
class QuestionGrade:
    question_id: str
    qtype: str
    marks: float
    max_marks: float
    answer_text: str
    status: Literal["graded", "not_attempted"]
    feedback: str = ""
    review_reasons: list[str] = field(default_factory=list)
    detail: MCQResult | SubjectiveResult | MixedResult | None = None

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)


@dataclass
class PaperGrade:
    exam_name: str
    strictness: float
    questions: list[QuestionGrade]
    unplaced_text: str = ""

    @property
    def total(self) -> float:
        return sum(q.marks for q in self.questions)

    @property
    def max_total(self) -> float:
        return sum(q.max_marks for q in self.questions)

    @property
    def percentage(self) -> float:
        return 100 * self.total / self.max_total if self.max_total else 0.0

    @property
    def review_queue(self) -> list[QuestionGrade]:
        return [q for q in self.questions if q.needs_review]


class GradingEngine:
    def __init__(
        self,
        answer_key: AnswerKey,
        *,
        subjective_grader: SubjectiveGrader | None = None,
        llm: CompletionFn | None = None,
        strictness: float = 50,
    ):
        self.answer_key = answer_key
        self.subjective = subjective_grader or SubjectiveGrader()
        self.llm = llm
        self.strictness = strictness

    def grade_question(self, question: Question, answer: str,
                       examples: list[GradedExample] | None = None) -> QuestionGrade:
        if not answer.strip():
            return QuestionGrade(question.id, question.qtype, 0.0, question.max_marks, "", "not_attempted",
                                 feedback="Not attempted.")
        if question.qtype == "mcq":
            result = grade_mcq(question, answer)
            return QuestionGrade(question.id, question.qtype, result.marks, question.max_marks, answer, "graded",
                                 feedback=_option_feedback(question, result), review_reasons=_mcq_reasons(result),
                                 detail=result)
        if question.qtype == "mixed":
            return self._grade_mixed(question, answer, examples)

        result = self.subjective.grade(question, answer, self.strictness, llm=self.llm, examples=examples or [])
        reasons = [_UNCERTAIN] if result.needs_review else []
        return QuestionGrade(question.id, question.qtype, result.marks, question.max_marks, answer, "graded",
                             feedback=result.feedback, review_reasons=reasons, detail=result)

    def _grade_mixed(self, question: Question, answer: str,
                     examples: list[GradedExample] | None) -> QuestionGrade:
        option_part, justification = split_mixed_answer(answer, question.options)
        choice = grade_mcq(question, option_part, marks=question.option_marks)
        options = "; ".join(f"{k}) {v}" for k, v in question.options.items())
        correct = f"{question.correct_option}) {question.options.get(question.correct_option, '')}"
        justification_question = replace(
            question, qtype="short", max_marks=question.justification_marks,
            text=f"{question.text}\nOptions: {options}\nCorrect option: {correct}\n"
                 f"Grade ONLY the student's justification of their choice.",
        )
        justified = self.subjective.grade(justification_question, justification, self.strictness,
                                          llm=self.llm, examples=examples or [])

        reasons = _mcq_reasons(choice) if option_part else ["could not find which option was chosen"]
        if justified.needs_review:
            reasons.append(_UNCERTAIN)
        if not choice.correct and justified.marks > 0:
            reasons.append("wrong option, but the justification earned marks: check your board's rule")
        feedback = f"{_option_feedback(question, choice)} {justified.feedback}".strip()
        return QuestionGrade(question.id, question.qtype, choice.marks + justified.marks, question.max_marks,
                             answer, "graded", feedback=feedback, review_reasons=reasons,
                             detail=MixedResult(choice, justified))

    def grade_answers(self, answers: dict[str, str]) -> PaperGrade:
        return PaperGrade(
            exam_name=self.answer_key.exam_name,
            strictness=self.strictness,
            questions=[self.grade_question(q, answers.get(q.id, "")) for q in self.answer_key.questions],
        )

    def grade_segmentation(self, segmentation: SegmentationResult) -> PaperGrade:
        answers = {qid: seg.text for qid, seg in segmentation.answers.items()}
        paper = self.grade_answers(answers)
        unplaced = "\n".join(line.text for line in segmentation.unassigned if line.text.strip())
        paper.unplaced_text = unplaced

        for grade in paper.questions:
            segment = segmentation.answers.get(grade.question_id)
            if segment is not None:
                shaky = [line for line in segment.lines if line.needs_review]
                if shaky:
                    grade.review_reasons.insert(0, f"{len(shaky)} line(s) read with low OCR confidence")
            elif grade.status == "not_attempted" and unplaced:
                grade.review_reasons.append("not found, but the sheet has text that wasn't matched to a question")
        return paper
