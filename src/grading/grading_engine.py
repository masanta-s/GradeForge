"""Orchestrator: answer key + a student's segmented answers -> a graded paper.

Routes MCQs to the MCQ grader, written answers to the hybrid subjective grader, and mixed
"choose and justify" questions to both (option -> option_marks, justification -> the rest), then
collects *reasons* for teacher review — low-confidence OCR, ambiguous MCQ marks, LLM/local
disagreement, text the segmenter couldn't place — so the review screen can say why.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from src.diagram.detector import DiagramRegion
from src.diagram.evaluator import DiagramEvaluator, DiagramScore
from src.grading.answer_key import AnswerKey, Question
from src.grading.mcq_grader import MCQResult, grade_mcq, split_mixed_answer
from src.grading.structured_output import CompletionFn
from src.grading.subjective_grader import GradedExample, SubjectiveGrader, SubjectiveResult
from src.ocr.answer_segmenter import SegmentationResult


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
    diagram: DiagramScore | None = None

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
        diagram_evaluator: DiagramEvaluator | None = None,
        llm: CompletionFn | None = None,
        strictness: float = 50,
    ):
        self.answer_key = answer_key
        self.subjective = subjective_grader or SubjectiveGrader()
        self._diagram_evaluator = diagram_evaluator
        self.llm = llm
        self.strictness = strictness

    @property
    def diagram_evaluator(self) -> DiagramEvaluator:
        if self._diagram_evaluator is None:
            self._diagram_evaluator = DiagramEvaluator(vlm=self.llm)
        return self._diagram_evaluator

    def grade_question(self, question: Question, answer: str,
                       examples: list[GradedExample] | None = None,
                       diagrams: list[DiagramRegion] | None = None) -> QuestionGrade:
        diagrams = diagrams or []
        if not answer.strip() and not diagrams:
            return QuestionGrade(question.id, question.qtype, 0.0, question.max_marks, "", "not_attempted",
                                 feedback="Not attempted.")
        grade = self._grade_text(question, answer, examples)
        if question.diagram is not None:
            score = self.diagram_evaluator.evaluate(question.text, question.diagram, diagrams, self.strictness)
            grade.marks += score.marks
            grade.diagram = score
            grade.review_reasons += score.review_reasons
            grade.feedback = f"{grade.feedback} Diagram: {score.feedback}".strip()
        grade.max_marks = question.max_marks
        return grade

    def _grade_text(self, question: Question, answer: str,
                    examples: list[GradedExample] | None) -> QuestionGrade:
        """The option and/or written part. A diagram-only question has none (0 of 0 marks)."""
        if not answer.strip():
            return QuestionGrade(question.id, question.qtype, 0.0, question.max_marks - question.diagram_marks,
                                 "", "graded", feedback="No written answer." if question.has_written_part else "")
        if question.qtype == "mcq":
            result = grade_mcq(question, answer)
            return QuestionGrade(question.id, question.qtype, result.marks, question.max_marks, answer, "graded",
                                 feedback=_option_feedback(question, result), review_reasons=_mcq_reasons(result),
                                 detail=result)
        if question.qtype == "mixed":
            return self._grade_mixed(question, answer, examples)
        if not question.has_written_part:
            return QuestionGrade(question.id, question.qtype, 0.0, 0.0, answer, "graded")

        written = replace(question, max_marks=question.written_marks)
        result = self.subjective.grade(written, answer, self.strictness, llm=self.llm, examples=examples or [])
        reasons = [f"Check: {result.review_reason}"] if result.needs_review else []
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
            reasons.append(f"Check the justification: {justified.review_reason}")
        if not choice.correct and justified.marks > 0:
            reasons.append("wrong option, but the justification earned marks: check your board's rule")
        feedback = f"{_option_feedback(question, choice)} {justified.feedback}".strip()
        return QuestionGrade(question.id, question.qtype, choice.marks + justified.marks, question.max_marks,
                             answer, "graded", feedback=feedback, review_reasons=reasons,
                             detail=MixedResult(choice, justified))

    def grade_answers(self, answers: dict[str, str],
                      diagrams: dict[str, list[DiagramRegion]] | None = None) -> PaperGrade:
        diagrams = diagrams or {}
        return PaperGrade(
            exam_name=self.answer_key.exam_name,
            strictness=self.strictness,
            questions=[self.grade_question(q, answers.get(q.id, ""), diagrams=diagrams.get(q.id))
                       for q in self.answer_key.questions],
        )

    def grade_segmentation(self, segmentation: SegmentationResult) -> PaperGrade:
        answers = {qid: seg.text for qid, seg in segmentation.answers.items()}
        diagrams = {qid: seg.diagrams for qid, seg in segmentation.answers.items() if seg.diagrams}
        paper = self.grade_answers(answers, diagrams)
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
