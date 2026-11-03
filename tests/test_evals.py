"""The gate's own correctness: the case file, and how a case is scored.

A gate that scores wrongly is worse than no gate, and none of this needs a
model: `CaseResult` is built by hand here and the judgement is asserted on it.
The model's part is `emporio eval`, which bills real calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emporio.agent import STORE_TOOLS
from emporio.evals import Case, CaseResult, load_cases, summarize

CASES = load_cases()


def _case(**overrides: object) -> Case:
    base: dict[str, object] = {
        "id": "caso",
        "question": "pergunta?",
        "customer_id": None,
        "tools": ("get_order",),
        "accepted": (),
        "arguments": {},
        "answer": (),
        "reason": "",
    }
    return Case(**(base | overrides))  # type: ignore[arg-type]


class TestCaseFile:
    def test_the_cases_of_this_stage_are_there(self) -> None:
        assert len(CASES) == 9
        assert len({case.id for case in CASES}) == 9

    def test_every_case_declares_a_mode_and_a_reason(self) -> None:
        for case in CASES:
            assert case.question
            assert case.reason, f"{case.id} has no `motivo`"
            # An identified case must name a customer that exists; an anonymous
            # one must not name any.
            assert case.customer_id is None or case.customer_id > 0

    def test_the_cases_cover_both_modes_and_the_no_tool_decision(self) -> None:
        assert any(case.customer_id is not None for case in CASES)
        assert any(case.customer_id is None for case in CASES)
        assert any(case.tools == () for case in CASES)
        # Both order tools are exercised: one with a number, one without. The
        # policy tool is in the gate because the failure it guards is a tool the
        # model skips, not one it calls wrongly.
        called = {name for case in CASES for name in case.tools}
        assert {"get_order", "list_orders", "search_policies"} <= called
        assert called <= {t.name for t in STORE_TOOLS}

    def test_a_tool_that_does_not_exist_is_rejected(self, tmp_path: Path) -> None:
        """A typo in a tool name would fail every run for the wrong reason."""
        path = tmp_path / "casos.yaml"
        path.write_text("- id: x\n  pergunta: oi\n  ferramentas: [consultar_pedido]\n")
        with pytest.raises(ValueError, match="ferramenta inexistente"):
            load_cases(path)

    def test_an_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "casos.yaml"
        path.write_text("- id: x\n  pergunta: oi\n  ferramanta: [get_order]\n")
        with pytest.raises(ValueError, match="chave desconhecida"):
            load_cases(path)


class TestScoring:
    def test_the_expected_sequence_of_tools_is_what_counts(self) -> None:
        right = CaseResult(case=_case(), calls=[("get_order", {"order_id": 1})])
        wrong = CaseResult(case=_case(), calls=[("list_orders", {})])
        extra = CaseResult(
            case=_case(), calls=[("get_order", {"order_id": 1}), ("list_orders", {})]
        )
        assert right.chose_right
        assert not wrong.chose_right
        assert not extra.chose_right

    def test_a_case_can_tolerate_a_drill_down_after_the_expected_call(self) -> None:
        """`list_orders` and then `get_order` is one decision, not two."""
        case = _case(tools=("list_orders",), accepted=("get_order",))
        drilled = CaseResult(
            case=case, calls=[("list_orders", {}), ("get_order", {"order_id": 19})]
        )
        assert drilled.chose_right
        # Tolerated after the expected call, never instead of it.
        assert not CaseResult(case=case, calls=[("get_order", {"order_id": 19})]).chose_right
        # And nothing else is tolerated.
        assert not CaseResult(
            case=case, calls=[("list_orders", {}), ("search_products", {})]
        ).chose_right

    def test_calling_nothing_is_the_right_answer_for_a_no_tool_case(self) -> None:
        result = CaseResult(case=_case(tools=()))
        assert result.chose_right
        assert result.passed

    def test_a_number_matches_whatever_shape_the_model_sent_it_in(self) -> None:
        case = _case(tools=("search_products",), arguments={"price_max": 1000})
        for sent in (1000, 1000.0, "1000"):
            result = CaseResult(case=case, calls=[("search_products", {"price_max": sent})])
            assert result.arguments_right, sent
        off = CaseResult(case=case, calls=[("search_products", {"price_max": 100})])
        assert not off.arguments_right

    def test_text_arguments_match_by_content(self) -> None:
        """ "Yamaha C40" and "Yamaha C40 Nylon Natural" are the same decision."""
        case = _case(tools=("get_product",), arguments={"name": "Yamaha C40"})
        result = CaseResult(
            case=case, calls=[("get_product", {"name": "yamaha c40 nylon natural"})]
        )
        assert result.arguments_right
        other = CaseResult(case=case, calls=[("get_product", {"name": "Giannini GF-1R"})])
        assert not other.arguments_right

    def test_a_missing_argument_is_a_miss_not_a_crash(self) -> None:
        case = _case(arguments={"order_id": 1})
        assert not CaseResult(case=case, calls=[("get_order", {})]).arguments_right

    def test_the_answer_check_accepts_any_of_the_fragments(self) -> None:
        case = _case(tools=(), answer=("campo grande", "ms"))
        assert CaseResult(case=case, answer="Ficamos em Campo Grande!").answer_right
        assert not CaseResult(case=case, answer="Ficamos em Dourados.").answer_right
        # Declared nothing: not checked, and not counted either way.
        assert CaseResult(case=_case(tools=()), answer="qualquer coisa").answer_right is None

    def test_a_wrong_answer_does_not_fail_the_gate_but_a_malformed_call_does(self) -> None:
        wording = CaseResult(case=_case(tools=(), answer=("campo grande",)), answer="Em MS.")
        assert wording.answer_right is False
        assert wording.passed

        malformed = CaseResult(case=_case(tools=()), stop_reasons=["malformed_tool_use"])
        assert not malformed.passed

    def test_invalid_arguments_fail_the_gate(self) -> None:
        result = CaseResult(
            case=_case(),
            calls=[("get_order", {"order_id": "um"})],
            invalid_args=["get_order: input inválido"],
        )
        assert not result.passed

    def test_the_summary_is_green_only_when_every_case_is(self) -> None:
        green = [CaseResult(case=_case(tools=()))]
        assert summarize(green).passed

        red = [*green, CaseResult(case=_case(), calls=[("list_orders", {})])]
        summary = summarize(red)
        assert not summary.passed
        assert summary.chose_right == 1
        assert summary.cases == 2

    def test_a_call_error_is_never_a_pass(self) -> None:
        assert not CaseResult(case=_case(tools=()), error="ThrottlingException").passed
