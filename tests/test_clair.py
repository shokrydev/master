from scripts.score_bentxt_clair import build_request_payload
from src.evaluation.bentxt_records import BENTxTPrediction
from src.evaluation.clair import (
    caption_records,
    format_clair_prompt,
    parse_clair_response,
    summarize_clair_rows,
)


def test_format_clair_prompt_uses_published_bullet_format():
    prompt = format_clair_prompt("candidate", ("reference one", "reference two"))
    assert "Candidate set:\n- candidate\n" in prompt
    assert "Reference set:\n- reference one\n- reference two\n" in prompt
    assert 'key "score"' in prompt


def test_llama_request_is_greedy_and_disables_hidden_reasoning():
    payload = build_request_payload("prompt", max_new_tokens=256, judge_label="judge")
    assert payload["model"] == "judge"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 256
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_format"] == "none"


def test_parse_clair_response_accepts_json_after_reasoning():
    parsed = parse_clair_response('<think>comparison</think>\n{"score": 87, "reason": "same scene"}')
    assert parsed.score == 87
    assert parsed.reason == "same scene"
    assert parsed.parse_method == "json"
    assert parsed.error is None

    after_unrelated_json = parse_clair_response('{"note": 7}\n{"score": 91, "reason": "match"}')
    assert after_unrelated_json.score == 91


def test_parse_clair_response_records_fallback_and_failure():
    fallback = parse_clair_response("Score: 62.5. Reason: partially aligned")
    assert fallback.score == 62.5
    assert fallback.parse_method == "numeric_fallback"
    assert parse_clair_response("cannot determine").score is None
    assert parse_clair_response('{"score": 101}').score is None


def test_caption_filter_and_summary_do_not_hide_failures():
    base = {
        "prediction": "answer",
        "target_texts": ("target",),
        "sample_id": "sample",
        "patch_id": "patch",
        "task_category": "category",
        "split": "bench",
    }
    records = [
        BENTxTPrediction(task_type="captioning", **base),
        BENTxTPrediction(task_type="mcq", **base),
    ]
    assert len(caption_records(records)) == 1
    summary = summarize_clair_rows(
        [
            {"score": 80, "parse_method": "json"},
            {"score": 40, "parse_method": "numeric_fallback"},
            {"score": None, "parse_method": None},
        ]
    )
    assert summary["mean_clair_0_100"] == 60
    assert summary["num_parse_failures"] == 1
    assert summary["parse_success_rate"] == 2 / 3
