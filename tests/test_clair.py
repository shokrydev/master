from scripts.score_bentxt_clair import (
    batches,
    build_judge_messages,
    tokenize_rendered_prompts,
)
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


def test_local_judge_messages_and_batching_are_stable():
    assert build_judge_messages("prompt") == [{"role": "user", "content": "prompt"}]
    assert list(batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_multimodal_processor_receives_prompts_as_text():
    class RecordingProcessor:
        def __init__(self):
            self.args = None
            self.kwargs = None

        def __call__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return {"input_ids": "tokens"}

    processor = RecordingProcessor()
    result = tokenize_rendered_prompts(
        processor,
        ["first prompt", "second prompt"],
        max_input_length=3584,
    )

    assert processor.args == ()
    assert processor.kwargs == {
        "text": ["first prompt", "second prompt"],
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 3584,
    }
    assert result == {"input_ids": "tokens"}


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
