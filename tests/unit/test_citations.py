from linegate.console import citations

SCHEMA = {"L3_S32_F3850", "L1_S24_F1846", "L3_S33_F3857"}
RECORD = {"part_id": 42, "out_of_range": [{"column": "L3_S32_F3850", "value": 0.5, "low": -0.2, "high": 0.2}]}


def disposition(**overrides):
    base = {
        "route": "L1_S24-L3_S32", "recommendation": "scrap",
        "reason": "L3_S32_F3850 read 0.5 against a normal range of -0.2 to 0.2. Similar parts failed often.",
        "out_of_range": [{"column": "L3_S32_F3850", "value": 0.5, "low": -0.2, "high": 0.2}],
        "neighbor_summary": "6 of 20 similar parts failed.",
        "citations": [{"kind": "column", "value": "L3_S32_F3850"}, {"kind": "part", "value": "1001"}],
    }
    return base | overrides


def check(d, seen_columns=frozenset({"L3_S32_F3850", "L1_S24_F1846"}), seen_parts=frozenset({42, 1001})):
    return citations.validate_disposition(d, SCHEMA, set(seen_columns), set(seen_parts), RECORD)


def test_grounded_disposition_is_valid():
    result = check(disposition())
    assert result.valid, result.errors


def test_fabricated_column_in_citation_rejected():
    result = check(disposition(citations=[{"kind": "column", "value": "L9_S99_F9999"}]))
    assert not result.valid and any("L9_S99_F9999" in e for e in result.errors)


def test_fabricated_column_in_free_text_rejected():
    result = check(disposition(reason="L3_S32_F3850 was high and L3_S33_F3999 drifted."))
    assert not result.valid and any("L3_S33_F3999" in e for e in result.errors)


def test_real_column_not_returned_by_tools_rejected():
    result = check(disposition(citations=[{"kind": "column", "value": "L3_S33_F3857"}]))
    assert not result.valid and any("not returned" in e for e in result.errors)


def test_unseen_part_id_rejected():
    result = check(disposition(citations=[{"kind": "part", "value": "777"}]))
    assert not result.valid and any("777" in e for e in result.errors)


def test_out_of_range_values_must_match_the_record():
    wrong = [{"column": "L3_S32_F3850", "value": 0.9, "low": -0.2, "high": 0.2}]
    assert not check(disposition(out_of_range=wrong)).valid
    invented = [{"column": "L1_S24_F1846", "value": 3.0, "low": 0.0, "high": 1.0}]
    assert not check(disposition(out_of_range=invented)).valid


def test_invalid_recommendation_rejected():
    assert not check(disposition(recommendation="rework")).valid
