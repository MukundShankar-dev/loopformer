import gzip
from scripts.eval.evidence_inventory import inventory, table_profile


def test_reserved_data_is_not_read_and_orphan_weights_are_visible(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.eval.evidence_inventory.git_output", lambda *args: "test")
    protected = ["data/pointer/seed-17/test.jsonl", "data/pointer/seed-29/validation.jsonl"]
    for name in protected:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("This is deliberately not parseable JSON.")
    opened = tmp_path / "data/pointer/seed-61-independent/test.jsonl"
    opened.parent.mkdir(parents=True)
    opened.write_text('{"task_depth": 12}\n')
    orphan = tmp_path / "models/stage1_pointer/old/step-000001/adapter_model.pt"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"Not a tensor; inventory must never deserialize it")
    result = inventory(tmp_path, "test")
    indexed = {record["path"]: record for record in result["files"]}
    for name in protected:
        assert indexed[name]["content_inspected"] is False
        assert "records" not in indexed[name] and "sha256" not in indexed[name]
    assert indexed[str(opened.relative_to(tmp_path))]["records"]["rows"] == 1
    assert result["weight_directories_without_config"] == [str(orphan.parent.relative_to(tmp_path))]


def test_csv_profiles_count_records_not_physical_lines(tmp_path):
    payload = 'graph_index,depth,note\n0,12,"two\nlines"\n1,256,plain\n'
    plain = tmp_path / "records.csv"
    compressed = tmp_path / "records.csv.gz"
    plain.write_text(payload)
    compressed.write_bytes(gzip.compress(payload.encode()))
    expected = {"columns": ["graph_index", "depth", "note"], "rows": 2,
                "numeric_spans": {"depth": [12.0, 256.0]},
                "distinct_ids": {"graph_index": 2}}
    assert table_profile(plain) == table_profile(compressed) == expected
