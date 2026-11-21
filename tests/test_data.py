import pytest

from stopret import data
from stopret.data import download, load, sha256_of, verify


@pytest.fixture
def fake_source(tmp_path, monkeypatch):
    """A local folder standing in for the GitHub repository, with its own hash table."""
    source = tmp_path / "source"
    source.mkdir()
    files = {name: f"content of {name}".encode() for name in data.SHA256}
    table = {}
    for name, content in files.items():
        (source / name).write_bytes(content)
        table[name] = sha256_of(source / name)
    monkeypatch.setattr(data, "BASE_URL", source.as_uri())
    monkeypatch.setattr(data, "SHA256", table)
    return source


def test_download_verifies_and_is_idempotent(tmp_path, fake_source, capsys):
    target = tmp_path / "raw"
    assert len(verify(target)) == 6                      # nothing there yet
    download(target)
    assert verify(target) == [] and not list(target.glob("*.part"))
    capsys.readouterr()
    download(target)                                     # everything valid: nothing is fetched again
    assert "downloading" not in capsys.readouterr().out


def test_corrupt_file_is_detected_and_replaced(tmp_path, fake_source, capsys):
    target = tmp_path / "raw"
    download(target)
    (target / "musique.json").write_bytes(b"truncated")
    assert verify(target) == ["musique.json"]
    capsys.readouterr()
    download(target)
    assert capsys.readouterr().out.count("downloading") == 1 and verify(target) == []


def test_hash_mismatch_never_leaves_a_bad_file_behind(tmp_path, fake_source):
    (fake_source / "hotpotqa.json").write_bytes(b"the upstream file changed")
    target = tmp_path / "raw"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download(target)
    assert not (target / "hotpotqa.json").exists() and not list(target.glob("*.part"))


def test_pinned_hashes_cover_every_dataset_file():
    expected = {f"{n}{s}.json" for n in data.DATASETS for s in ("", "_corpus")}
    assert set(data.SHA256) == expected
    assert all(len(h) == 64 for h in data.SHA256.values())
    assert len(data.DATA_COMMIT) == 40 and data.DATA_COMMIT in data.BASE_URL


def test_load_explains_how_to_get_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m stopret download"):
        load("hotpotqa", tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        load("nope", tmp_path)
