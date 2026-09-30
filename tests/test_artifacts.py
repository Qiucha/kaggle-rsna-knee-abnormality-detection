import os

from contrastive_learning import artifacts


def test_write_atomic_flushes_data_before_the_rename_and_the_directory_after(tmp_path, monkeypatch):
    events = []
    real_replace = type(tmp_path).replace
    monkeypatch.setattr(os, "fsync", lambda fd: events.append(("fsync", os.path.isdir(f"/proc/self/fd/{fd}"))))
    monkeypatch.setattr(type(tmp_path), "replace", lambda self, target: events.append(("rename",))
                        or real_replace(self, target))

    path = artifacts.write_atomic(tmp_path / "out.bin", lambda t: t.write_bytes(b"payload"))

    assert path.read_bytes() == b"payload"
    assert events == [("fsync", False), ("rename",), ("fsync", True)]
    assert not list(tmp_path.glob(".*.tmp"))
