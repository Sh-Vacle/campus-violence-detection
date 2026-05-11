from pathlib import Path

from violence_detection.data import count_images


def test_count_images(tmp_path: Path):
    (tmp_path / "Violence").mkdir()
    (tmp_path / "NonViolence").mkdir()
    (tmp_path / "Violence" / "a.jpg").write_bytes(b"x")
    (tmp_path / "Violence" / "b.txt").write_text("not image")
    (tmp_path / "NonViolence" / "c.png").write_bytes(b"x")

    assert count_images(tmp_path) == {"NonViolence": 1, "Violence": 1}
