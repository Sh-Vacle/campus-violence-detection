from violence_detection.paths import safe_stem


def test_safe_stem_keeps_normal_name():
    assert safe_stem("video_01.mp4") == "video_01"


def test_safe_stem_replaces_spaces_and_symbols():
    assert safe_stem("a b#c.mp4") == "a_b_c"
