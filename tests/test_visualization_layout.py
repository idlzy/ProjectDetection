from tools.visualize_test import safe_group_name, sequential_indices


def test_group_name_preserves_chinese_and_replaces_path_characters():
    assert safe_group_name("大类/D105__2026-01-01") == "大类_D105__2026-01-01"


def test_sequential_indices_support_resume_and_limit():
    assert sequential_indices(10, 3, 4) == [3, 4, 5, 6]
    assert sequential_indices(5, 2, None) == [2, 3, 4]
