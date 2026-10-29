from .effect_names import get_effect_name


def test_get_effect_name_single_unchanged():
    actual_name = get_effect_name("dataset")

    assert actual_name == "dataset"


def test_get_effect_name_single():
    actual_name = get_effect_name("host")

    assert actual_name == "platform"


def test_get_effect_name_two_way():
    actual_name = get_effect_name("host x tool")

    assert actual_name == "platform x utility"


def test_get_effect_name_two_way_pandas():
    actual_name = get_effect_name("tool × strength")

    assert actual_name == "utility x level"
