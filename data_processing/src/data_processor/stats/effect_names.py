
EFFECT_NAMES = {
    1: "Main effect",
    2: "Two-way interaction",
    3: "Three-way interaction",
    4: "Four-way interaction"
}

_effect_names = {
    "host": "platform",
    "tool": "utility",
}


def get_effect_name(effect: str) -> str:
    effect = effect.replace("×", "x")
    names = effect.split(" x ")
    result_names = []
    for name in names:
        new_name = _effect_names.get(name, name)
        result_names.append(new_name)
    result = " x ".join(result_names)
    return result
