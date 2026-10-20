import itertools

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


def build_effect_names(factors: list[str], factor_axes: range):
    effect_subsets = [
        subset
        for order in range(1, len(factors) + 1)
        for subset in itertools.combinations(factor_axes, order)
    ]

    contrast_labels = {
        factor: f"C({factor}, Sum)"
        for factor in factors
    }
    effect_names = [
        ":".join(
            contrast_labels[factors[axis]]
            for axis in subset
        )
        for subset in effect_subsets
    ]
    return effect_names, effect_subsets
