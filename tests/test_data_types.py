"""Unit tests for data types"""

from themis_ml.datasets import data_types


def create_variables():
    return [
        data_types.Variable(
            "var1", data_types.VariableType.BINARY),
        data_types.Variable(
            "var2", data_types.VariableType.NON_ORDERED_CATEGORICAL),
        data_types.Variable(
            "var3", data_types.VariableType.ORDERED_CATEGORICAL),
        data_types.Variable(
            "var4", data_types.VariableType.NUMERIC),
        data_types.Variable(
            "target", data_types.VariableType.BINARY, is_target=True)
    ]


def test_variable_map():
    variables = create_variables()
    variable_map = data_types.VariableMap(variables)
    assert variable_map.variable_map.items() == [
        (v.name, v) for v in variables]
    assert variable_map.targets == ["target"]
