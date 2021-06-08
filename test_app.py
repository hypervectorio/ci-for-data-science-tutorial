import hypervector
import pytest

from app import get_prediction

hypervector.API_KEY = "32I8kjJNTr6i6x0U7ir2Y_tkm1r2RoJnrW962kbOHbY4Jw8"


@pytest.fixture
def hypervector_fixture():
    definition = hypervector.Definition.get("32b74bf4-06b8-4c14-a5d2-4c6db4d9fb2c")
    ensemble = definition.ensembles[0]
    hypervectors = ensemble.hypervectors()
    benchmark = ensemble.benchmarks[0]
    return hypervectors, benchmark


def test_single_prediction():
    test_case = [0, 0, 0, 0]
    result = get_prediction(test_case)['prediction']
    assert result == [1]


def test_bulk_prediction(hypervector_fixture):
    hypervectors, benchmark = hypervector_fixture
    results = get_prediction(hypervectors)['prediction']
    assertion = benchmark.assert_equal(results)

    assert assertion['asserted'] is True
