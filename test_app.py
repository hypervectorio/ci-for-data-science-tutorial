import hypervector
import pytest

from app import get_prediction

hypervector.API_KEY = "YOUR_API_KEY"


@pytest.fixture
def hypervector_fixture():
    definition = hypervector.Definition.get("YOUR_DEFINITION_UUID")
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
