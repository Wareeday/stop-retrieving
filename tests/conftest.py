import pytest

from stopret.data import Dataset, Example


@pytest.fixture
def tiny_dataset() -> Dataset:
    """A hand-made two-hop corpus: the answer to each question needs two passages."""
    titles = ["Alice Smith", "Brightwater", "Bob Jones", "Cedar Falls", "Dana Lee", "Eastport"]
    texts = [
        "Alice Smith is a painter who was born in Brightwater.",
        "Brightwater is a small town with a population of 3,000 people.",
        "Bob Jones is a sculptor who lives in Cedar Falls.",
        "Cedar Falls is a city on the Redstone river.",
        "Dana Lee is a poet from Eastport.",
        "Eastport is a harbour town known for its lighthouse.",
    ]
    examples = [
        Example("q1", "What is the population of the town where Alice Smith was born?", "3,000", [0, 1]),
        Example("q2", "Which river runs through the city where Bob Jones lives?", "Redstone", [2, 3]),
    ]
    return Dataset("tiny", titles, texts, examples)
