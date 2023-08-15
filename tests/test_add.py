import pytest
from capfinder.add import add


def test_add_function()-> None:
    result = add(1, 2, 3)
    assert result == 6, "Incorrect sum"


if __name__ == "__main__":
    test_add_function()
