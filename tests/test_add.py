from capfinder.add import add


def test_add_function() -> None:
    result = add(1, 2, 3)
    assert result == 6, "Incorrect sum"
    result = add(1, 3, 3)
    assert result == 7, "Incorrect sum"
    # more test for 100% coverage
    result = add(1, 3, 5)
    assert result == 9
    result = add(-2, 3, 5)
    assert result == 6


if __name__ == "__main__":
    test_add_function()
