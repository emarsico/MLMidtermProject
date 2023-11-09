def test(word):
    if (not char.isalpha() for char in word):
        return 1
    else:
        return 0


if __name__ == "__main__":
    print(test("ABC"))
