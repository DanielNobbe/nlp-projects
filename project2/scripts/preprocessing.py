from pathlib import Path


def remove_closing_bracket(word):
    i = word.index(")")
    return word[:i]


def preprocess_line(line):
    words = line.strip().split(" ")
    new_words = [
        remove_closing_bracket(word) for word in words if not word.startswith("(")
    ]
    # Remove the . at the end of the sentence
    new_words = new_words[:-1]
    preprocessed = " ".join(new_words)
    return preprocessed


def preprocess_lines(lines):
    preprocessed = [preprocess_line(line) for line in lines]
    return preprocessed


if __name__ == "__main__":

    data_path = Path("../Data/Dataset")
    training_set_path = data_path / "train"

    with open(training_set_path, "r") as train_file:
        train_text = train_file.readlines()
