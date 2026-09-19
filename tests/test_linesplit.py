"""Tests of the verse reflow: capitals, function words and punctuation."""

from __future__ import annotations

from lyricsmith.utils.linesplit import relineate


def test_splits_on_capitalised_line_starts() -> None:
    """A capitalized word opens a new verse."""
    text = "les autres manquent de sel J'espère en vain allumer l'étincelle"
    out = relineate(text, min_words_to_split=4)
    lines = out.split("\n")
    assert lines[0] == "les autres manquent de sel"
    assert lines[1].startswith("J'espère")


def test_capitalised_pronoun_i_does_not_split() -> None:
    """The English pronoun "I" does not cut a verse in half."""
    text = "this is the moment I feel that I am alive tonight and free"
    out = relineate(text, min_words_to_split=4)
    assert not any(line.startswith("I ") for line in out.split("\n"))


def test_acronym_all_caps_does_not_split() -> None:
    """An all caps word, meaning an acronym, does not open a verse."""
    text = "je connais le vice de CRDN depuis toujours vraiment très bien"
    out = relineate(text, min_words_to_split=4)
    assert not any(line.startswith("CRDN") for line in out.split("\n"))


def test_lowercase_run_splits_on_function_words() -> None:
    """In a lowercase passage the cut lands on a function word."""
    text = (
        "i only exist in the moments you are talking to me "
        "if we can not be together then i will go back to sleep"
    )
    out = relineate(text, max_words=8, hard_max_words=14, min_words_to_split=6)
    lines = out.split("\n")
    assert len(lines) > 1
    assert any(line.startswith(("if", "then", "i ")) for line in lines[1:])


def test_strong_punctuation_splits() -> None:
    """Strong punctuation ends a verse."""
    text = "how am I nothing to you? tell me how am I nothing to you?"
    out = relineate(text, min_words_to_split=4)
    assert out.split("\n")[0].endswith("?")


def test_short_lines_are_preserved() -> None:
    """An already short line is left untouched."""
    text = "They say\nYou'll get used to it\nBut it never goes away"
    out = relineate(text, min_words_to_split=12)
    assert out == text


def test_disabled_via_min_threshold_keeps_single_line() -> None:
    """A text shorter than the threshold stays intact."""
    text = "just a short line here"
    assert relineate(text, min_words_to_split=12) == text


def test_empty_and_whitespace() -> None:
    """Empty or blank strings return an empty string without failing."""
    assert relineate("") == ""
    assert relineate("   \n  \n") == ""
