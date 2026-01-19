"""Tests for suggestion ranker."""

from compass.ranker import SuggestionRanker


class TestSuggestionRanker:
    """Tests for SuggestionRanker."""

    def test_returns_empty_for_empty_input(self):
        ranker = SuggestionRanker()
        assert ranker.rank([]) == []

    def test_returns_all_unique_suggestions(self):
        ranker = SuggestionRanker()
        suggestions = [
            "Would you like more details about Python?",
            "Interested in learning about JavaScript?",
            "Should I explain the differences?",
        ]
        result = ranker.rank(suggestions)
        assert len(result) == 3
        assert result == suggestions

    def test_filters_similar_to_previous(self):
        ranker = SuggestionRanker()
        suggestions = ["Would you like more details about Python?"]
        previous = ["Would you like more details about Python programming?"]
        result = ranker.rank(suggestions, previous)
        assert len(result) == 0

    def test_filters_duplicates_in_candidates(self):
        ranker = SuggestionRanker()
        suggestions = [
            "Would you like more details?",
            "Would you like more details?",  # Exact duplicate
        ]
        result = ranker.rank(suggestions)
        assert len(result) == 1

    def test_filters_substring_matches(self):
        ranker = SuggestionRanker()
        suggestions = [
            "Would you like details?",
            "Would you like more details about this?",  # Contains first
        ]
        result = ranker.rank(suggestions)
        assert len(result) == 1
        assert result[0] == "Would you like details?"

    def test_respects_similarity_threshold(self):
        ranker = SuggestionRanker(similarity_threshold=0.9)
        suggestions = [
            "Would you like to explore machine learning?",
            "Would you like to explore deep learning?",  # Different topic
        ]
        result = ranker.rank(suggestions)
        assert len(result) == 2

    def test_disabled_similarity_check(self):
        ranker = SuggestionRanker(penalize_similarity=False)
        suggestions = ["Would you like more details?"]
        previous = ["Would you like more details?"]
        result = ranker.rank(suggestions, previous)
        assert len(result) == 1

    def test_meaningful_word_extraction(self):
        """Test that stop words are properly filtered."""
        ranker = SuggestionRanker()

        # These should be considered different because meaningful words differ
        suggestions = [
            "Would you like to explore Python?",
            "Should I explain JavaScript?",
        ]
        result = ranker.rank(suggestions)
        assert len(result) == 2

    def test_handles_none_previous(self):
        ranker = SuggestionRanker()
        suggestions = ["Would you like more details?"]
        result = ranker.rank(suggestions, None)
        assert len(result) == 1

    def test_preserves_order(self):
        ranker = SuggestionRanker()
        suggestions = ["First question?", "Second question?", "Third question?"]
        result = ranker.rank(suggestions)
        assert result == suggestions
