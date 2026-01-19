"""Suggestion ranking with novelty and diversity filtering."""


class SuggestionRanker:
    """Ranks and filters suggestions for novelty and diversity.

    Ensures follow-up questions don't repeat previous suggestions
    and maintains diversity among selected questions.

    Example:
        >>> ranker = SuggestionRanker(similarity_threshold=0.7)
        >>> suggestions = ["Would you like details?", "Want me to explain more?"]
        >>> previous = ["Would you like more details?"]
        >>> ranker.rank(suggestions, previous)
        ['Want me to explain more?']
    """

    def __init__(
        self,
        *,
        penalize_similarity: bool = True,
        similarity_threshold: float = 0.7,
    ) -> None:
        """Initialize the ranker.

        Args:
            penalize_similarity: Whether to filter similar suggestions.
            similarity_threshold: Word overlap ratio above which to filter.
        """
        self.penalize_similarity = penalize_similarity
        self.similarity_threshold = similarity_threshold

    def rank(
        self,
        suggestions: list[str],
        previous_followups: list[str] | None = None,
    ) -> list[str]:
        """Rank suggestions, filtering out similar ones.

        Args:
            suggestions: Candidate suggestions to rank.
            previous_followups: Previous follow-ups to avoid repeating.

        Returns:
            Filtered and ranked list of suggestions.
        """
        if not suggestions:
            return []

        previous = previous_followups or []
        ranked: list[str] = []

        for suggestion in suggestions:
            # Skip if too similar to previous follow-ups
            if (
                self.penalize_similarity
                and previous
                and self._is_similar_to_any(suggestion, previous)
            ):
                continue

            # Skip if too similar to already-ranked suggestions
            if self._is_similar_to_any(suggestion, ranked):
                continue

            ranked.append(suggestion)

        return ranked

    def _is_similar_to_any(self, text: str, others: list[str]) -> bool:
        """Check if text is similar to any in the list."""
        if not others:
            return False

        text_lower = text.lower()
        text_words = self._extract_meaningful_words(text_lower)

        for other in others:
            other_lower = other.lower()

            # Substring containment check
            if text_lower in other_lower or other_lower in text_lower:
                return True

            # Word overlap check
            other_words = self._extract_meaningful_words(other_lower)
            if not text_words or not other_words:
                continue

            overlap = len(text_words & other_words)
            min_len = min(len(text_words), len(other_words))

            if min_len > 0 and overlap / min_len > self.similarity_threshold:
                return True

        return False

    def _extract_meaningful_words(self, text: str) -> set[str]:
        """Extract meaningful words, filtering out common stop words and starters."""
        stop_words = {
            "would",
            "you",
            "like",
            "me",
            "to",
            "want",
            "should",
            "i",
            "the",
            "a",
            "an",
            "is",
            "are",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "can",
            "could",
            "may",
            "might",
            "must",
            "shall",
            "this",
            "that",
            "these",
            "those",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
            "interested",
        }
        words = set(text.replace("?", "").replace(".", "").replace(",", "").split())
        return words - stop_words
