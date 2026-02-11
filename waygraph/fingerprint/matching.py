"""
Star Pattern Matching
======================

Match WOMD star pattern fingerprints against an OSM star pattern database
to identify the real-world location of each WOMD scenario.

The matching pipeline:
    1. **Coarse filter**: Reject candidates with incompatible intersection
       types or approach counts (typically eliminates 70-80% of candidates).
    2. **Feature distance**: Compute weighted Euclidean distance between
       48D star pattern vectors.
    3. **Ranking**: Return top-k matches sorted by similarity score.

Performance on synthetic benchmarks:
    - Top-1 accuracy: ~90% (clean noise), ~70% (realistic noise)
    - MRR: ~0.92 (clean), ~0.78 (realistic)
    - Matching 250 patterns against 5000+ OSM intersections: <2 seconds

Example::

    from waygraph.fingerprint import StarPatternMatcher, StarPattern

    # Build matcher with an OSM database
    matcher = StarPatternMatcher()
    matcher.build_database(osm_star_patterns)

    # Match a WOMD pattern
    results = matcher.match(womd_pattern, top_k=10)
    for osm_id, score in results[:3]:
        print(f"  OSM {osm_id}: score={score:.3f}")
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from waygraph.fingerprint.star_pattern import StarPattern


# Default feature weights for the 48D star pattern vector
# Center features get higher weight (more discriminative)
DEFAULT_WEIGHTS = np.array(
    [4.0, 3.0, 2.0, 1.0, 1.0, 2.0]  # center
    + [3.0, 4.0, 2.0, 1.5, 3.0, 2.0, 1.5] * 6,  # 6 arms
    dtype=np.float64,
)

# Compatible intersection types for coarse filtering
COMPATIBLE_TYPES: Dict[str, Set[str]] = {
    "cross": {"cross", "multi"},
    "T": {"T", "Y"},
    "Y": {"Y", "T"},
    "multi": {"multi", "cross"},
    "roundabout": {"roundabout"},
    "merge": {"merge", "none"},
    "none": {"none", "merge"},
}


def star_distance(
    v1: np.ndarray,
    v2: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute weighted Euclidean distance between two star pattern vectors.

    Args:
        v1: First star pattern vector, shape (48,).
        v2: Second star pattern vector, shape (48,).
        weights: Optional per-dimension weights, shape (48,). If None,
            uses default weights that emphasize center features and
            arm angles/neighbor types.

    Returns:
        Weighted Euclidean distance (lower = more similar).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    diff = (v1 - v2) * weights
    return float(np.sqrt(np.sum(diff**2)))


class StarPatternMatcher:
    """Match star patterns against a database of reference patterns.

    The matcher pre-computes feature vectors for a database of OSM star
    patterns, then efficiently matches new query patterns using coarse
    filtering + weighted distance.

    Args:
        weights: Optional custom feature weights for distance computation.
        max_approach_diff: Maximum allowed difference in approach count
            for the coarse filter (default 1).
        score_normalization: Distance at which the similarity score equals 0.
            Distances below this produce scores in (0, 1].

    Example::

        matcher = StarPatternMatcher()
        matcher.build_database(osm_patterns)

        results = matcher.match(query_pattern, top_k=5)
        best_id, best_score = results[0]
    """

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        max_approach_diff: int = 1,
        score_normalization: float = 15.0,
    ):
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        self.max_approach_diff = max_approach_diff
        self.score_normalization = score_normalization

        # Database
        self._db_patterns: List[StarPattern] = []
        self._db_vectors: Optional[np.ndarray] = None

    def build_database(self, patterns: List[StarPattern]) -> None:
        """Build the reference database from a list of star patterns.

        Pre-computes and caches feature vectors for all patterns.

        Args:
            patterns: List of reference StarPattern objects (e.g., from OSM).
        """
        self._db_patterns = list(patterns)
        self._db_vectors = np.array(
            [p.to_vector() for p in patterns], dtype=np.float64
        )

    @property
    def database_size(self) -> int:
        """Number of patterns in the reference database."""
        return len(self._db_patterns)

    def match(
        self,
        query: StarPattern,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Match a single query pattern against the database.

        Args:
            query: The star pattern to match.
            top_k: Number of top matches to return.

        Returns:
            List of (pattern_id, similarity_score) tuples, sorted by
            descending score. Scores are in [0, 1] where 1 is perfect.

        Raises:
            RuntimeError: If the database has not been built.
        """
        if self._db_vectors is None:
            raise RuntimeError(
                "Database not built. Call build_database() first."
            )

        query_vec = query.to_vector()
        compat = COMPATIBLE_TYPES.get(
            query.center_type, {query.center_type}
        )

        scored: List[Tuple[str, float]] = []

        for j, db_pattern in enumerate(self._db_patterns):
            # Coarse filter: type compatibility
            if db_pattern.center_type not in compat:
                continue
            # Coarse filter: approach count
            if (
                abs(db_pattern.center_approaches - query.center_approaches)
                > self.max_approach_diff
            ):
                continue

            dist = star_distance(query_vec, self._db_vectors[j], self.weights)
            score = max(0.0, 1.0 - dist / self.score_normalization)
            scored.append((db_pattern.id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def match_batch(
        self,
        queries: List[StarPattern],
        top_k: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Match multiple query patterns against the database.

        Args:
            queries: List of query StarPattern objects.
            top_k: Number of top matches per query.
            verbose: Whether to print progress.

        Returns:
            Dictionary mapping query ID to list of (db_id, score) tuples.
        """
        results: Dict[str, List[Tuple[str, float]]] = {}
        t0 = time.time()

        for i, query in enumerate(queries):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Matching {i + 1}/{len(queries)}...")

            results[query.id] = self.match(query, top_k=top_k)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"  Matched {len(queries)} patterns in {elapsed:.1f}s"
            )
        return results

    def evaluate(
        self,
        queries: List[StarPattern],
        ground_truth: Dict[str, str],
        top_k_values: Tuple[int, ...] = (1, 5, 10),
    ) -> Dict[str, float]:
        """Evaluate matching accuracy against ground truth.

        Args:
            queries: Query star patterns.
            ground_truth: Mapping of query_id to true db_pattern_id.
            top_k_values: K values for top-k accuracy metrics.

        Returns:
            Dictionary with metrics: top1, top5, top10, mrr, median_rank.
        """
        if self._db_vectors is None:
            raise RuntimeError("Database not built.")

        rr_list: List[float] = []
        rank_list: List[int] = []
        top_k_counts = {k: 0 for k in top_k_values}

        for query in queries:
            true_id = ground_truth.get(query.id)
            if true_id is None:
                continue

            matches = self.match(query, top_k=max(top_k_values) * 20)

            found_rank = None
            for rank, (db_id, score) in enumerate(matches):
                if db_id == true_id:
                    found_rank = rank + 1
                    break

            if found_rank is not None:
                for k in top_k_values:
                    if found_rank <= k:
                        top_k_counts[k] += 1
                rr_list.append(1.0 / found_rank)
                rank_list.append(found_rank)
            else:
                rr_list.append(0.0)
                rank_list.append(max(top_k_values) * 20 + 1)

        n = len(rr_list)
        if n == 0:
            return {"n": 0}

        result = {"n": n}
        for k in top_k_values:
            result[f"top{k}"] = round(top_k_counts[k] / n * 100, 1)
        result["mrr"] = round(float(np.mean(rr_list)), 3)
        result["median_rank"] = int(np.median(rank_list))

        return result
