"""
Tests for the star pattern fingerprinting module.
"""

import numpy as np
import pytest

from waygraph.fingerprint.star_pattern import (
    ApproachArm,
    StarPattern,
    VECTOR_DIM,
)
from waygraph.fingerprint.matching import (
    StarPatternMatcher,
    star_distance,
)


class TestApproachArm:
    """Tests for the ApproachArm dataclass."""

    def test_default_construction(self):
        arm = ApproachArm()
        assert arm.angle_deg == 0.0
        assert arm.road_length_m == 0.0
        assert arm.road_type == ""
        assert arm.num_lanes == 1
        assert arm.neighbor_type == "none"
        assert arm.neighbor_degree == 0
        assert arm.neighbor_has_signal is False

    def test_custom_construction(self):
        arm = ApproachArm(
            angle_deg=90.0,
            road_length_m=200.0,
            road_type="primary",
            num_lanes=3,
            neighbor_type="cross",
            neighbor_degree=8,
            neighbor_has_signal=True,
        )
        assert arm.angle_deg == 90.0
        assert arm.road_length_m == 200.0
        assert arm.road_type == "primary"
        assert arm.num_lanes == 3
        assert arm.neighbor_type == "cross"
        assert arm.neighbor_degree == 8
        assert arm.neighbor_has_signal is True


class TestStarPattern:
    """Tests for the StarPattern dataclass."""

    def _make_cross_pattern(self):
        """Create a standard 4-way cross intersection pattern."""
        return StarPattern(
            id="test_cross",
            center_type="cross",
            center_approaches=4,
            center_has_signal=True,
            center_has_stop=False,
            center_has_crosswalk=True,
            arms=[
                ApproachArm(angle_deg=0, road_length_m=150, road_type="primary", num_lanes=2),
                ApproachArm(angle_deg=90, road_length_m=200, road_type="secondary", num_lanes=2),
                ApproachArm(angle_deg=180, road_length_m=150, road_type="primary", num_lanes=2),
                ApproachArm(angle_deg=270, road_length_m=120, road_type="tertiary", num_lanes=1),
            ],
        )

    def test_to_vector_shape(self):
        star = self._make_cross_pattern()
        vec = star.to_vector()
        assert vec.shape == (VECTOR_DIM,)
        assert vec.dtype == np.float64

    def test_to_vector_deterministic(self):
        star = self._make_cross_pattern()
        vec1 = star.to_vector()
        vec2 = star.to_vector()
        np.testing.assert_array_equal(vec1, vec2)

    def test_empty_pattern_vector(self):
        star = StarPattern()
        vec = star.to_vector()
        assert vec.shape == (VECTOR_DIM,)
        # Center type "none" = 0, no arms, everything zero
        assert vec[0] == 0.0
        assert vec[1] == 0.0

    def test_to_vector_normalized_range(self):
        star = self._make_cross_pattern()
        vec = star.to_vector()
        # All features should be in [0, 1] range
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_arms_sorted_by_angle(self):
        """Verify that arms are sorted by angle in the feature vector."""
        star = StarPattern(
            center_type="T",
            center_approaches=3,
            arms=[
                ApproachArm(angle_deg=270, road_length_m=100),
                ApproachArm(angle_deg=90, road_length_m=200),
                ApproachArm(angle_deg=0, road_length_m=150),
            ],
        )
        vec = star.to_vector()
        # First arm should be angle_deg=0 -> 0/360 = 0.0
        assert vec[6] == pytest.approx(0.0 / 360.0)
        # Second arm should be angle_deg=90 -> 90/360 = 0.25
        assert vec[13] == pytest.approx(90.0 / 360.0)

    def test_serialization_roundtrip(self):
        star = self._make_cross_pattern()
        data = star.to_dict()
        recovered = StarPattern.from_dict(data)

        assert recovered.id == star.id
        assert recovered.center_type == star.center_type
        assert recovered.center_approaches == star.center_approaches
        assert len(recovered.arms) == len(star.arms)

        vec_orig = star.to_vector()
        vec_recovered = recovered.to_vector()
        np.testing.assert_array_almost_equal(vec_orig, vec_recovered)

    def test_max_arms_padding(self):
        """Patterns with fewer than 6 arms should be zero-padded."""
        star = StarPattern(
            center_type="T",
            center_approaches=3,
            arms=[
                ApproachArm(angle_deg=0, road_length_m=100),
                ApproachArm(angle_deg=120, road_length_m=100),
                ApproachArm(angle_deg=240, road_length_m=100),
            ],
        )
        vec = star.to_vector()
        # Arms 3, 4, 5 should be all zeros (indices 27..48)
        assert np.all(vec[27:48] == 0.0)


class TestStarDistance:
    """Tests for the star_distance function."""

    def test_identical_vectors(self):
        vec = np.random.rand(VECTOR_DIM)
        assert star_distance(vec, vec) == pytest.approx(0.0)

    def test_symmetry(self):
        v1 = np.random.rand(VECTOR_DIM)
        v2 = np.random.rand(VECTOR_DIM)
        assert star_distance(v1, v2) == pytest.approx(star_distance(v2, v1))

    def test_positive_distance(self):
        v1 = np.zeros(VECTOR_DIM)
        v2 = np.ones(VECTOR_DIM)
        assert star_distance(v1, v2) > 0

    def test_custom_weights(self):
        v1 = np.zeros(VECTOR_DIM)
        v2 = np.ones(VECTOR_DIM)
        uniform_w = np.ones(VECTOR_DIM)
        heavy_w = np.ones(VECTOR_DIM) * 10.0

        d_uniform = star_distance(v1, v2, weights=uniform_w)
        d_heavy = star_distance(v1, v2, weights=heavy_w)
        assert d_heavy > d_uniform


class TestStarPatternMatcher:
    """Tests for the StarPatternMatcher."""

    def _build_test_database(self, n=100, seed=42):
        rng = np.random.RandomState(seed)
        patterns = []
        for i in range(n):
            n_arms = rng.choice([3, 4, 5])
            itype = {3: "T", 4: "cross", 5: "multi"}[n_arms]
            arms = [
                ApproachArm(
                    angle_deg=(j * 360.0 / n_arms + rng.normal(0, 5)) % 360,
                    road_length_m=50 + rng.exponential(80),
                    road_type="secondary",
                    num_lanes=rng.choice([1, 2, 3]),
                    neighbor_type=rng.choice(["T", "cross"]),
                    neighbor_degree=rng.choice([4, 6, 8]),
                )
                for j in range(n_arms)
            ]
            patterns.append(StarPattern(
                id=f"osm_{i}",
                center_type=itype,
                center_approaches=n_arms,
                center_has_signal=rng.random() > 0.5,
                arms=arms,
            ))
        return patterns

    def test_build_database(self):
        patterns = self._build_test_database()
        matcher = StarPatternMatcher()
        matcher.build_database(patterns)
        assert matcher.database_size == len(patterns)

    def test_match_returns_results(self):
        patterns = self._build_test_database()
        matcher = StarPatternMatcher()
        matcher.build_database(patterns)

        query = patterns[0]  # Perfect match should rank #1
        results = matcher.match(query, top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_perfect_match_rank_one(self):
        patterns = self._build_test_database()
        matcher = StarPatternMatcher()
        matcher.build_database(patterns)

        query = patterns[10]
        results = matcher.match(query, top_k=5)
        # The exact same pattern should be the top match
        assert results[0][0] == query.id
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_match_without_database_raises(self):
        matcher = StarPatternMatcher()
        query = StarPattern(center_type="cross", center_approaches=4)
        with pytest.raises(RuntimeError):
            matcher.match(query)

    def test_match_batch(self):
        patterns = self._build_test_database()
        matcher = StarPatternMatcher()
        matcher.build_database(patterns)

        queries = patterns[:5]
        results = matcher.match_batch(queries, top_k=3, verbose=False)
        assert len(results) == 5
        for qid, matches in results.items():
            assert len(matches) <= 3

    def test_evaluate(self):
        patterns = self._build_test_database(n=50)
        matcher = StarPatternMatcher()
        matcher.build_database(patterns)

        # Use database patterns as queries (perfect matches)
        queries = patterns[:20]
        gt = {p.id: p.id for p in queries}

        metrics = matcher.evaluate(queries, gt)
        assert metrics["n"] == 20
        assert metrics["top1"] == 100.0  # Perfect matches
        assert metrics["mrr"] == pytest.approx(1.0)
