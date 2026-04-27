"""Tests for graph.py module."""

import matplotlib
import networkx as nx
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
from src.graph import build_sfm
from src.visualisation.graph import visualize_sfm


class TestBuildSFM:
    """Test suite for build_sfm function."""

    def test_basic_sfm_structure(self):
        """Test basic SFM graph structure with all components."""
        sfm = build_sfm(
            sensitive_attr="Gender",
            outcome_attr="Income",
            confounder_attrs=["Education"],
            mediator_attrs=["Occupation"],
        )

        assert isinstance(sfm, nx.DiGraph)
        assert "Gender" in sfm.nodes()
        assert "Income" in sfm.nodes()
        assert "Education" in sfm.nodes()
        assert "Occupation" in sfm.nodes()

    def test_sensitive_to_outcome_edge(self):
        """Test that sensitive attribute connects to outcome."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=[],
            mediator_attrs=[],
        )

        assert sfm.has_edge("X", "Y")

    def test_confounder_edges(self):
        """Test that confounders connect to both sensitive and outcome."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=["Z1", "Z2"],
            mediator_attrs=[],
        )

        # Confounders should point to sensitive attribute
        assert sfm.has_edge("Z1", "X")
        assert sfm.has_edge("Z2", "X")

        # Confounders should point to outcome
        assert sfm.has_edge("Z1", "Y")
        assert sfm.has_edge("Z2", "Y")

    def test_mediator_edges(self):
        """Test that mediators connect correctly."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=[],
            mediator_attrs=["W1", "W2"],
        )

        # Sensitive should point to mediators
        assert sfm.has_edge("X", "W1")
        assert sfm.has_edge("X", "W2")

        # Mediators should point to outcome
        assert sfm.has_edge("W1", "Y")
        assert sfm.has_edge("W2", "Y")

    def test_confounder_to_mediator_edges(self):
        """Test that confounders connect to mediators."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=["Z"],
            mediator_attrs=["W"],
        )

        assert sfm.has_edge("Z", "W")

    def test_node_types(self):
        """Test that node types are correctly assigned."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=["Z"],
            mediator_attrs=["W"],
        )

        assert sfm.nodes["X"]["type"] == "sensitive"
        assert sfm.nodes["Y"]["type"] == "outcome"
        assert sfm.nodes["Z"]["type"] == "confounder"
        assert sfm.nodes["W"]["type"] == "mediator"

    def test_latent_variables(self):
        """Test that latent variables are added correctly."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=[],
            mediator_attrs=["W"],
            latents=[("U", ["X", "W"])],
        )

        assert "U" in sfm.nodes()
        assert sfm.nodes["U"]["type"] == "latent"
        assert sfm.has_edge("U", "X")
        assert sfm.has_edge("U", "W")

    def test_empty_confounders_and_mediators(self):
        """Test SFM with no confounders or mediators."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=[],
            mediator_attrs=[],
        )

        assert len(sfm.nodes()) == 2
        assert len(sfm.edges()) == 1
        assert sfm.has_edge("X", "Y")

    def test_multiple_confounders_and_mediators(self):
        """Test SFM with multiple confounders and mediators."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=["Z1", "Z2", "Z3"],
            mediator_attrs=["W1", "W2"],
        )

        # Total nodes: 1 (X) + 1 (Y) + 3 (Z) + 2 (W) = 7
        assert len(sfm.nodes()) == 7

        # Check all confounder-to-mediator edges exist
        for z in ["Z1", "Z2", "Z3"]:
            for w in ["W1", "W2"]:
                assert sfm.has_edge(z, w)

    def test_invalid_sensitive_attr_type(self):
        """Test that non-string sensitive attribute raises error."""
        with pytest.raises(ValueError, match="Sensitive attribute must be a string"):
            build_sfm(
                sensitive_attr=123,
                outcome_attr="Y",
                confounder_attrs=[],
                mediator_attrs=[],
            )

    def test_invalid_outcome_attr_type(self):
        """Test that non-string outcome attribute raises error."""
        with pytest.raises(ValueError, match="Outcome attribute must be a string"):
            build_sfm(
                sensitive_attr="X",
                outcome_attr=["Y"],
                confounder_attrs=[],
                mediator_attrs=[],
            )

    def test_invalid_confounder_attrs_type(self):
        """Test that non-list confounder attributes raise error."""
        with pytest.raises(ValueError, match="Confounder attributes must be a list"):
            build_sfm(
                sensitive_attr="X",
                outcome_attr="Y",
                confounder_attrs="Z",
                mediator_attrs=[],
            )

    def test_invalid_mediator_attrs_type(self):
        """Test that non-list mediator attributes raise error."""
        with pytest.raises(ValueError, match="Mediator attributes must be a list"):
            build_sfm(
                sensitive_attr="X",
                outcome_attr="Y",
                confounder_attrs=[],
                mediator_attrs="W",
            )

    def test_confounder_attrs_non_string_elements(self):
        """Test that non-string elements in confounder list raise error."""
        with pytest.raises(
            ValueError, match="Confounder attributes must be a list of strings"
        ):
            build_sfm(
                sensitive_attr="X",
                outcome_attr="Y",
                confounder_attrs=["Z1", 123],
                mediator_attrs=[],
            )


class TestVisualizeSFM:
    """Test suite for visualize_sfm function."""

    @pytest.fixture(autouse=True)
    def mock_plt_show(self, monkeypatch):
        """Mock plt.show() to prevent blocking during tests."""
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)

    def test_visualize_sfm_returns_pgm(self):
        """Test that visualize_sfm returns a PGM object."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=["Z"],
            mediator_attrs=["W"],
        )

        pgm = visualize_sfm(sfm, scale_factor=1.0)

        from daft import PGM

        assert isinstance(pgm, PGM)

    def test_visualize_sfm_with_custom_scale(self):
        """Test visualize_sfm with custom scale factor."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=[],
            mediator_attrs=[],
        )

        pgm = visualize_sfm(sfm, scale_factor=2.5)

        from daft import PGM

        assert isinstance(pgm, PGM)

    def test_visualize_sfm_skips_latents(self):
        """Test that latent variables are not visualized."""
        sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=[],
            mediator_attrs=["W"],
            latents=[("U", ["X", "W"])],
        )

        # This should not raise an error even with latents
        pgm = visualize_sfm(sfm)

        from daft import PGM

        assert isinstance(pgm, PGM)
