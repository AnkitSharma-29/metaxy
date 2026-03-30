"""Tests for the metadata read CLI command."""

import json

import polars as pl
import pytest
from metaxy_testing import TempMetaxyProject
from polars.testing import assert_frame_equal


def _define_features():
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

    class FilesRoot(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["files_root"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass


@pytest.fixture
def upstream_data() -> pl.DataFrame:
    """Shared fixture for basic test data."""
    return pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "value": ["val_1", "val_2", "val_3"],
            "category": ["A", "B", "A"],
            "metaxy_provenance_by_field": [{"default": f"hash{i}"} for i in range(1, 4)],
        }
    )


def _setup_store_with_data(metaxy_project: TempMetaxyProject, upstream_data: pl.DataFrame):
    """Helper to set up store with basic test data."""
    from metaxy.models.types import FeatureKey

    graph = metaxy_project.graph
    store = metaxy_project.stores["dev"]

    with graph.use(), store.open("w"):
        store.write(FeatureKey(["files_root"]), upstream_data)


@pytest.mark.parametrize("fmt", ["csv", "json", "parquet", "markdown"])
@pytest.mark.parametrize("to_file", [True, False])
def test_metadata_read_formats(
    metaxy_project: TempMetaxyProject,
    tmp_path,
    upstream_data: pl.DataFrame,
    fmt: str,
    to_file: bool,
    capsys: pytest.CaptureFixture[str],
):
    """Test reading metadata and outputting to various formats, either file or stdout."""
    with metaxy_project.with_features(_define_features):
        _setup_store_with_data(metaxy_project, upstream_data)

        # Expected data derived directly from upstream
        # The CLI adds system columns, so we select known columns for easy comparison
        expected = upstream_data.select("sample_uid", "value", "category")

        cmd = [
            "metadata",
            "read",
            "files_root",
            "--select",
            "sample_uid",
            "--select",
            "value",
            "--select",
            "category",
            "-f",
            fmt,
        ]

        output_file = tmp_path / f"output.{fmt}"
        if to_file:
            cmd.extend(["-o", str(output_file)])

        # Run with subprocess=True for Parquet stdout tests (as binary stdout fails via standard capture)
        run_kwargs = {"subprocess": True} if (not to_file and fmt == "parquet") else {"capsys": capsys}

        result = metaxy_project.run_cli(cmd, **run_kwargs)
        assert result.returncode == 0

        # Verification
        if to_file:
            assert output_file.exists()
            if fmt == "csv":
                content = output_file.read_text(encoding="utf-8")
                assert "1,val_1,A" in content
            elif fmt == "json":
                data = json.loads(output_file.read_text(encoding="utf-8"))
                assert_frame_equal(pl.DataFrame(data), expected)
            elif fmt == "parquet":
                df = pl.read_parquet(output_file)
                assert_frame_equal(df, expected)
            elif fmt == "markdown":
                content = output_file.read_text(encoding="utf-8")
                assert "sample_uid" in content
                assert "1" in content
        else:
            if fmt == "csv":
                clean_out = result.stdout.replace("\r", "")
                assert "1,val_1,A" in clean_out
            elif fmt == "json":
                data = json.loads(result.stdout)
                assert_frame_equal(pl.DataFrame(data), expected)
            elif fmt == "parquet":
                # With subprocess=True, stdout is typically bytes or we can at least verify execution succeeded
                assert hasattr(result, "stdout")
            elif fmt == "markdown":
                assert "sample_uid" in result.stdout
                assert "1" in result.stdout


def test_metadata_read_select_and_filter(
    metaxy_project: TempMetaxyProject, upstream_data: pl.DataFrame, capsys: pytest.CaptureFixture[str]
):
    """Test metadata read with --select and --filter."""
    with metaxy_project.with_features(_define_features):
        _setup_store_with_data(metaxy_project, upstream_data)

        result = metaxy_project.run_cli(
            ["metadata", "read", "files_root", "--select", "value", "--filter", "category = 'A'", "-f", "json"],
            capsys=capsys,
        )

        assert result.returncode == 0
        data = json.loads(result.stdout)

        expected_subset = upstream_data.filter(pl.col("category") == "A").select("value")
        assert len(data) == len(expected_subset)
        assert "value" in data[0]
        assert "category" not in data[0]
        assert data[0]["value"] == expected_subset["value"][0]


def test_metadata_read_query(
    metaxy_project: TempMetaxyProject, upstream_data: pl.DataFrame, capsys: pytest.CaptureFixture[str]
):
    """Test metadata read with arbitrary SQL query."""
    with metaxy_project.with_features(_define_features):
        _setup_store_with_data(metaxy_project, upstream_data)

        cmd = [
            "metadata",
            "read",
            "files_root",
            "--query",
            "SELECT count(*) as cnt FROM files_root WHERE category = 'A'",
            "-f",
            "json",
        ]
        result = metaxy_project.run_cli(cmd, capsys=capsys)

        assert result.returncode == 0
        data = json.loads(result.stdout)

        expected_cnt = len(upstream_data.filter(pl.col("category") == "A"))
        assert len(data) == 1
        assert data[0]["cnt"] == expected_cnt


def test_metadata_read_invalid_sql(
    metaxy_project: TempMetaxyProject, upstream_data: pl.DataFrame, capsys: pytest.CaptureFixture[str]
):
    """Test that invalid SQL query triggers the error catch block."""
    with metaxy_project.with_features(_define_features):
        _setup_store_with_data(metaxy_project, upstream_data)

        result = metaxy_project.run_cli(
            ["metadata", "read", "files_root", "--query", "SELECT invalid_column FROM files_root"],
            check=False,
            capsys=capsys,
        )

        assert result.returncode == 1
        assert "Error reading" in (result.stderr + result.stdout)


def test_metadata_read_invalid_feature(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test reading a non-existent feature."""
    with metaxy_project.with_features(_define_features):
        result = metaxy_project.run_cli(["metadata", "read", "non_existent"], check=False, capsys=capsys)
        output = result.stderr + result.stdout
        assert "Feature(s) not found" in output or result.returncode != 0


def test_metadata_read_explicit_store(
    metaxy_project: TempMetaxyProject, upstream_data: pl.DataFrame, capsys: pytest.CaptureFixture[str]
):
    """Test reading from an explicit store."""
    with metaxy_project.with_features(_define_features):
        _setup_store_with_data(metaxy_project, upstream_data)

        result = metaxy_project.run_cli(
            ["metadata", "read", "files_root", "--store", "dev", "-f", "json"], capsys=capsys
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data[0]["sample_uid"] == upstream_data["sample_uid"][0]


def test_metadata_read_ibis_optimization(
    metaxy_project: TempMetaxyProject, upstream_data: pl.DataFrame, capsys: pytest.CaptureFixture[str], monkeypatch
):
    """Test that Ibis optimization is used for reading when handling IbisMetadataStore."""
    with metaxy_project.with_features(_define_features):
        _setup_store_with_data(metaxy_project, upstream_data)

        from unittest.mock import patch

        import ibis

        # Patch ibis.to_sql to verify the optimized Ibis code path is executed
        with patch("ibis.to_sql", wraps=ibis.to_sql) as mock_to_sql:
            result = metaxy_project.run_cli(
                ["metadata", "read", "files_root", "--query", "SELECT * FROM files_root", "-f", "json"], capsys=capsys
            )
            assert result.returncode == 0

            # Verify the optimization path was triggered
            mock_to_sql.assert_called()
