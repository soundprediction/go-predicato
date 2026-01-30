"""
Offline unit tests for StatPearls NXML parsing.

These tests do NOT require a running Predicato server or network access.
Tests using real NXML fixture files are skipped if fixtures are not present
(the StatPearls data is not committed to the repo due to licensing).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from predicato.statpearls import (
    iter_articles_from_directory,
    parse_nxml_file,
    parse_nxml_string,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "statpearls"

# Minimal valid NXML content for tests that don't need real data
SAMPLE_NXML = """\
<book-part book-part-type="chapter">
  <book-part-meta>
    <title-group>
      <title>Sample Medical Article</title>
    </title-group>
    <contrib-group>
      <contrib contrib-type="author">
        <name><surname>Smith</surname><given-names>Jane</given-names></name>
      </contrib>
      <contrib contrib-type="author">
        <name><surname>Doe</surname><given-names>John</given-names></name>
      </contrib>
    </contrib-group>
  </book-part-meta>
  <body>
    <sec>
      <title>Introduction</title>
      <p>This is a sample paragraph about a medical topic with enough content to pass validation checks for minimum length requirements in the parser.</p>
    </sec>
    <sec>
      <title>Clinical Significance</title>
      <p>Another paragraph discussing clinical significance of the topic at hand with additional detail.</p>
    </sec>
  </body>
  <abstract>
    <p>This is the abstract of the article summarizing key findings.</p>
  </abstract>
</book-part>
"""

SAMPLE_NXML_NO_BODY = """\
<book-part book-part-type="chapter">
  <book-part-meta>
    <title-group><title>Empty Article</title></title-group>
  </book-part-meta>
</book-part>
"""


has_fixtures = pytest.mark.skipif(
    not FIXTURES_DIR.exists() or not list(FIXTURES_DIR.glob("*.nxml")),
    reason="StatPearls fixture files not present (not committed to repo)",
)


class TestParseNxmlString:
    """Tests for parse_nxml_string using inline XML."""

    def test_basic_parsing(self):
        result = parse_nxml_string(SAMPLE_NXML, source_id="test-article")
        assert result is not None
        assert result["id"] == "test-article"
        assert result["title"] == "Sample Medical Article"
        assert len(result["content"]) > 100
        assert "sample paragraph" in result["content"]

    def test_authors_extracted(self):
        result = parse_nxml_string(SAMPLE_NXML, source_id="test")
        assert result is not None
        assert result["authors"] == ["Jane Smith", "John Doe"]

    def test_abstract_extracted(self):
        result = parse_nxml_string(SAMPLE_NXML, source_id="test")
        assert result is not None
        assert "abstract" in result["abstract"].lower() or "summarizing" in result["abstract"].lower()

    def test_source_path_preserved(self):
        result = parse_nxml_string(SAMPLE_NXML, source_id="x", source_path="/some/path.nxml")
        assert result is not None
        assert result["source_path"] == "/some/path.nxml"

    def test_strips_xml_declaration(self):
        xml_with_decl = '<?xml version="1.0" encoding="UTF-8"?>\n' + SAMPLE_NXML
        result = parse_nxml_string(xml_with_decl, source_id="test")
        assert result is not None
        assert result["title"] == "Sample Medical Article"

    def test_strips_doctype(self):
        xml_with_doctype = '<!DOCTYPE book-part PUBLIC "-//NLM//DTD">\n' + SAMPLE_NXML
        result = parse_nxml_string(xml_with_doctype, source_id="test")
        assert result is not None
        assert result["title"] == "Sample Medical Article"

    def test_no_body_returns_none(self):
        result = parse_nxml_string(SAMPLE_NXML_NO_BODY, source_id="empty")
        assert result is None

    def test_malformed_xml_returns_none(self):
        result = parse_nxml_string("<broken><xml>", source_id="bad")
        assert result is None

    def test_empty_string_returns_none(self):
        result = parse_nxml_string("", source_id="empty")
        assert result is None

    def test_section_headers_in_content(self):
        result = parse_nxml_string(SAMPLE_NXML, source_id="test")
        assert result is not None
        assert "Introduction" in result["content"]
        assert "Clinical Significance" in result["content"]


class TestParseNxmlFile:
    """Tests for parse_nxml_file."""

    @has_fixtures
    def test_parse_fixture_file(self):
        files = list(FIXTURES_DIR.glob("*.nxml"))
        result = parse_nxml_file(files[0])
        assert result is not None
        assert result["title"]
        assert len(result["content"]) > 100
        assert result["source_path"] == str(files[0])

    @has_fixtures
    def test_parse_all_fixtures(self):
        files = list(FIXTURES_DIR.glob("*.nxml"))
        for f in files:
            result = parse_nxml_file(f)
            assert result is not None, f"Failed to parse {f.name}"
            assert result["title"], f"No title in {f.name}"
            assert result["content"], f"No content in {f.name}"

    def test_nonexistent_file_returns_none(self, tmp_path: Path):
        result = parse_nxml_file(tmp_path / "nonexistent.nxml")
        assert result is None

    def test_parse_from_tmp(self, tmp_path: Path):
        """Write sample NXML to a temp file and parse it."""
        nxml_file = tmp_path / "test.nxml"
        nxml_file.write_text(SAMPLE_NXML, encoding="utf-8")
        result = parse_nxml_file(nxml_file)
        assert result is not None
        assert result["id"] == "test"
        assert result["title"] == "Sample Medical Article"


class TestIterArticlesFromDirectory:
    """Tests for iter_articles_from_directory."""

    @has_fixtures
    def test_iterates_all_fixtures(self):
        articles = list(iter_articles_from_directory(FIXTURES_DIR))
        expected = len(list(FIXTURES_DIR.glob("*.nxml")))
        assert len(articles) == expected

    @has_fixtures
    def test_max_articles_limit(self):
        articles = list(iter_articles_from_directory(FIXTURES_DIR, max_articles=1))
        assert len(articles) == 1

    def test_empty_directory(self, tmp_path: Path):
        articles = list(iter_articles_from_directory(tmp_path))
        assert articles == []

    def test_directory_with_sample_files(self, tmp_path: Path):
        """Create temp NXML files and iterate them."""
        for i in range(3):
            (tmp_path / f"article-{i}.nxml").write_text(SAMPLE_NXML, encoding="utf-8")
        articles = list(iter_articles_from_directory(tmp_path))
        assert len(articles) == 3

    def test_max_articles_with_sample_files(self, tmp_path: Path):
        for i in range(5):
            (tmp_path / f"article-{i}.nxml").write_text(SAMPLE_NXML, encoding="utf-8")
        articles = list(iter_articles_from_directory(tmp_path, max_articles=2))
        assert len(articles) == 2
