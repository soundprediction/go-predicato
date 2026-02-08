"""
StatPearls NXML parsing utilities.

This module provides functions for parsing StatPearls NXML files (NCBI/JATS XML format)
and iterating over articles from tarballs or directories. It uses only the Python
standard library (xml.etree.ElementTree).

Example:
    >>> from predicato.statpearls import parse_nxml_file, iter_articles_from_directory
    >>> article = parse_nxml_file("path/to/article.nxml")
    >>> print(article["title"])
    >>> for article in iter_articles_from_directory("path/to/nxml/", max_articles=10):
    ...     print(article["title"])
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree

if TYPE_CHECKING:
    from collections.abc import Iterator


def parse_nxml_file(path: str | Path) -> dict[str, str | list[str]] | None:
    """
    Parse a StatPearls NXML file from disk.

    Args:
        path: Path to the .nxml or .xml file.

    Returns:
        Article dict with keys: id, title, content, authors, abstract, source_path.
        Returns None if the file cannot be parsed or has no content.
    """
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            xml_content = f.read()
    except (OSError, UnicodeDecodeError):
        return None

    return parse_nxml_string(xml_content, source_id=path.stem, source_path=str(path))


def parse_nxml_string(
    xml_content: str,
    source_id: str = "",
    source_path: str = "",
) -> dict[str, str | list[str]] | None:
    """
    Parse NXML content from a string.

    Args:
        xml_content: Raw XML string.
        source_id: Identifier for the article (defaults to empty string).
        source_path: Original file path for metadata.

    Returns:
        Article dict with keys: id, title, content, authors, abstract, source_path.
        Returns None if parsing fails or content is empty.
    """
    # Strip XML declaration and DOCTYPE for simpler parsing
    xml_content = re.sub(r"<\?xml[^>]+\?>", "", xml_content)
    xml_content = re.sub(r"<!DOCTYPE[^>]+>", "", xml_content)

    try:
        root = ElementTree.fromstring(xml_content)
    except ElementTree.ParseError:
        return None

    # Extract title
    title = _find_title(root)
    if not title:
        title = source_id or "Unknown"

    # Extract authors
    authors = _extract_authors(root)

    # Extract abstract
    abstract = ""
    abstract_elem = root.find(".//abstract")
    if abstract_elem is not None:
        abstract_parts = _extract_text_from_element(abstract_elem)
        abstract = "\n\n".join(abstract_parts)

    # Extract body content
    content_parts: list[str] = []
    body = root.find(".//body")
    if body is not None:
        content_parts.extend(_extract_text_from_element(body))

    # Prepend abstract if present
    if abstract:
        content_parts = ["Abstract:", abstract, "", *content_parts]

    content = "\n\n".join(content_parts)

    if not content:
        return None

    return {
        "id": source_id,
        "title": title,
        "content": content,
        "authors": authors,
        "abstract": abstract,
        "source_path": source_path,
    }


def iter_articles_from_tarball(
    tarball_path: str | Path,
    max_articles: int | None = None,
) -> Iterator[dict[str, str | list[str]]]:
    """
    Iterate through articles in a StatPearls tarball without extracting to disk.

    Args:
        tarball_path: Path to the .tar.gz file.
        max_articles: Maximum number of articles to yield.

    Yields:
        Article dicts with keys: id, title, content, authors, abstract, source_path.
    """
    count = 0

    with tarfile.open(str(tarball_path), "r:gz") as tar:
        for member in tar:
            if max_articles is not None and count >= max_articles:
                break

            if not (member.name.endswith(".xml") or member.name.endswith(".nxml")):
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue

                xml_content = f.read().decode("utf-8")
                article = parse_nxml_string(
                    xml_content,
                    source_id=Path(member.name).stem,
                    source_path=member.name,
                )
                if article and article.get("content"):
                    count += 1
                    yield article
            except Exception:
                continue


def iter_articles_from_directory(
    data_dir: str | Path,
    max_articles: int | None = None,
) -> Iterator[dict[str, str | list[str]]]:
    """
    Iterate through articles in a directory containing NXML/XML files.

    Args:
        data_dir: Path to directory containing .nxml or .xml files.
        max_articles: Maximum number of articles to yield.

    Yields:
        Article dicts with keys: id, title, content, authors, abstract, source_path.
    """
    data_path = Path(data_dir)
    xml_files = sorted(
        list(data_path.glob("**/*.xml")) + list(data_path.glob("**/*.nxml"))
    )

    count = 0
    for xml_path in xml_files:
        if max_articles is not None and count >= max_articles:
            break

        article = parse_nxml_file(xml_path)
        if article and article.get("content"):
            count += 1
            yield article


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_title(root: ElementTree.Element) -> str:
    """Find the article title from various possible locations."""
    for title_path in [
        ".//title-group/title",
        ".//book-part-meta/title-group/title",
        ".//book-title",
        ".//article-title",
        ".//title",
    ]:
        title_elem = root.find(title_path)
        if title_elem is not None:
            text = _get_element_text(title_elem)
            if text:
                return text
    return ""


def _extract_authors(root: ElementTree.Element) -> list[str]:
    """Extract author names from contrib elements."""
    authors: list[str] = []
    for contrib in root.findall(".//contrib[@contrib-type='author']"):
        surname_elem = contrib.find(".//surname")
        given_elem = contrib.find(".//given-names")
        surname = surname_elem.text.strip() if surname_elem is not None and surname_elem.text else ""
        given = given_elem.text.strip() if given_elem is not None and given_elem.text else ""
        name = f"{given} {surname}".strip()
        if name:
            authors.append(name)
    return authors


def _get_element_text(elem: ElementTree.Element) -> str:
    """Get all text content from an element, including nested elements."""
    text_parts: list[str] = []
    if elem.text:
        text_parts.append(elem.text.strip())
    for child in elem:
        text_parts.append(_get_element_text(child))
        if child.tail:
            text_parts.append(child.tail.strip())
    return " ".join(filter(None, text_parts))


def _extract_text_from_element(elem: ElementTree.Element) -> list[str]:
    """Extract text content from body-like elements, preserving structure."""
    content: list[str] = []

    for child in elem:
        tag = child.tag.lower() if isinstance(child.tag, str) else ""

        if tag in ("p", "para"):
            text = _get_element_text(child)
            if text:
                content.append(text)

        elif tag in ("sec", "section"):
            sec_title = child.find("title")
            if sec_title is not None:
                title_text = _get_element_text(sec_title)
                if title_text:
                    content.append(f"\n## {title_text}\n")
            content.extend(_extract_text_from_element(child))

        elif tag == "title":
            continue

        elif tag in ("list", "def-list"):
            for item in child.findall(".//list-item") or child.findall(".//def-item"):
                item_text = _get_element_text(item)
                if item_text:
                    content.append(f"- {item_text}")

        else:
            content.extend(_extract_text_from_element(child))

    return content
