"""Extract clean, section-segmented text from EDGAR filing HTML.

Responsibilities:
  - Parse raw EDGAR HTML into one text block per target 10-K item.
  - Try three extraction tiers per item (anchor, heading, plain-text
    regex) and fall back to the whole document if none succeeds.
  - Clean the text of boilerplate, short lines and numeric table rows.
  - Drop the hidden iXBRL header but keep tagged visible text.

Dependencies:
  - The lxml parser, used through BeautifulSoup.
"""

from __future__ import annotations

import re
import string

from bs4 import BeautifulSoup

from src.core import config
from src.core.logging import get_logger

logger = get_logger(__name__)

# Maps each target 10-K item label to the section name stored in
# chunk metadata.
_CANONICAL_NAMES = {
    "Item 1": "business_overview",
    "Item 1A": "risk_factors",
    "Item 1C": "cybersecurity",
    "Item 3": "legal_proceedings",
    "Item 7": "mda",
    "Item 7A": "quantitative_risk",
}


def _clean_extracted_text(text: str) -> str:
    """Clean extracted text of boilerplate and noise.

    Strips residual HTML tags, drops lines shorter than 20 characters
    and purely numeric lines (financial table rows), and collapses
    extra whitespace.

    Args:
        text: Raw extracted text, possibly with HTML fragments.

    Returns:
        Normalized plain text, ready for chunking.
    """
    soup = BeautifulSoup(text, "lxml")
    text = soup.get_text(separator=" ")

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if len(line_stripped) < 20:
            continue

        numeric_test = re.sub(r"[\s,\.$%]", "", line_stripped)
        if numeric_test.isdigit():
            continue

        punct_test = re.sub(rf"[{re.escape(string.punctuation)}\s]", "", line_stripped)
        if not punct_test:
            continue

        cleaned_lines.append(line_stripped)

    text = "\n".join(cleaned_lines)

    text = re.sub(r"[ ]{3,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_tier_1(soup: BeautifulSoup, item: str) -> str | None:
    """Extract a section by anchor (tier 1, highest fidelity).

    Finds an HTML anchor whose id or name exactly matches the
    normalized item label (for example 'item1a'), then collects all
    text up to the next item anchor.

    Args:
        soup: Parsed BeautifulSoup tree of the filing HTML.
        item: The SEC item label to locate, for example 'Item 1A'.

    Returns:
        The extracted text, or None if no matching anchor is found.
    """
    item_clean = item.replace(" ", "").lower()
    pattern = re.compile(rf"^{item_clean}$", re.IGNORECASE)

    anchor = soup.find(
        lambda tag: (
            tag.name
            and (
                (tag.get("id") and pattern.match(tag.get("id")))
                or (tag.name == "a" and tag.get("name") and pattern.match(tag.get("name")))
            )
        )
    )

    if not anchor:
        return None

    next_pattern = re.compile(r"^item\s*\d+[a-z]?$", re.IGNORECASE)
    texts = []

    for elem in anchor.next_elements:
        if getattr(elem, "name", None):
            if elem.get("id") and next_pattern.match(elem.get("id")):
                break
            if elem.name == "a" and elem.get("name") and next_pattern.match(elem.get("name")):
                break
        if isinstance(elem, str):
            texts.append(str(elem))

    return " ".join(texts)


def _extract_tier_2(soup: BeautifulSoup, item: str) -> str | None:
    """Extract a section by heading tag (tier 2, structural fallback).

    Finds an h1, h2 or h3 tag whose text starts with the item label,
    then collects all text up to the next item heading. Used when
    anchor ids are absent or malformed.

    Args:
        soup: Parsed BeautifulSoup tree of the filing HTML.
        item: The SEC item label to locate, for example 'Item 1A'.

    Returns:
        The extracted text, or None if no matching heading is found.
    """
    item_pattern = item.replace(" ", r"\s+")
    pattern = re.compile(rf"^\s*{item_pattern}[\.\s:-]", re.IGNORECASE)

    heading = soup.find(lambda tag: tag.name in ["h1", "h2", "h3"] and pattern.match(tag.get_text(strip=True)))

    if not heading:
        return None

    next_pattern = re.compile(r"^\s*item\s*\d+[a-z]?[\.\s:-]", re.IGNORECASE)
    texts = []

    for elem in heading.next_elements:
        if getattr(elem, "name", None) in ["h1", "h2", "h3"]:
            text_val = elem.get_text(strip=True)
            if pattern.match(text_val):
                continue
            if next_pattern.match(text_val):
                break
        if isinstance(elem, str):
            texts.append(str(elem))

    return " ".join(texts)


def _extract_tier_3(text: str, item: str) -> str | None:
    """Extract a section by plain-text regex (tier 3, last resort).

    Matches item boundaries in the whole document text and keeps the
    longest candidate, which avoids table-of-contents entries because
    those are short.

    Args:
        text: Full plain text of the filing, from soup.get_text.
        item: The SEC item label to locate, for example 'Item 1A'.

    Returns:
        The longest matching chunk if it exceeds 100 characters,
        otherwise None.
    """
    item_pattern = item.replace(" ", r"\s+")
    start_pattern = rf"(?i)\b{item_pattern}(?:[\.\s-]|:)"
    end_pattern = r"(?i)\bItem\s*\d+[A-Z]?(?:[\.\s-]|:)"

    matches = list(re.finditer(start_pattern, text))
    if not matches:
        return None

    best_chunk = ""
    for match in matches:
        start_idx = match.start()
        next_match = re.search(end_pattern, text[match.end() :])
        if next_match:
            end_idx = match.end() + next_match.start()
        else:
            end_idx = len(text)

        chunk = text[start_idx:end_idx]
        if len(chunk) > len(best_chunk):
            best_chunk = chunk

    if len(best_chunk) > 100:
        return best_chunk
    return None


def extract_sections(html_content: str) -> dict[str, str]:
    """Extract the target sections from an EDGAR filing as clean text.

    Removes the hidden iXBRL header, then tries three tiers for each
    item in config.EDGAR_TARGET_SECTIONS: anchor, heading tag and
    plain-text regex. A section needs at least 100 characters to
    count. If no item yields a section, the whole cleaned document is
    returned under the 'full_text' key.

    Args:
        html_content: Raw HTML string of the EDGAR filing.

    Returns:
        A dict mapping section names, for example 'mda' or
        'risk_factors', to cleaned text.
    """
    # Only ix:header holds hidden XBRL metadata. Other ix:* tags wrap
    # visible text such as financial figures, so their text stays.
    html_cleaned = re.sub(r"<ix:header\b[^>]*>.*?</ix:header>", "", html_content, flags=re.DOTALL | re.IGNORECASE)

    soup = BeautifulSoup(html_cleaned, "lxml")
    full_text = soup.get_text(separator=" ")

    results = {}
    extracted_any = False

    for item in config.EDGAR_TARGET_SECTIONS:
        key = _CANONICAL_NAMES.get(item, item.lower().replace(" ", "_"))
        content = None

        content = _extract_tier_1(soup, item)

        if not content or len(content.strip()) < 100:
            content = _extract_tier_2(soup, item)

        if not content or len(content.strip()) < 100:
            content = _extract_tier_3(full_text, item)

        if content and len(content.strip()) >= 100:
            cleaned = _clean_extracted_text(content)
            if cleaned:
                results[key] = cleaned
                extracted_any = True

    if not extracted_any:
        logger.warning("No sections extracted via Tiers 1-3. Falling back to full text.")
        cleaned_full = _clean_extracted_text(full_text)
        if cleaned_full:
            results["full_text"] = cleaned_full

    return results
