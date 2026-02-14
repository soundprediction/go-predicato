"""
URL filtering and content relevance detection.

This module provides utilities for filtering URLs and determining
content relevance using keyword-based or LLM-based analysis.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

# Keyword categories for relevance detection
PREGNANCY_KEYWORDS = [
    "pregnancy", "pregnant", "prenatal", "maternity", "fertility", "gestation",
    "newborn", "baby", "delivery", "birth", "trimester", "conception",
    "motherhood", "fetus", "obstetrics", "gynecology", "postpartum",
    "breastfeeding", "lamaze", "doula", "midwife", "nausea",
    "morning sickness", "ultrasound", "due date", "contractions", "labor",
    "pregnant woman", "expecting", "maternal", "infant"
]

DIABETES_KEYWORDS = [
    "diabetes", "diabetic", "blood sugar", "insulin", "glucose", "type 1",
    "type 2", "gestational diabetes", "A1C", "prediabetes", "hypoglycemia",
    "hyperglycemia", "pancreas", "neuropathy", "retinopathy", "nephropathy",
    "diabetic ketoacidosis", "metformin", "glycemic", "blood glucose monitor",
    "endocrinology", "sugar level", "insulin resistance", "diabetic diet"
]

NUTRITION_KEYWORDS = [
    "nutrition", "diet", "food", "healthy eating", "vitamins", "minerals",
    "calories", "nutrients", "health", "wellness", "dietary", "balanced diet",
    "carbohydrates", "proteins", "fats", "fiber", "antioxidants",
    "superfood", "vegetarian", "vegan", "gluten-free", "macronutrients",
    "micronutrients", "hydration", "meal plan", "cooking", "recipes",
    "dietitian", "nutritional", "food guide", "eating habits", "supplements"
]

# Administrative URL keywords to filter out
ADMIN_KEYWORDS = [
    "about", "contact", "privacy", "terms", "legal", "admin", "login",
    "signin", "signup", "register", "account", "profile", "cart", "checkout",
    "billing", "payment", "sitemap", "search", "faq", "help", "support",
    "careers", "jobs", "investors", "press", "media", "newsroom",
    "corporate", "company", "us", "our-story", "policy", "cookie",
    "disclaimer", "feedback", "subscribe", "newsletter", "dashboard",
    "settings", "my-", "user", "licence", "terms-of-use", "accessibility",
    "advertising", "financial", "sponsorship", "contract",
    "services", "visitor", "visitor-information", "charity", "plugins", "rss"
]

# Combined keywords for default relevance checking
DEFAULT_KEYWORDS = PREGNANCY_KEYWORDS + DIABETES_KEYWORDS + NUTRITION_KEYWORDS


class KeywordFilter:
    """
    Keyword-based content relevance filter.

    Args:
        keywords: List of keywords to check for. If None, uses default health keywords.
        case_sensitive: Whether matching should be case-sensitive.

    Example:
        >>> filter = KeywordFilter(keywords=["python", "programming"])
        >>> filter.is_relevant("This article is about Python programming")
        True
    """

    def __init__(
        self,
        keywords: list[str] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.case_sensitive = case_sensitive

        if not case_sensitive:
            self.keywords = [k.lower() for k in self.keywords]

    def is_relevant(self, text: str) -> bool:
        """
        Check if text contains any of the configured keywords.

        Args:
            text: Text content to analyze.

        Returns:
            True if any keyword is found in the text.
        """
        if not text:
            return False

        check_text = text if self.case_sensitive else text.lower()
        return any(keyword in check_text for keyword in self.keywords)


class URLFilter:
    """
    URL filtering utilities.

    Provides methods for cleaning URLs, extracting domains, and
    filtering administrative or top-level URLs.

    Args:
        admin_keywords: Keywords indicating administrative pages.
        filter_admin: Whether to filter administrative URLs.
        filter_toplevel: Whether to filter top-level domain URLs.

    Example:
        >>> filter = URLFilter()
        >>> filter.is_administrative("/privacy-policy")
        True
        >>> filter.clean("https://example.com/page?foo=bar#section")
        'https://example.com/page'
    """

    def __init__(
        self,
        admin_keywords: list[str] | None = None,
        filter_admin: bool = True,
        filter_toplevel: bool = True,
    ) -> None:
        self.admin_keywords = admin_keywords or ADMIN_KEYWORDS
        self.filter_admin = filter_admin
        self.filter_toplevel = filter_toplevel

    def clean(self, url: str) -> str:
        """
        Remove query parameters and fragments from a URL.

        Args:
            url: URL to clean.

        Returns:
            Cleaned URL without query string or fragment.
        """
        parsed = urlparse(url)
        return urlunparse(parsed._replace(query="", fragment=""))

    def get_domain(self, url: str) -> str:
        """
        Extract the domain (netloc) from a URL.

        Args:
            url: URL to extract domain from.

        Returns:
            The domain portion of the URL.
        """
        return urlparse(url).netloc

    def is_administrative(self, url_or_path: str) -> bool:
        """
        Check if a URL or path indicates an administrative page.

        Args:
            url_or_path: Full URL or just the path portion.

        Returns:
            True if the URL contains administrative keywords.
        """
        if not self.filter_admin:
            return False

        parsed = urlparse(url_or_path)
        path = parsed.path.lower() if parsed.path else url_or_path.lower()

        return any(keyword in path for keyword in self.admin_keywords)

    def is_toplevel(self, url_or_path: str) -> bool:
        """
        Check if a URL refers to the root/top-level of a domain.

        Args:
            url_or_path: Full URL or just the path portion.

        Returns:
            True if the URL is a top-level page.
        """
        if not self.filter_toplevel:
            return False

        parsed = urlparse(url_or_path)
        path = parsed.path.strip("/")

        # Empty path or just index file
        if not path:
            return True

        # Check for common index file patterns at root
        index_patterns = [".html", ".htm", ".php", ".asp", ".aspx", ".jsp"]
        if path.count("/") == 0 and any(path.lower().endswith(ext) for ext in index_patterns):
            return True

        return False

    def should_crawl(self, url: str, base_domain: str | None = None) -> bool:
        """
        Determine if a URL should be crawled.

        Args:
            url: URL to check.
            base_domain: Optional domain to restrict crawling to.

        Returns:
            True if the URL should be crawled.
        """
        # Must be HTTP/HTTPS
        if not url.startswith(("http://", "https://")):
            return False

        # Check domain restriction
        if base_domain and self.get_domain(url) != base_domain:
            return False

        # Check filters
        if self.is_administrative(url):
            return False

        if self.is_toplevel(url):
            return False

        return True


# Boilerplate patterns for content cleaning
BOILERPLATE_PATTERNS = [
    # Navigation and menu patterns
    r"^(Navigation|Menu|Breadcrumb|Home|Search|Login|Register|Sign in|Sign up|Contact us|About us|Privacy|Terms|Copyright|All rights reserved)",
    r"^\s*(Home\s*\|\s*About\s*\|\s*Contact|Skip to main content|Skip to navigation)",
    # Header/footer patterns
    r"^\s*Header\s*$",
    r"^\s*Footer\s*$",
    r"^\s*Site Navigation\s*$",
    r"^\s*Main Navigation\s*$",
    # Social media and sharing
    r"^\s*(Share|Follow us|Social media|Facebook|Twitter|LinkedIn|Instagram|YouTube)",
    r"^\s*(Subscribe|Newsletter|Email updates)",
    # Common website elements
    r"^\s*(Search this site|Site search|Search results)",
    r"^\s*(Related articles|Related links|See also|More information)",
    r"^\s*(Print this page|Email this page|Share this page)",
    # Cookie and legal notices
    r"^\s*(Cookie|Cookies|This site uses cookies)",
    r"^\s*(Legal notice|Disclaimer|Terms of service|Privacy policy)",
    # Advertisement patterns
    r"^\s*(Advertisement|Ads|Sponsored)",
    # Empty or minimal content lines
    r"^\s*\|\s*\|\s*$",
    r"^\s*[\-\*\+]\s*$",
]

FOOTER_PATTERNS = [
    r"^\s*(Back to top|Top of page|Return to top)",
    r"^\s*(Last updated|Last modified|Page updated)",
    r"^\s*(Copyright|©|\(c\))",
    r"^\s*(Contact|Help|Support|Feedback)",
]


def remove_boilerplate(content: str) -> str:
    """
    Remove common boilerplate content from text.

    Removes headers, footers, navigation elements, and repetitive patterns
    commonly found in web page content.

    Args:
        content: Text or markdown content to clean.

    Returns:
        Cleaned content with boilerplate removed.

    Example:
        >>> content = "Navigation | Home | About\\n\\nActual article content here..."
        >>> cleaned = remove_boilerplate(content)
    """
    if not content:
        return content

    lines = content.split("\n")
    cleaned_lines = []

    # Compile patterns
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]
    footer_compiled = [re.compile(p, re.IGNORECASE) for p in FOOTER_PATTERNS]

    skip_until_content = True
    content_started = False

    for line in lines:
        line_stripped = line.strip()

        # Skip empty lines at the beginning
        if skip_until_content and not line_stripped:
            continue

        # Check if line matches boilerplate patterns
        is_boilerplate = any(p.match(line_stripped) for p in compiled_patterns)

        # Skip very short lines that are likely navigation elements
        if len(line_stripped) <= 3 and not content_started:
            continue

        # Skip lines that are just punctuation or symbols
        if line_stripped and all(c in "|-*+.◦▪▫=_~`" for c in line_stripped):
            continue

        if not is_boilerplate:
            # Mark that we've found actual content
            if line_stripped and len(line_stripped) > 10:
                content_started = True
                skip_until_content = False

            cleaned_lines.append(line)

    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    # Remove footer content from the end
    lines_to_check = min(5, len(cleaned_lines))
    for _ in range(lines_to_check):
        if not cleaned_lines:
            break

        last_line = cleaned_lines[-1].strip()
        is_footer = any(p.match(last_line) for p in footer_compiled)

        if is_footer or (len(last_line) <= 3 and last_line):
            cleaned_lines.pop()
        else:
            break

    return "\n".join(cleaned_lines).strip()
