"""URN parsing for Internet Archive identifiers.

Implements RFC 8141 URN syntax with r-component, q-component, and f-component:

    urn:NID:NSS[?+r-component][?=q-component][#f-component]

Examples:
    urn:ia:taleoftwocities00dick
    urn:ia:taleoftwocities00dick?+format=epub
    urn:ia:taleoftwocities00dick?+format=epub&file=specific.pdf
    urn:ia:taleoftwocities00dick?=page=5
    urn:ia:taleoftwocities00dick?+format=epub?=page=5#chapter1
"""

import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs

# RFC 8141 URN pattern with optional components
# urn:ia:NSS[?+r-component][?=q-component][#f-component]
URN_RFC8141_PATTERN = re.compile(
    r"^urn:(?P<nid>[a-zA-Z][a-zA-Z0-9-]*):(?P<nss>[^?#]+)"
    r"(?:\?\+(?P<r_component>[^?#]*))?"
    r"(?:\?=(?P<q_component>[^#]*))?"
    r"(?:#(?P<f_component>.*))?$"
)

# Simple identifier pattern (no URN prefix)
IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


class WrongResolverError(ValueError):
    """URN is valid but for a different resolver."""
    def __init__(self, source: str, urn: str):
        self.source = source
        self.urn = urn
        super().__init__(f"Wrong resolver: '{urn}' is a {source} URN, not ia")


class InvalidUrnError(ValueError):
    """URN format is invalid."""
    def __init__(self, urn: str, reason: str):
        self.urn = urn
        self.reason = reason
        super().__init__(f"Invalid URN: {reason}")


@dataclass
class ParsedUrn:
    """Parsed URN components per RFC 8141.

    Attributes:
        source: Namespace identifier (e.g., "ia")
        identifier: Namespace-specific string (the IA item identifier)
        r_component: Resolution parameters (hints for the resolver)
        q_component: Query parameters (passed to the resource)
        f_component: Fragment identifier (client-side, not sent to server)
    """

    source: str  # NID: "ia"
    identifier: str  # NSS: Internet Archive item identifier

    # RFC 8141 components
    r_component: dict[str, list[str]] = field(default_factory=dict)  # ?+key=value&...
    q_component: dict[str, list[str]] = field(default_factory=dict)  # ?=key=value&...
    f_component: str | None = None  # #fragment

    @property
    def format(self) -> str | None:
        """Get format hint from r-component (convenience accessor)."""
        formats = self.r_component.get("format", [])
        return formats[0] if formats else None

    @property
    def file(self) -> str | None:
        """Get specific file from r-component (convenience accessor)."""
        files = self.r_component.get("file", [])
        return files[0] if files else None

    def canonical(self) -> str:
        """Return canonical URN (without components)."""
        return f"urn:{self.source}:{self.identifier}"

    def with_r_component(self, **kwargs: str) -> str:
        """Return URN with r-component parameters."""
        base = self.canonical()
        if kwargs:
            params = "&".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{base}?+{params}"
        return base


def parse_urn(urn: str) -> ParsedUrn:
    """Parse a URN into its components per RFC 8141.

    Accepts:
    - Full URN: urn:ia:taleoftwocities00dick
    - URN with r-component: urn:ia:taleoftwocities00dick?+format=epub
    - URN with q-component: urn:ia:taleoftwocities00dick?=page=5
    - URN with fragment: urn:ia:taleoftwocities00dick#chapter1
    - Raw identifier: taleoftwocities00dick

    Raises:
        WrongResolverError: If URN is for a different namespace (anna, isbn, etc.)
        InvalidUrnError: If URN format is invalid
    """
    # Try RFC 8141 URN format
    if match := URN_RFC8141_PATTERN.match(urn):
        nid = match.group("nid").lower()
        nss = match.group("nss")
        r_raw = match.group("r_component")
        q_raw = match.group("q_component")
        f_raw = match.group("f_component")

        # Check namespace
        if nid != "ia":
            raise WrongResolverError(nid, urn)

        # Validate NSS (identifier)
        if not IDENTIFIER_PATTERN.match(nss):
            raise InvalidUrnError(urn, "ia URN must have valid identifier (alphanumeric, dots, hyphens, underscores)")

        # Parse components
        r_component = parse_qs(r_raw) if r_raw else {}
        q_component = parse_qs(q_raw) if q_raw else {}
        f_component = f_raw if f_raw else None

        return ParsedUrn(
            source="ia",
            identifier=nss,
            r_component=r_component,
            q_component=q_component,
            f_component=f_component,
        )

    # Try raw identifier (no urn: prefix)
    if IDENTIFIER_PATTERN.match(urn):
        return ParsedUrn(source="ia", identifier=urn)

    # Not a valid URN or identifier
    raise InvalidUrnError(urn, "expected urn:ia:<identifier> or raw Internet Archive identifier")


def to_urn(identifier: str, format: str | None = None, file: str | None = None) -> str:
    """Convert an identifier to URN format, optionally with r-component.

    Args:
        identifier: Internet Archive item identifier
        format: Optional format hint (epub, pdf, djvu, etc.)
        file: Optional specific filename to request

    Returns:
        URN string, with r-component if format/file specified
    """
    base = f"urn:ia:{identifier}"

    r_params = []
    if format:
        r_params.append(f"format={format}")
    if file:
        r_params.append(f"file={file}")

    if r_params:
        return f"{base}?+{'&'.join(r_params)}"
    return base
