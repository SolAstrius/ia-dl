"""Internet Archive download microservice.

A FastAPI service that downloads files from Internet Archive,
caches them in S3, and returns presigned URLs.

Designed to run with Python 3.13 free-threaded mode for true parallelism.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from .ia_client import (
    IAClient,
    IAClientError,
    ItemMetadata,
    ItemNotFoundError,
    RateLimitedError,
    SearchItem,
    SearchResult,
)
from .config import Settings, get_settings
from .db import Database
from .downloader import DownloadError, download_item
from .s3 import S3Storage, content_type_for_format
from .urn import parse_urn, to_urn, ParsedUrn, WrongResolverError, InvalidUrnError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Global state (initialized in lifespan)
_ia_client: IAClient | None = None
_s3_storage: S3Storage | None = None
_db: Database | None = None


@lru_cache
def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    return get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global _ia_client, _s3_storage, _db

    settings = get_cached_settings()

    # Validate required settings for server mode
    if not settings.s3_bucket:
        raise RuntimeError("IA_DL_S3_BUCKET is required for server mode")

    # Initialize Internet Archive client
    _ia_client = IAClient.create(
        timeout=settings.download_timeout,
        access_key=settings.ia_access_key,
        secret_key=settings.ia_secret_key,
    )
    logger.info(
        "Initialized Internet Archive client (auth=%s)",
        "enabled" if settings.ia_access_key else "disabled",
    )

    # Initialize S3 storage
    _s3_storage = S3Storage.create(settings)
    logger.info("Initialized S3 storage (bucket=%s)", settings.s3_bucket)

    # Initialize PostgreSQL (shared annas-mcp database, optional)
    if settings.database_url:
        try:
            _db = Database(settings.database_url)
            await _db.connect()
        except Exception as exc:
            logger.warning("Failed to connect to PostgreSQL: %s", exc)
            _db = None

    yield

    # Cleanup
    if _db:
        await _db.close()

    if _ia_client:
        await _ia_client.close()
        logger.info("Closed Internet Archive client")


app = FastAPI(
    title="Internet Archive Download Service",
    description="Microservice for downloading files from Internet Archive with S3 caching",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models


class DownloadRequest(BaseModel):
    """Request to download an item."""

    preferred_formats: list[str] = ["pdf", "epub", "fb2", "mobi", "djvu"]
    specific_file: str | None = None  # Download specific file instead of auto-selecting


class DownloadResponse(BaseModel):
    """Response from download endpoint."""

    id: str  # URN: urn:ia:<identifier>
    identifier: str
    title: str
    filename: str
    format: str
    download_url: str
    size_bytes: int
    duration_ms: int
    cached: bool


class BatchDownloadRequest(BaseModel):
    """Request to download multiple items."""

    items: list[dict]  # List of {identifier, preferred_formats, specific_file}


class BatchDownloadResponse(BaseModel):
    """Response from batch download endpoint."""

    results: dict[str, DownloadResponse | dict]
    duration_ms: int
    successful: int
    failed: int


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: str  # "ok" or "unavailable"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "ok" or "degraded"
    version: str
    components: dict[str, ComponentHealth] = {}


# =============================================================================
# Search Models
# =============================================================================


class SearchItemResponse(BaseModel):
    """A single search result item with normalized fields.

    All format names are normalized to lowercase file extensions (pdf, epub, djvu)
    rather than IA's verbose names (Text PDF, EPUB, DjVu).
    """

    id: str
    """URN identifier: urn:ia:<identifier>. Use this for download/metadata endpoints."""

    identifier: str
    """Internet Archive item identifier (the part after urn:ia:)."""

    title: str
    """Item title."""

    creator: list[str]
    """Author(s) / creator(s)."""

    publisher: str
    """Publisher name."""

    date: str
    """Publication date (ISO format when available)."""

    year: int | None
    """Publication year (extracted from date)."""

    language: str
    """ISO 639-2 language code (eng, fra, deu, etc.)."""

    mediatype: str
    """IA mediatype: texts, audio, movies, software, image, data, web, collection."""

    formats: list[str]
    """Available formats as normalized lowercase extensions: pdf, epub, djvu, txt, mp3, etc.
    Excludes metadata formats (json, xml, marc, etc.)."""

    downloads: int
    """Total download count (all time)."""

    item_size: int
    """Total size of all files in bytes."""

    imagecount: int
    """Number of page images (for scanned books)."""

    collection: list[str]
    """IA collections this item belongs to."""

    subject: list[str]
    """Subject tags/keywords."""


class SearchResponse(BaseModel):
    """Paginated search results.

    Results are sorted by downloads (most popular first) by default.
    Use the `sort` query parameter to change sort order.
    """

    total: int
    """Total number of matching items across all pages."""

    page: int
    """Current page number (1-indexed)."""

    rows: int
    """Number of results requested per page."""

    items: list[SearchItemResponse]
    """Search results for this page."""

    query: str
    """The processed query that was sent to Internet Archive (useful for debugging)."""


# =============================================================================
# RFC 2483 Resolution Service Models
# =============================================================================


class I2LResponse(BaseModel):
    """I2L: URN to URL resolution response."""

    urn: str
    url: str


class I2LsResponse(BaseModel):
    """I2Ls: URN to multiple URLs resolution response."""

    urn: str
    urls: list[str]


class I2CResponse(BaseModel):
    """I2C: URN to URC (Uniform Resource Characteristics) response.

    URC provides metadata about the resource without fetching it.
    """

    urn: str
    identifier: str
    title: str | None = None
    filename: str | None = None
    format: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None
    cached: bool = False


class I2NResponse(BaseModel):
    """I2N: URN to canonical URN resolution response."""

    input_urn: str
    canonical_urn: str


class ItemInfoResponse(BaseModel):
    """Item metadata from Internet Archive (without downloading).

    Fetched directly from Internet Archive /metadata/ endpoint.
    """

    urn: str
    identifier: str

    # Core fields
    title: str
    description: str = ""
    creator: str | list[str] = ""
    date: str = ""
    year: str = ""

    # Additional metadata
    subject: list[str] = []
    collection: list[str] = []
    language: str = ""
    mediatype: str = ""

    # Size info
    item_size: int = 0
    files_count: int = 0

    # Download counts
    downloads: int = 0

    # External identifiers
    isbn: list[str] = []
    oclc: list[str] = []
    lccn: list[str] = []
    external_identifier: list[str] = []  # ACS6, LCP, OCLC record URNs

    # Available files (name, format, size)
    files: list[dict] = []


class ErrorResponse(BaseModel):
    """RFC 2483 compliant error response.

    Error categories from RFC 2483:
    - malformed_uri: URI syntax is invalid (400)
    - wrong_resolver: Valid URN but wrong namespace for this resolver (421)
    - not_found: URI doesn't exist in any form (404)
    - gone: URI existed in the past but no longer available (410)
    - access_denied: Authentication/authorization failure (401/403)
    - quota_exceeded: Rate limit or quota exhausted (429)
    - upstream_error: Resolution service dependency failed (502)
    - unavailable: Service temporarily unavailable (503)
    """

    error: str  # Error category (e.g., "not_found", "wrong_resolver")
    detail: str  # Human-readable description
    urn: str | None = None  # The URN that caused the error, if applicable


# Error helper
def error_response(status_code: int, error: str, detail: str, urn: str | None = None) -> HTTPException:
    """Create an HTTPException with RFC 2483 compliant error body."""
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(error=error, detail=detail, urn=urn).model_dump(exclude_none=True),
    )


# Endpoints


@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """Health check endpoint with component-level status."""
    components: dict[str, ComponentHealth] = {}

    s3_ok = _s3_storage is not None
    components["s3"] = ComponentHealth(status="ok" if s3_ok else "unavailable")

    if _db is not None:
        db_ok = await _db.ping()
        components["db"] = ComponentHealth(status="ok" if db_ok else "unavailable")

    degraded = any(c.status != "ok" for c in components.values())

    return HealthResponse(
        status="degraded" if degraded else "ok",
        version="0.1.0",
        components=components,
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
@app.get("/items/search", response_model=SearchResponse, tags=["Search"], deprecated=True)
async def search_items(
    q: str = Query(
        ...,
        description="Search query. Supports natural language or Lucene syntax. "
        "Use comma to separate title and author: 'Great Expectations, Dickens'. "
        "Without comma, terms are searched independently: '19th century maritime navigation'.",
        examples=["Great Expectations, Dickens", "19th century maritime navigation", "creator:dickens AND year:[1850 TO 1870]"],
    ),
    rows: int = Query(50, ge=1, le=1000, description="Results per page (max 1000)"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    sort: str | None = Query(
        None,
        description="Sort field and direction. Default: 'downloads desc'. "
        "Options: downloads, date, title, creator, publicdate (each with 'asc' or 'desc').",
        examples=["downloads desc", "date asc", "title asc"],
    ),
    mediatype: str | None = Query(
        None,
        description="Filter by mediatype",
        examples=["texts", "audio", "movies", "software", "image"],
    ),
    year_min: int | None = Query(None, description="Minimum publication year", examples=[1800, 1900, 2000]),
    year_max: int | None = Query(None, description="Maximum publication year", examples=[1900, 2000, 2024]),
    language: str | None = Query(
        None,
        description="Filter by ISO 639-2 language code",
        examples=["eng", "fra", "deu", "spa", "rus"],
    ),
    format: str | None = Query(
        None,
        description="Filter by format (normalized lowercase). Added to query as format filter.",
        examples=["pdf", "epub", "djvu", "mp3"],
    ),
) -> SearchResponse:
    """Search Internet Archive items.

    ## Query Syntax

    The search supports two modes, automatically detected:

    ### Natural Language (recommended for most searches)

    **With comma** — Title + Author pattern. Each part is quoted as a phrase:
    - `Great Expectations, Dickens` → searches for "Great Expectations" AND Dickens
    - `War and Peace, Tolstoy` → searches for "War and Peace" AND Tolstoy

    **Without comma** — Topic/conceptual search. Terms searched independently:
    - `19th century maritime navigation` → finds books about maritime navigation in the 19th century
    - `french revolution causes` → finds books discussing causes of the French revolution

    ### Advanced Lucene Syntax

    For precise control, use Lucene query syntax (auto-detected):

    | Syntax | Example | Description |
    |--------|---------|-------------|
    | Field search | `creator:dickens` | Search specific field |
    | Phrase | `title:"great expectations"` | Exact phrase match |
    | Boolean | `dickens AND london` | Both terms required |
    | OR | `dickens OR thackeray` | Either term |
    | NOT | `dickens NOT christmas` | Exclude term |
    | Wildcard | `dick*` | Prefix matching |
    | Range | `year:[1800 TO 1900]` | Numeric/date range |
    | Grouping | `(dickens OR austen) AND london` | Complex queries |

    ## Response Format

    - **id**: URN identifier (`urn:ia:<identifier>`)
    - **formats**: Normalized lowercase extensions (`["pdf", "epub"]` not `["Text PDF", "EPUB"]`)
    - **Results sorted by downloads** (most popular first) unless `sort` specified

    ## Examples

    | Query | Finds |
    |-------|-------|
    | `Great Expectations, Dickens` | The novel by Charles Dickens |
    | `19th century maritime navigation` | Books about sailing/navigation history |
    | `creator:dickens year:[1850 TO 1870]` | Dickens works from 1850-1870 |
    | `subject:philosophy language:grc` | Philosophy texts in Greek |
    """
    if _ia_client is None:
        raise error_response(503, "unavailable", "Service not initialized")

    # Add format filter to query if specified
    full_query = q
    if format:
        # IA format field uses verbose names, but we accept normalized names
        format_map = {
            "pdf": "PDF",
            "epub": "EPUB",
            "djvu": "DjVu",
            "mobi": "Mobi",
            "txt": "Text",
            "mp3": "MP3",
            "mp4": "MP4",
        }
        ia_format = format_map.get(format.lower(), format)
        full_query = f"({q}) AND format:{ia_format}"

    try:
        result = await _ia_client.search(
            query=full_query,
            rows=rows,
            page=page,
            sort=sort,
            mediatype=mediatype,
            year_min=year_min,
            year_max=year_max,
            language=language,
        )
    except RateLimitedError as exc:
        raise error_response(429, "quota_exceeded", str(exc))
    except IAClientError as exc:
        raise error_response(502, "upstream_error", str(exc))

    # Convert to response format with URNs
    items = [
        SearchItemResponse(
            id=item.to_urn(),
            identifier=item.identifier,
            title=item.title,
            creator=item.creator,
            publisher=item.publisher,
            date=item.date,
            year=item.year,
            language=item.language,
            mediatype=item.mediatype,
            formats=item.formats,
            downloads=item.downloads,
            item_size=item.item_size,
            imagecount=item.imagecount,
            collection=item.collection,
            subject=item.subject,
        )
        for item in result.items
    ]

    return SearchResponse(
        total=result.total,
        page=page,
        rows=rows,
        items=items,
        query=result.query,
    )


@app.post("/download/{id:path}", response_model=DownloadResponse)
@app.post("/item/{id:path}/download", response_model=DownloadResponse, deprecated=True)
async def download_item_endpoint(
    id: str,
    request: DownloadRequest | None = None,
) -> DownloadResponse:
    """Download an item from Internet Archive.

    If the item is already cached in S3, returns the presigned URL immediately.
    Otherwise, downloads from Internet Archive.

    Args:
        id: URN (urn:ia:<identifier>) or raw identifier
        request: Optional request body with format preferences

    Returns:
        DownloadResponse with presigned URL and metadata
    """
    if _ia_client is None or _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    # Parse URN or raw identifier
    try:
        parsed = parse_urn(id)
        identifier = parsed.identifier
        urn = to_urn(identifier)
    except WrongResolverError as e:
        raise error_response(421, "wrong_resolver", str(e), urn=id)
    except InvalidUrnError as e:
        raise error_response(400, "malformed_uri", str(e), urn=id)

    settings = get_cached_settings()

    preferred_formats = request.preferred_formats if request else ["pdf", "epub", "fb2", "mobi", "djvu"]
    specific_file = request.specific_file if request else None

    # Check cache - look for any of the preferred formats
    for fmt in preferred_formats:
        book_key = _s3_storage.book_key(identifier, fmt)
        if _s3_storage.exists(book_key):
            filename = f"{identifier}.{fmt}"
            logger.info("Cache hit for identifier=%s format=%s", identifier, fmt)

            # Try to get title from metadata
            title = ""
            try:
                meta_bytes = _s3_storage.download(_s3_storage.meta_key(identifier))
                cached_metadata = json.loads(meta_bytes)
                title = cached_metadata.get("title", "")
            except Exception:
                pass

            url = _s3_storage.get_presigned_url(book_key, filename)
            return DownloadResponse(
                id=urn,
                identifier=identifier,
                title=title,
                filename=filename,
                format=fmt,
                download_url=url,
                size_bytes=0,  # Could HEAD the object but not worth it
                duration_ms=0,
                cached=True,
            )

    logger.info("Cache miss for identifier=%s formats=%s, downloading", identifier, preferred_formats)
    start_time = time.monotonic()

    try:
        result = await download_item(
            _ia_client, settings, identifier, preferred_formats, specific_file
        )
    except DownloadError as exc:
        logger.error("Download failed for identifier=%s: %s", identifier, exc)
        if exc.last_status == 404:
            raise error_response(404, "not_found", str(exc), urn=urn)
        elif exc.last_status == 429:
            raise error_response(429, "quota_exceeded", str(exc), urn=urn)
        elif exc.last_status and exc.last_status >= 500:
            raise error_response(502, "upstream_error", str(exc), urn=urn)
        else:
            raise error_response(500, "upstream_error", str(exc), urn=urn)

    # Upload to S3
    actual_key = _s3_storage.book_key(identifier, result.format)
    meta_key = _s3_storage.meta_key(identifier)
    _s3_storage.upload(actual_key, result.content, result.content_type)

    # Fetch and store metadata
    metadata = {
        "identifier": identifier,
        "filename": result.filename,
        "format": result.format,
        "size_bytes": result.size_bytes,
        "title": "",
    }

    # Try to fetch rich metadata from Internet Archive
    item_meta: ItemMetadata | None = None
    try:
        item_meta = await _ia_client.fetch_metadata(identifier)
        metadata["title"] = item_meta.title
        metadata["ia"] = asdict(item_meta)
        # Include JSON-LD for interoperability
        base_url = settings.base_url or ""
        metadata["jsonld"] = item_meta.to_jsonld(urn, base_url)
        # Promote key fields to top level for MCP server compatibility
        if item_meta.title:
            metadata["title"] = item_meta.title
        if item_meta.creator:
            if isinstance(item_meta.creator, list):
                metadata["authors"] = ", ".join(item_meta.creator)
            else:
                metadata["authors"] = item_meta.creator
        if item_meta.publisher:
            metadata["publisher"] = item_meta.publisher
        if item_meta.language:
            metadata["language"] = item_meta.language
        if item_meta.item_size:
            b = item_meta.item_size
            metadata["size"] = (
                f"{b / 1024**3:.1f}GB" if b >= 1024**3
                else f"{b / 1024**2:.1f}MB" if b >= 1024**2
                else f"{b / 1024:.1f}KB" if b >= 1024
                else f"{b} bytes"
            )
        metadata["url"] = f"https://archive.org/details/{identifier}"
    except Exception as exc:
        logger.warning("Failed to fetch metadata for identifier=%s: %s", identifier, exc)

    _s3_storage.upload(meta_key, json.dumps(metadata).encode(), "application/json")

    # Persist to PostgreSQL
    if _db and item_meta:
        await _db.upsert_book(identifier, item_meta)

    # Generate presigned URL
    url = _s3_storage.get_presigned_url(actual_key, result.filename)

    total_duration_ms = int((time.monotonic() - start_time) * 1000)

    return DownloadResponse(
        id=urn,
        identifier=identifier,
        title=metadata.get("title", ""),
        filename=result.filename,
        format=result.format,
        download_url=url,
        size_bytes=result.size_bytes,
        duration_ms=total_duration_ms,
        cached=False,
    )


# =============================================================================
# RFC 2483 Resolution Service Endpoints
# =============================================================================


def _parse_and_validate_urn(id: str) -> ParsedUrn:
    """Parse URN per RFC 8141 and return ParsedUrn. Raises HTTPException on error.

    The returned ParsedUrn contains:
    - identifier: the IA item identifier
    - r_component: resolution hints (e.g., format=epub)
    - q_component: query params (passed to resource)
    - f_component: fragment

    Use parsed.canonical() to get the canonical URN string.
    Use parsed.format to get format hint from r-component.
    """
    try:
        return parse_urn(id)
    except WrongResolverError as e:
        raise error_response(421, "wrong_resolver", str(e), urn=id)
    except InvalidUrnError as e:
        raise error_response(400, "malformed_uri", str(e), urn=id)


async def _ensure_cached(
    identifier: str,
    urn: str,
    preferred_formats: list[str] | None = None,
    format_required: bool = False,
) -> tuple[str, str]:
    """Ensure resource is cached, fetching from upstream if needed.

    Args:
        identifier: IA item identifier
        urn: Canonical URN
        preferred_formats: List of formats in preference order
        format_required: If True, only the first format is acceptable (explicit request).
                        If False, any cached format from the list is acceptable.

    Returns (filename, format) of cached file.
    """
    if _ia_client is None or _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    preferred_formats = preferred_formats or ["pdf", "epub", "fb2", "mobi", "djvu"]

    # Check cache based on whether format was explicitly requested
    if format_required:
        # Only check for the specific requested format
        fmt = preferred_formats[0]
        key = _s3_storage.book_key(identifier, fmt)
        if _s3_storage.exists(key):
            filename = f"{identifier}.{fmt}"
            logger.info("Cache hit for %s format=%s (required)", identifier, fmt)
            return filename, fmt
    else:
        # Check if ANY of the preferred formats is cached
        for fmt in preferred_formats:
            key = _s3_storage.book_key(identifier, fmt)
            if _s3_storage.exists(key):
                filename = f"{identifier}.{fmt}"
                logger.info("Cache hit for %s format=%s", identifier, fmt)
                return filename, fmt

    # Not cached - fetch from upstream
    logger.info("Cache miss for %s formats=%s, downloading", identifier, preferred_formats)
    settings = get_cached_settings()

    try:
        result = await download_item(_ia_client, settings, identifier, preferred_formats)
    except DownloadError as exc:
        if exc.last_status == 404:
            raise error_response(404, "not_found", str(exc), urn=urn)
        elif exc.last_status == 429:
            raise error_response(429, "quota_exceeded", str(exc), urn=urn)
        else:
            raise error_response(502, "upstream_error", str(exc), urn=urn)

    # Cache the file
    actual_key = _s3_storage.book_key(identifier, result.format)
    _s3_storage.upload(actual_key, result.content, result.content_type)

    # Update metadata (stores latest download info + full IA metadata)
    metadata = {
        "identifier": identifier,
        "filename": result.filename,
        "format": result.format,
        "size_bytes": result.size_bytes,
    }

    item_meta: ItemMetadata | None = None
    try:
        item_meta = await _ia_client.fetch_metadata(identifier)
        metadata["title"] = item_meta.title
        metadata["ia"] = asdict(item_meta)
        base_url = settings.base_url or ""
        metadata["jsonld"] = item_meta.to_jsonld(urn, base_url)
        # Promote key fields to top level for MCP server compatibility
        if item_meta.title:
            metadata["title"] = item_meta.title
        if item_meta.creator:
            if isinstance(item_meta.creator, list):
                metadata["authors"] = ", ".join(item_meta.creator)
            else:
                metadata["authors"] = item_meta.creator
        if item_meta.publisher:
            metadata["publisher"] = item_meta.publisher
        if item_meta.language:
            metadata["language"] = item_meta.language
        if item_meta.item_size:
            b = item_meta.item_size
            metadata["size"] = (
                f"{b / 1024**3:.1f}GB" if b >= 1024**3
                else f"{b / 1024**2:.1f}MB" if b >= 1024**2
                else f"{b / 1024:.1f}KB" if b >= 1024
                else f"{b} bytes"
            )
        metadata["url"] = f"https://archive.org/details/{identifier}"
    except Exception as exc:
        logger.warning("Failed to fetch metadata in _ensure_cached for identifier=%s: %s", identifier, exc)

    _s3_storage.upload(_s3_storage.meta_key(identifier), json.dumps(metadata).encode(), "application/json")

    # Persist to PostgreSQL
    if _db and item_meta:
        await _db.upsert_book(identifier, item_meta)

    return result.filename, result.format


@app.get("/urn/{id:path}/urls", response_model=I2LsResponse)
async def resolve_i2ls(
    id: str,
    format: str | None = Query(None, description="Preferred format"),
    r_format: str | None = Query(None, alias="+format", description="Format via r-component (?+format=)"),
) -> I2LsResponse:
    """I2Ls: Resolve URN to multiple URLs.

    RFC 2483 I2Ls operation. Returns all available URLs for the resource.
    For Internet Archive items, this returns the cached file URL.

    Format can be specified via:
    - URN r-component: urn:ia:item?+format=epub (RFC 8141)
    - Query param: ?format=epub (fallback)
    """
    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    parsed = _parse_and_validate_urn(id)
    urn = parsed.canonical()

    # Format priority: r-component > +format param > format param > default
    explicit_format = parsed.format or r_format or format
    preferred_format = explicit_format or "pdf"

    # Ensure at least one format is cached
    # format_required=True if user explicitly requested a format
    filename, fmt = await _ensure_cached(
        parsed.identifier, urn, [preferred_format], format_required=bool(explicit_format)
    )

    key = _s3_storage.book_key(parsed.identifier, fmt)
    url = _s3_storage.get_presigned_url(key, filename)

    return I2LsResponse(urn=urn, urls=[url])


@app.get("/urn/{id:path}/resource")
async def resolve_i2r(
    id: str,
    format: str | None = Query(None, description="Preferred format"),
    r_format: str | None = Query(None, alias="+format", description="Format via r-component (?+format=)"),
) -> StreamingResponse:
    """I2R: Resolve URN directly to resource bytes.

    RFC 2483 I2R operation. Streams the actual resource content.
    Fetches from upstream if not cached.

    Format can be specified via:
    - URN r-component: urn:ia:item?+format=epub (RFC 8141)
    - Query param: ?format=epub (fallback)
    """
    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    parsed = _parse_and_validate_urn(id)
    urn = parsed.canonical()
    explicit_format = parsed.format or r_format or format
    preferred_format = explicit_format or "pdf"

    # Ensure resource is cached
    filename, fmt = await _ensure_cached(
        parsed.identifier, urn, [preferred_format], format_required=bool(explicit_format)
    )

    key = _s3_storage.book_key(parsed.identifier, fmt)
    content = _s3_storage.download(key)

    return StreamingResponse(
        iter([content]),
        media_type=content_type_for_format(fmt),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-URN": urn,
        },
    )


@app.get("/urn/{id:path}/metadata", response_model=I2CResponse)
async def resolve_i2c(
    id: str,
    format: str | None = Query(None, description="Preferred format"),
    r_format: str | None = Query(None, alias="+format", description="Format via r-component (?+format=)"),
) -> I2CResponse:
    """I2C: Resolve URN to URC (Uniform Resource Characteristics).

    RFC 2483 I2C operation. Returns metadata about the resource.
    Fetches from upstream if not cached.

    Format can be specified via:
    - URN r-component: urn:ia:item?+format=epub (RFC 8141)
    - Query param: ?format=epub (fallback)
    """

    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    parsed = _parse_and_validate_urn(id)
    urn = parsed.canonical()
    explicit_format = parsed.format or r_format or format
    preferred_format = explicit_format or "pdf"

    # Ensure resource is cached
    filename, fmt = await _ensure_cached(
        parsed.identifier, urn, [preferred_format], format_required=bool(explicit_format)
    )

    # Load metadata
    meta_key = _s3_storage.meta_key(parsed.identifier)
    try:
        meta_bytes = _s3_storage.download(meta_key)
        metadata = json.loads(meta_bytes)
    except Exception:
        metadata = {"identifier": parsed.identifier, "format": fmt, "filename": filename}

    size_bytes = metadata.get("size_bytes")
    return I2CResponse(
        urn=urn,
        identifier=metadata.get("identifier", parsed.identifier),
        title=metadata.get("title"),
        filename=metadata.get("filename", filename),
        format=metadata.get("format", fmt),
        size_bytes=int(size_bytes) if size_bytes else None,
        content_type=content_type_for_format(metadata.get("format", fmt)),
        cached=True,
    )


@app.get("/urn/{id:path}/canonical", response_model=I2NResponse)
async def resolve_i2n(id: str) -> I2NResponse:
    """I2N: Resolve to canonical URN.

    RFC 2483 I2N operation. Normalizes the input URN to its canonical form.
    Strips r-component, q-component, and f-component per RFC 8141.
    """
    parsed = _parse_and_validate_urn(id)
    return I2NResponse(input_urn=id, canonical_urn=parsed.canonical())


@app.get("/urn/{id:path}/info", response_model=None)
async def get_item_info(
    id: str,
    format: str = Query("json", description="Response format: 'json' or 'jsonld'"),
) -> ItemInfoResponse | JSONResponse:
    """Get item metadata from Internet Archive without downloading.

    Fetches metadata directly from Internet Archive /metadata/ endpoint.
    Does NOT download any files - only retrieves metadata.

    Args:
        id: URN (urn:ia:<identifier>) or raw identifier
        format: Response format - 'json' (default) or 'jsonld' (JSON-LD with schema.org/Dublin Core)

    This is useful for:
    - Checking if an item exists before downloading
    - Getting descriptions, identifiers, file lists
    - Building search indexes or catalogs
    - Linked data integration (with format=jsonld)
    """
    if _ia_client is None:
        raise error_response(503, "unavailable", "Service not initialized")

    parsed = _parse_and_validate_urn(id)
    identifier = parsed.identifier
    urn = parsed.canonical()

    try:
        meta = await _ia_client.fetch_metadata(identifier)
    except ItemNotFoundError as exc:
        raise error_response(404, "not_found", str(exc), urn=urn)
    except RateLimitedError as exc:
        raise error_response(429, "quota_exceeded", str(exc), urn=urn)
    except IAClientError as exc:
        raise error_response(502, "upstream_error", str(exc), urn=urn)

    # Update S3 metadata if item is already cached (backfill rich metadata)
    already_cached = False
    if _s3_storage is not None:
        meta_key = _s3_storage.meta_key(identifier)
        try:
            existing = json.loads(_s3_storage.download(meta_key))
        except Exception:
            existing = None

        if existing is not None:
            already_cached = True
            existing["ia"] = asdict(meta)
            base_url = get_cached_settings().base_url or ""
            existing["jsonld"] = meta.to_jsonld(urn, base_url)
            if meta.title:
                existing["title"] = meta.title
            if meta.creator:
                if isinstance(meta.creator, list):
                    existing["authors"] = ", ".join(meta.creator)
                else:
                    existing["authors"] = meta.creator
            if meta.publisher:
                existing["publisher"] = meta.publisher
            if meta.language:
                existing["language"] = meta.language
            if meta.item_size:
                b = meta.item_size
                existing["size"] = (
                    f"{b / 1024**3:.1f}GB" if b >= 1024**3
                    else f"{b / 1024**2:.1f}MB" if b >= 1024**2
                    else f"{b / 1024:.1f}KB" if b >= 1024
                    else f"{b} bytes"
                )
            existing["url"] = f"https://archive.org/details/{identifier}"
            _s3_storage.upload(meta_key, json.dumps(existing).encode(), "application/json")

    # Persist to PostgreSQL only if the book is already downloaded/cached
    if _db and already_cached:
        await _db.upsert_book(identifier, meta)

    # Return JSON-LD format if requested
    if format.lower() == "jsonld":
        settings = get_cached_settings()
        base_url = settings.base_url or ""
        jsonld = meta.to_jsonld(urn, base_url)
        return JSONResponse(
            content=jsonld,
            media_type="application/ld+json",
        )

    # Build file list for response
    files = [
        {"name": f.name, "format": f.format, "size": f.size}
        for f in meta.files[:50]  # Limit to 50 files
    ]

    # Handle description as list or string
    description = meta.description
    if isinstance(description, list):
        description = "\n".join(description)

    return ItemInfoResponse(
        urn=urn,
        identifier=identifier,
        title=meta.title,
        description=description,
        creator=meta.creator,
        date=meta.date,
        year=meta.year,
        subject=meta.subject,
        collection=meta.collection,
        language=meta.language,
        mediatype=meta.mediatype,
        item_size=meta.item_size,
        files_count=meta.files_count,
        downloads=meta.downloads,
        isbn=meta.isbn,
        oclc=meta.oclc,
        lccn=meta.lccn,
        external_identifier=meta.external_identifier,
        files=files,
    )


@app.get("/urn/{id:path}", response_model=I2LResponse)
async def resolve_i2l(
    id: str,
    redirect: bool = Query(False, description="If true, return 302 redirect instead of JSON"),
    format: str | None = Query(None, description="Preferred format"),
    r_format: str | None = Query(None, alias="+format", description="Format via r-component (?+format=)"),
) -> I2LResponse | RedirectResponse:
    """I2L: Resolve URN to a single URL.

    RFC 2483 I2L operation. Returns a URL where the resource can be accessed.
    If not cached, fetches from Internet Archive automatically.

    Format can be specified via:
    - URN r-component: urn:ia:item?+format=epub (RFC 8141)
    - Query param: ?format=epub (fallback)

    Args:
        id: URN (urn:ia:<identifier>) or raw identifier, optionally with r-component
        redirect: If true, returns HTTP 302 redirect to the URL
        format: Preferred format (fallback if not in URN r-component)
    """
    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    parsed = _parse_and_validate_urn(id)
    urn = parsed.canonical()
    explicit_format = parsed.format or r_format or format
    preferred_format = explicit_format or "pdf"

    # Ensure resource is cached (fetches from upstream if needed)
    filename, fmt = await _ensure_cached(
        parsed.identifier, urn, [preferred_format], format_required=bool(explicit_format)
    )

    key = _s3_storage.book_key(parsed.identifier, fmt)
    url = _s3_storage.get_presigned_url(key, filename)

    if redirect:
        return RedirectResponse(url=url, status_code=302)
    return I2LResponse(urn=urn, url=url)


@app.post("/download", response_model=BatchDownloadResponse)
@app.post("/items/download", response_model=BatchDownloadResponse, deprecated=True)
async def download_items_batch(request: BatchDownloadRequest) -> BatchDownloadResponse:
    """Download multiple items in parallel.

    Leverages free-threaded Python 3.13 for true parallel downloads.

    Args:
        request: Batch download request with list of items

    Returns:
        BatchDownloadResponse with results for each item
    """
    import asyncio

    if _ia_client is None or _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    start_time = time.monotonic()

    async def download_one(item: dict) -> tuple[str, DownloadResponse | dict]:
        identifier = item.get("identifier", "")
        if not identifier:
            return "", {"error": "malformed_uri", "detail": "Missing identifier"}

        preferred_formats = item.get("preferred_formats", ["pdf", "epub", "fb2", "mobi", "djvu"])
        specific_file = item.get("specific_file")

        try:
            req = DownloadRequest(preferred_formats=preferred_formats, specific_file=specific_file)
            response = await download_item_endpoint(identifier, req)
            return identifier, response
        except HTTPException as exc:
            return identifier, exc.detail if isinstance(exc.detail, dict) else {"error": "unknown", "detail": exc.detail}
        except Exception as exc:
            return identifier, {"error": "unknown", "detail": str(exc)}

    # Run all downloads concurrently (semaphore is in downloader)
    tasks = [download_one(item) for item in request.items]
    results_list = await asyncio.gather(*tasks)

    results = {}
    successful = 0
    failed = 0

    for identifier, result in results_list:
        if not identifier:
            failed += 1
            continue

        results[identifier] = result
        if isinstance(result, DownloadResponse):
            successful += 1
        else:
            failed += 1

    total_duration_ms = int((time.monotonic() - start_time) * 1000)

    return BatchDownloadResponse(
        results=results,
        duration_ms=total_duration_ms,
        successful=successful,
        failed=failed,
    )


def main():
    """Run the service with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "ia_dl.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
