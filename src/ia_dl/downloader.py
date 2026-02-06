"""Internet Archive file downloader with retry logic.

This module leverages Python 3.13's free-threaded mode to perform
truly parallel downloads without GIL contention.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .ia_client import IAClient, ItemMetadata, FileInfo
    from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a successful download."""

    content: bytes
    filename: str
    format: str
    identifier: str
    duration_ms: int
    size_bytes: int
    content_type: str = "application/octet-stream"


class DownloadError(Exception):
    """Download failed after all retry attempts."""

    def __init__(self, message: str, last_status: int | None = None):
        super().__init__(message)
        self.last_status = last_status


def normalize_ia_format(format_name: str) -> str:
    """Normalize IA format name to a file extension.

    IA uses verbose names like "Text PDF", "DjVu", "EPUB".
    This normalizes them to simple extensions: pdf, djvu, epub.
    """
    fmt = format_name.lower()

    # Document formats
    if "pdf" in fmt:
        return "pdf"
    if "djvu" in fmt:
        return "djvu"
    if "epub" in fmt:
        return "epub"
    if "mobi" in fmt or "kindle" in fmt:
        return "mobi"
    if fmt in ("txt", "text", "plain text", "djvutxt", "ocr search text"):
        return "txt"

    # Web/markup
    if fmt in ("html", "htm") or "html" in fmt:
        return "html"

    # Images
    if "jp2" in fmt:
        return "jp2"
    if fmt in ("jpeg", "jpg") or "jpeg" in fmt:
        return "jpg"
    if fmt == "png":
        return "png"
    if fmt == "gif" or "gif" in fmt:
        return "gif"

    # Audio/video
    if "mp3" in fmt:
        return "mp3"
    if "mp4" in fmt or "mpeg4" in fmt:
        return "mp4"
    if "ogv" in fmt or "ogg video" in fmt:
        return "ogv"
    if "ogg" in fmt:
        return "ogg"

    # Archives
    if fmt == "zip":
        return "zip"
    if fmt == "tar":
        return "tar"

    return "bin"


def detect_format(file_info: "FileInfo") -> str:
    """Detect file format from FileInfo.

    Uses the IA format field first (more reliable), falls back to filename extension.
    """
    # First try the format field from IA metadata
    if file_info.format:
        normalized = normalize_ia_format(file_info.format)
        if normalized != "bin":
            return normalized

    # Fall back to filename extension
    filename = file_info.name
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in ("pdf", "epub", "djvu", "mobi", "azw3", "fb2", "cbr", "cbz", "txt", "html", "htm", "jp2", "mp3", "mp4"):
            return ext

    return "bin"


async def download_item(
    client: "IAClient",
    settings: "Settings",
    identifier: str,
    preferred_formats: list[str] | None = None,
    specific_file: str | None = None,
) -> DownloadResult:
    """Download an item from Internet Archive.

    Fetches metadata first to find the best file, then downloads it.

    Args:
        client: Internet Archive API client
        settings: Service settings
        identifier: Internet Archive item identifier
        preferred_formats: List of formats in order of preference
        specific_file: If provided, download this specific file instead of auto-selecting

    Returns:
        DownloadResult with content bytes and metadata

    Raises:
        DownloadError: If download fails after retries
    """
    from .ia_client import ItemNotFoundError, RateLimitedError, IAClientError

    last_error: Exception | None = None
    last_status: int | None = None

    # Fetch metadata to find the file to download
    try:
        metadata = await client.fetch_metadata(identifier)
    except ItemNotFoundError as exc:
        raise DownloadError(f"Item not found: {identifier}", 404) from exc
    except RateLimitedError as exc:
        raise DownloadError(f"Rate limited: {exc}", 429) from exc
    except IAClientError as exc:
        raise DownloadError(f"Failed to fetch metadata: {exc}") from exc

    # Determine which file to download
    if specific_file:
        # Check if the specific file exists
        matching_files = [f for f in metadata.files if f.name == specific_file]
        if not matching_files:
            raise DownloadError(f"File not found: {identifier}/{specific_file}", 404)
        target_file = matching_files[0]
    else:
        # Auto-select best file based on format preferences
        target_file = metadata.get_best_file(preferred_formats)
        if not target_file:
            raise DownloadError(f"No downloadable files in item: {identifier}", 404)

    filename = target_file.name
    detected_format = detect_format(target_file)

    logger.info(
        "Downloading %s/%s (format=%s, size=%d bytes)",
        identifier,
        filename,
        detected_format,
        target_file.size,
    )

    # Download with retries
    for attempt in range(settings.max_retries):
        attempt_start = time.monotonic()

        try:
            content, content_type = await client.download_file(
                identifier,
                filename,
                timeout=settings.download_timeout,
            )

            duration_ms = int((time.monotonic() - attempt_start) * 1000)
            size_mb = len(content) / 1024 / 1024

            logger.info(
                "Download complete: %.2f MB in %d ms (attempt=%d)",
                size_mb,
                duration_ms,
                attempt + 1,
            )

            return DownloadResult(
                content=content,
                filename=filename,
                format=detected_format,
                identifier=identifier,
                duration_ms=duration_ms,
                size_bytes=len(content),
                content_type=content_type,
            )

        except ItemNotFoundError as exc:
            raise DownloadError(f"File not found: {identifier}/{filename}", 404) from exc

        except RateLimitedError as exc:
            logger.warning(
                "Rate limited (attempt=%d): %s",
                attempt + 1,
                exc,
            )
            last_error = exc
            last_status = 429
            # Longer backoff for rate limiting
            await asyncio.sleep(2.0 * (attempt + 1))
            continue

        except IAClientError as exc:
            logger.warning(
                "Download failed (attempt=%d): %s",
                attempt + 1,
                exc,
            )
            last_error = exc
            await asyncio.sleep(0.5 * (attempt + 1))
            continue

        except httpx.TimeoutException as exc:
            logger.warning(
                "Download timeout (attempt=%d): %s",
                attempt + 1,
                exc,
            )
            last_error = exc
            await asyncio.sleep(0.5 * (attempt + 1))
            continue

        except httpx.ConnectError as exc:
            logger.warning(
                "Connection failed (attempt=%d): %s",
                attempt + 1,
                exc,
            )
            last_error = exc
            await asyncio.sleep(0.5 * (attempt + 1))
            continue

    # All attempts failed
    raise DownloadError(
        f"All {settings.max_retries} download attempts failed: {last_error}",
        last_status,
    )


async def download_items_parallel(
    client: "IAClient",
    settings: "Settings",
    identifiers: list[str],
    preferred_formats: list[str] | None = None,
) -> dict[str, DownloadResult | DownloadError]:
    """Download multiple items in parallel.

    With free-threaded Python 3.13, this achieves true parallelism
    when combined with thread-based concurrency.

    Args:
        client: Internet Archive API client
        settings: Service settings
        identifiers: List of item identifiers to download
        preferred_formats: Optional list of preferred formats

    Returns:
        Dict mapping identifier -> DownloadResult or DownloadError
    """
    async def download_one(identifier: str) -> tuple[str, DownloadResult | DownloadError]:
        try:
            result = await download_item(
                client, settings, identifier, preferred_formats
            )
            return identifier, result
        except DownloadError as exc:
            return identifier, exc

    # Use semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)

    async def bounded_download(identifier: str) -> tuple[str, DownloadResult | DownloadError]:
        async with semaphore:
            return await download_one(identifier)

    # Run all downloads concurrently
    tasks = [bounded_download(id) for id in identifiers]
    results = await asyncio.gather(*tasks)

    return dict(results)
