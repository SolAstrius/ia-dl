"""Anna's Archive API client with mirror failover."""

import logging
from dataclasses import dataclass
from typing import Self

import httpx

logger = logging.getLogger(__name__)

# Available Anna's Archive mirror domains (in order of preference)
MIRROR_DOMAINS = [
    "https://annas-archive.li",
    "https://annas-archive.pm",
    "https://annas-archive.in",
]


@dataclass
class BookMetadata:
    """Metadata fetched from Anna's Archive API."""

    title: str
    author: str
    publisher: str
    format: str
    year: str
    language: str | None = None
    size: str | None = None
    ipfs_cid: str | None = None


@dataclass
class FastDownloadResult:
    """Result from fast download API including quota info."""

    download_url: str
    downloads_left: int
    downloads_per_day: int
    downloads_done_today: int


class AnnasClientError(Exception):
    """Error from Anna's Archive API."""

    # Known error messages from Anna's Archive API
    # See: https://software.annas-archive.li/AnnaArchivist/annas-archive/-/raw/main/allthethings/dyn/views.py
    ERROR_INVALID_MD5 = "Invalid md5"
    ERROR_INVALID_KEY = "Invalid secret key"
    ERROR_FETCH_ERROR = "Error during fetching"
    ERROR_NOT_FOUND = "Record not found"
    ERROR_NOT_MEMBER = "Not a member"
    ERROR_INVALID_INDICES = "Invalid domain_index or path_index"
    ERROR_NO_DOWNLOADS = "No downloads left"


class NoDownloadsLeftError(AnnasClientError):
    """Fast downloads exhausted (429)."""


class InvalidKeyError(AnnasClientError):
    """Invalid API secret key (401)."""


class NotMemberError(AnnasClientError):
    """Account is not a member (403)."""


class RecordNotFoundError(AnnasClientError):
    """Book not found in Anna's Archive (404)."""


class LoginError(AnnasClientError):
    """Failed to log in with secret key."""


@dataclass
class Session:
    """Authenticated session with Anna's Archive."""

    account_id: str
    cookie_name: str
    cookie_value: str

    def as_cookie_header(self) -> str:
        """Return cookie header value for requests."""
        return f"{self.cookie_name}={self.cookie_value}"

    def as_cookies_dict(self) -> dict[str, str]:
        """Return cookies dict for httpx."""
        return {self.cookie_name: self.cookie_value}


class AnnasClient:
    """Async client for Anna's Archive with automatic mirror failover."""

    def __init__(self, http: httpx.AsyncClient, timeout: float = 15.0):
        self._http = http
        self._timeout = timeout
        self._current_domain_idx = 0

    @classmethod
    def create(cls, timeout: float = 15.0) -> Self:
        """Create a client with a new HTTP client."""
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0"
            },
        )
        return cls(http, timeout)

    @property
    def current_domain(self) -> str:
        """Get the current active domain."""
        return MIRROR_DOMAINS[self._current_domain_idx % len(MIRROR_DOMAINS)]

    def _rotate_domain(self) -> str:
        """Rotate to the next domain, returns the new domain."""
        old_domain = self.current_domain
        self._current_domain_idx = (self._current_domain_idx + 1) % len(MIRROR_DOMAINS)
        new_domain = self.current_domain
        logger.info(
            "Rotating Anna's Archive mirror: %s -> %s", old_domain, new_domain
        )
        return new_domain

    @staticmethod
    def _is_recoverable_error(exc: Exception) -> bool:
        """Check if an error is recoverable by trying another domain."""
        if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        return False

    async def get_download_url(
        self, hash: str, secret_key: str, domain_index: int | None = None
    ) -> FastDownloadResult:
        """Get download URL for a book with automatic failover.

        Args:
            hash: MD5 hash of the book
            secret_key: Anna's Archive API key
            domain_index: Optional CDN server index (0-9+)

        Returns:
            FastDownloadResult with URL and quota info
        """
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/dyn/api/fast_download.json?md5={hash}&key={secret_key}"
            if domain_index is not None:
                url += f"&domain_index={domain_index}"

            logger.debug(
                "Fetching download URL (attempt %d): %s",
                attempt + 1,
                url.replace(secret_key, "***"),
            )

            try:
                response = await self._http.get(url, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()

                if error := data.get("error"):
                    # Raise specific exceptions for known error types
                    if error == AnnasClientError.ERROR_NO_DOWNLOADS:
                        raise NoDownloadsLeftError(error)
                    if error == AnnasClientError.ERROR_INVALID_KEY:
                        raise InvalidKeyError(error)
                    if error == AnnasClientError.ERROR_NOT_MEMBER:
                        raise NotMemberError(error)
                    if error == AnnasClientError.ERROR_NOT_FOUND:
                        raise RecordNotFoundError(error)
                    # Generic error for others
                    raise AnnasClientError(f"API error: {error}")

                download_url = data.get("download_url")
                if not download_url:
                    raise AnnasClientError("No download URL in response")

                # Extract quota info
                quota = data.get("account_fast_download_info", {})
                return FastDownloadResult(
                    download_url=download_url,
                    downloads_left=quota.get("downloads_left", 0),
                    downloads_per_day=quota.get("downloads_per_day", 0),
                    downloads_done_today=quota.get("downloads_done_today", 0),
                )

            except Exception as exc:
                logger.warning(
                    "Download URL request failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise AnnasClientError(str(exc)) from exc

        raise AnnasClientError(
            f"All mirrors failed: {last_error}"
        ) from last_error

    async def fetch_metadata(self, hash: str) -> BookMetadata:
        """Fetch metadata by hash with automatic failover."""
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/db/aarecord_elasticsearch/md5:{hash}.json"

            logger.debug("Fetching metadata (attempt %d): %s", attempt + 1, url)

            try:
                response = await self._http.get(url, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()

                unified = data.get("file_unified_data", {})

                # Extract IPFS CID if available
                ipfs_infos = unified.get("ipfs_infos", [])
                ipfs_cid = ipfs_infos[0].get("ipfs_cid") if ipfs_infos else None

                # Extract language
                lang_codes = unified.get("language_codes", [])
                language = lang_codes[0] if lang_codes else None

                # Format file size
                size_bytes = unified.get("filesize_best")
                size = _format_file_size(size_bytes) if size_bytes else None

                return BookMetadata(
                    title=unified.get("title_best", ""),
                    author=unified.get("author_best", ""),
                    publisher=unified.get("publisher_best", ""),
                    format=unified.get("extension_best", ""),
                    year=unified.get("year_best", ""),
                    language=language,
                    size=size,
                    ipfs_cid=ipfs_cid,
                )

            except Exception as exc:
                logger.warning(
                    "Metadata request failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise AnnasClientError(str(exc)) from exc

        raise AnnasClientError(
            f"All mirrors failed: {last_error}"
        ) from last_error

    async def login(self, secret_key: str) -> Session:
        """Log in with secret key and obtain session cookie.

        Posts the secret key to /account and extracts the session cookie
        from the response. This enables access to cookie-authenticated
        endpoints (lists, comments, account settings, etc.).

        Args:
            secret_key: Anna's Archive account secret key

        Returns:
            Session object with cookie for authenticated requests

        Raises:
            LoginError: If login fails (invalid key, network error, etc.)
        """
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/account/"

            logger.debug("Login attempt %d to %s", attempt + 1, domain)

            try:
                # POST the secret key as form data
                response = await self._http.post(
                    url,
                    data={"key": secret_key},
                    timeout=self._timeout,
                    follow_redirects=False,  # We want to capture the Set-Cookie header
                )

                # Successful login returns 302 redirect with Set-Cookie
                if response.status_code not in (302, 303):
                    # Check if we got an error page (200 with invalid_key message)
                    if response.status_code == 200:
                        raise LoginError("Invalid secret key")
                    raise LoginError(f"Unexpected status code: {response.status_code}")

                # Extract the session cookie
                # Cookie name is typically "aa_account_id2" based on the source
                cookie_name = None
                cookie_value = None

                for name, value in response.cookies.items():
                    # Look for the account cookie (starts with "aa_")
                    if name.startswith("aa_"):
                        cookie_name = name
                        cookie_value = value
                        break

                if not cookie_name or not cookie_value:
                    raise LoginError("No session cookie in response")

                # Extract account_id from secret_key (first 7 chars)
                account_id = secret_key[:7]

                logger.info("Login successful for account %s via %s", account_id, domain)

                return Session(
                    account_id=account_id,
                    cookie_name=cookie_name,
                    cookie_value=cookie_value,
                )

            except LoginError:
                raise
            except Exception as exc:
                logger.warning(
                    "Login failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise LoginError(str(exc)) from exc

        raise LoginError(f"All mirrors failed: {last_error}") from last_error

    async def close(self):
        """Close the HTTP client."""
        await self._http.aclose()


def _format_file_size(bytes_: int) -> str | None:
    """Format file size in human-readable form."""
    if bytes_ <= 0:
        return None
    for unit, threshold in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
        if bytes_ >= threshold:
            return f"{bytes_ / threshold:.1f} {unit}"
    return f"{bytes_} bytes"
