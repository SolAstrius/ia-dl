"""Anna's Archive API client with mirror failover and DDoS-Guard bypass."""

import logging
from dataclasses import dataclass, field
from typing import Self

import httpx

logger = logging.getLogger(__name__)

# Default user agent (will be replaced by FlareSolverr's)
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0"

# Available Anna's Archive mirror domains (in order of preference)
MIRROR_DOMAINS = [
    "https://annas-archive.li",
    "https://annas-archive.pm",
    "https://annas-archive.in",
]


@dataclass
class BookMetadata:
    """Metadata fetched from Anna's Archive API (/db/aarecord_elasticsearch/).

    All fields mirror the file_unified_data structure from the API.
    Lists default to empty, strings default to empty, ints default to 0.
    """

    # Core fields - always present as strings (may be empty)
    title_best: str = ""
    author_best: str = ""
    publisher_best: str = ""
    extension_best: str = ""  # format: epub, pdf, etc.
    year_best: str = ""

    # Additional/alternate values - always lists (may be empty)
    title_additional: list[str] = field(default_factory=list)
    author_additional: list[str] = field(default_factory=list)
    publisher_additional: list[str] = field(default_factory=list)
    extension_additional: list[str] = field(default_factory=list)
    year_additional: list[str] = field(default_factory=list)

    # Language - always lists
    language_codes: list[str] = field(default_factory=list)
    language_codes_detected: list[str] = field(default_factory=list)
    most_likely_language_codes: list[str] = field(default_factory=list)

    # Size
    filesize_best: int = 0
    filesize_additional: list[int] = field(default_factory=list)

    # Content info
    content_type_best: str = ""  # book_fiction, book_nonfiction, magazine
    stripped_description_best: str = ""
    stripped_description_additional: list[str] = field(default_factory=list)

    # Cover images
    cover_url_best: str = ""
    cover_url_additional: list[str] = field(default_factory=list)

    # Edition info
    edition_varia_best: str = ""
    edition_varia_additional: list[str] = field(default_factory=list)

    # Dates
    added_date_best: str = ""
    added_date_unified: dict[str, str] = field(default_factory=dict)

    # Identifiers (isbn, doi, oclc, etc.)
    identifiers_unified: dict[str, list[str]] = field(default_factory=dict)

    # IPFS - list of {"ipfs_cid": str, "from": str}
    ipfs_infos: list[dict[str, str]] = field(default_factory=list)

    # Availability flags - always ints
    has_aa_downloads: int = 0
    has_aa_exclusive_downloads: int = 0
    has_torrent_paths: int = 0
    has_scidb: int = 0

    # Problems/issues - always list/int
    problems: list[dict] = field(default_factory=list)
    has_meaningful_problems: int = 0

    # Other
    original_filename_best: str = ""
    comments_multiple: list[str] = field(default_factory=list)
    classifications_unified: dict = field(default_factory=dict)
    ol_is_primary_linked: bool = False


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


class DDoSGuardError(AnnasClientError):
    """DDoS-Guard challenge encountered."""


@dataclass
class Session:
    """Authenticated session with Anna's Archive including DDoS-Guard cookies."""

    account_id: str
    cookies: dict[str, str] = field(default_factory=dict)
    user_agent: str = DEFAULT_USER_AGENT

    def as_cookie_header(self) -> str:
        """Return cookie header value for requests."""
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    def as_cookies_dict(self) -> dict[str, str]:
        """Return cookies dict for httpx."""
        return self.cookies.copy()


class AnnasClient:
    """Async client for Anna's Archive with automatic mirror failover and DDoS-Guard bypass."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        timeout: float = 15.0,
        flaresolverr_url: str | None = None,
        secret_key: str | None = None,
    ):
        self._http = http
        self._timeout = timeout
        self._flaresolverr_url = flaresolverr_url
        self._secret_key = secret_key
        self._current_domain_idx = 0
        self._session: Session | None = None

    @classmethod
    def create(
        cls,
        timeout: float = 15.0,
        flaresolverr_url: str | None = None,
        secret_key: str | None = None,
    ) -> Self:
        """Create a client with a new HTTP client."""
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        return cls(http, timeout, flaresolverr_url, secret_key)

    @property
    def current_domain(self) -> str:
        """Get the current active domain."""
        return MIRROR_DOMAINS[self._current_domain_idx % len(MIRROR_DOMAINS)]

    @property
    def session(self) -> Session | None:
        """Get current session if authenticated."""
        return self._session

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

    @staticmethod
    def _is_ddos_guard_challenge(response: httpx.Response) -> bool:
        """Check if response is a DDoS-Guard JS challenge."""
        if response.status_code != 403:
            return False
        # DDoS-Guard returns HTML with challenge script
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return False
        return "ddos-guard" in response.text.lower()

    @staticmethod
    def _is_auth_required(response: httpx.Response) -> bool:
        """Check if response indicates authentication is required."""
        if response.status_code != 403:
            return False
        # "Not a member" error from Anna's Archive
        return "not a member" in response.text.lower()

    async def _refresh_session_via_flaresolverr(self) -> Session:
        """Use FlareSolverr to bypass DDoS-Guard and login.

        FlareSolverr handles the JS challenge, then we POST the login form
        to get both DDoS-Guard cookies and the session cookie.
        """
        if not self._flaresolverr_url:
            raise DDoSGuardError("DDoS-Guard challenge but no FlareSolverr configured")
        if not self._secret_key:
            raise DDoSGuardError("DDoS-Guard challenge but no secret key configured")

        logger.info("Refreshing session via FlareSolverr at %s", self._flaresolverr_url)

        # POST login form via FlareSolverr - this bypasses DDoS-Guard and logs in
        try:
            response = await self._http.post(
                f"{self._flaresolverr_url}/v1",
                json={
                    "cmd": "request.post",
                    "url": f"{self.current_domain}/account/",
                    "postData": f"key={self._secret_key}",
                    "maxTimeout": 60000,
                },
                timeout=90.0,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise DDoSGuardError(f"FlareSolverr error: {data.get('message')}")

            solution = data.get("solution", {})
            cookies_list = solution.get("cookies", [])
            user_agent = solution.get("userAgent", DEFAULT_USER_AGENT)

            # Convert cookies list to dict
            cookies = {c["name"]: c["value"] for c in cookies_list}

            if "aa_account_id2" not in cookies:
                raise LoginError("Login via FlareSolverr did not return session cookie")

            # Extract account_id from secret_key (first 7 chars)
            account_id = self._secret_key[:7]

            self._session = Session(
                account_id=account_id,
                cookies=cookies,
                user_agent=user_agent,
            )

            logger.info(
                "Session refreshed for account %s with %d cookies",
                account_id,
                len(cookies),
            )
            return self._session

        except httpx.HTTPError as exc:
            raise DDoSGuardError(f"FlareSolverr request failed: {exc}") from exc

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """Make a request with session cookies, auto-refreshing on DDoS-Guard challenge."""
        # Apply session cookies and user-agent if we have a session
        if self._session:
            headers = kwargs.pop("headers", {})
            headers["Cookie"] = self._session.as_cookie_header()
            headers["User-Agent"] = self._session.user_agent
            kwargs["headers"] = headers

        response = await self._http.request(method, url, **kwargs)

        # Check for DDoS-Guard challenge or auth required (when no session)
        needs_refresh = (
            self._is_ddos_guard_challenge(response)
            or (self._is_auth_required(response) and not self._session)
        )

        if needs_refresh:
            if self._is_ddos_guard_challenge(response):
                logger.warning("DDoS-Guard challenge detected, refreshing session")
            else:
                logger.warning("Auth required, refreshing session via FlareSolverr")

            await self._refresh_session_via_flaresolverr()

            # Retry with new session
            if self._session:
                headers = kwargs.pop("headers", {})
                headers["Cookie"] = self._session.as_cookie_header()
                headers["User-Agent"] = self._session.user_agent
                kwargs["headers"] = headers
            response = await self._http.request(method, url, **kwargs)

        return response

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
                response = await self._request("GET", url, timeout=self._timeout)
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

            except DDoSGuardError:
                raise
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
        """Fetch metadata by hash with automatic failover and DDoS-Guard bypass."""
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/db/aarecord_elasticsearch/md5:{hash}.json"

            logger.debug("Fetching metadata (attempt %d): %s", attempt + 1, url)

            try:
                response = await self._request("GET", url, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()

                unified = data.get("file_unified_data", {})

                return BookMetadata(
                    # Core fields
                    title_best=unified.get("title_best", ""),
                    author_best=unified.get("author_best", ""),
                    publisher_best=unified.get("publisher_best", ""),
                    extension_best=unified.get("extension_best", ""),
                    year_best=unified.get("year_best", ""),
                    # Additional values
                    title_additional=unified.get("title_additional") or [],
                    author_additional=unified.get("author_additional") or [],
                    publisher_additional=unified.get("publisher_additional") or [],
                    extension_additional=unified.get("extension_additional") or [],
                    year_additional=unified.get("year_additional") or [],
                    # Language
                    language_codes=unified.get("language_codes") or [],
                    language_codes_detected=unified.get("language_codes_detected") or [],
                    most_likely_language_codes=unified.get("most_likely_language_codes") or [],
                    # Size
                    filesize_best=unified.get("filesize_best") or 0,
                    filesize_additional=unified.get("filesize_additional") or [],
                    # Content info
                    content_type_best=unified.get("content_type_best") or "",
                    stripped_description_best=unified.get("stripped_description_best") or "",
                    stripped_description_additional=unified.get("stripped_description_additional") or [],
                    # Cover images
                    cover_url_best=unified.get("cover_url_best") or "",
                    cover_url_additional=unified.get("cover_url_additional") or [],
                    # Edition
                    edition_varia_best=unified.get("edition_varia_best") or "",
                    edition_varia_additional=unified.get("edition_varia_additional") or [],
                    # Dates
                    added_date_best=unified.get("added_date_best") or "",
                    added_date_unified=unified.get("added_date_unified") or {},
                    # Identifiers
                    identifiers_unified=unified.get("identifiers_unified") or {},
                    # IPFS
                    ipfs_infos=unified.get("ipfs_infos") or [],
                    # Availability flags
                    has_aa_downloads=unified.get("has_aa_downloads") or 0,
                    has_aa_exclusive_downloads=unified.get("has_aa_exclusive_downloads") or 0,
                    has_torrent_paths=unified.get("has_torrent_paths") or 0,
                    has_scidb=unified.get("has_scidb") or 0,
                    # Problems
                    problems=unified.get("problems") or [],
                    has_meaningful_problems=unified.get("has_meaningful_problems") or 0,
                    # Other
                    original_filename_best=unified.get("original_filename_best") or "",
                    comments_multiple=unified.get("comments_multiple") or [],
                    classifications_unified=unified.get("classifications_unified") or {},
                    ol_is_primary_linked=bool(unified.get("ol_is_primary_linked")),
                )

            except DDoSGuardError:
                raise
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

        Note: This direct login will fail if DDoS-Guard is active.
        Use FlareSolverr integration for automatic bypass.

        Args:
            secret_key: Anna's Archive account secret key

        Returns:
            Session object with cookies for authenticated requests

        Raises:
            LoginError: If login fails (invalid key, network error, etc.)
            DDoSGuardError: If DDoS-Guard challenge is encountered
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

                # Check for DDoS-Guard challenge
                if self._is_ddos_guard_challenge(response):
                    raise DDoSGuardError(
                        "DDoS-Guard challenge during login. "
                        "Configure FlareSolverr for automatic bypass."
                    )

                # Successful login returns 302 redirect with Set-Cookie
                if response.status_code not in (302, 303):
                    # Check if we got an error page (200 with invalid_key message)
                    if response.status_code == 200:
                        raise LoginError("Invalid secret key")
                    raise LoginError(f"Unexpected status code: {response.status_code}")

                # Extract all cookies from response
                cookies = dict(response.cookies.items())

                if not any(name.startswith("aa_") for name in cookies):
                    raise LoginError("No session cookie in response")

                # Extract account_id from secret_key (first 7 chars)
                account_id = secret_key[:7]

                logger.info("Login successful for account %s via %s", account_id, domain)

                self._session = Session(
                    account_id=account_id,
                    cookies=cookies,
                )
                return self._session

            except LoginError | DDoSGuardError:
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
