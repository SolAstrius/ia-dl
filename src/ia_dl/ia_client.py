"""Internet Archive API client."""

import logging
from dataclasses import dataclass, field
from typing import Self

import httpx

logger = logging.getLogger(__name__)

# Internet Archive base URL
IA_BASE_URL = "https://archive.org"

# User agent for requests
DEFAULT_USER_AGENT = "ia-dl/0.1.0 (https://github.com/solastrius/ia-dl)"


@dataclass
class FileInfo:
    """Information about a file in an Internet Archive item.

    Maps to entries in the 'files' array from /metadata/{id}
    """

    name: str
    format: str
    size: int = 0
    md5: str = ""
    sha1: str = ""
    crc32: str = ""
    source: str = ""  # "original" or "derivative"
    original: str = ""  # For derivatives, the source file
    mtime: str = ""
    rotation: str = ""  # For images


@dataclass
class ItemMetadata:
    """Metadata fetched from Internet Archive API.

    Maps to the structure returned by https://archive.org/metadata/{identifier}
    This is a comprehensive model covering most IA metadata fields.
    """

    # === Core bibliographic fields ===
    identifier: str = ""
    title: str = ""
    creator: str | list[str] = ""  # Author(s)
    date: str = ""  # Publication date
    year: str = ""  # Publication year (extracted)
    publisher: str = ""
    description: str | list[str] = ""  # Can be multi-valued

    # === Classification ===
    subject: list[str] = field(default_factory=list)
    collection: list[str] = field(default_factory=list)
    language: str = ""
    mediatype: str = ""  # "texts", "audio", "movies", "software", "image", "data", "web", "collection", "account"

    # === External identifiers ===
    isbn: list[str] = field(default_factory=list)
    oclc: list[str] = field(default_factory=list)  # oclc-id in API
    lccn: list[str] = field(default_factory=list)
    issn: list[str] = field(default_factory=list)
    doi: list[str] = field(default_factory=list)
    ark: str = ""  # identifier-ark
    openlibrary_edition: str = ""
    openlibrary_work: str = ""

    # === Contributors ===
    contributor: str | list[str] = ""  # Digitizing organization
    sponsor: str = ""  # Digitization sponsor

    # === Licensing ===
    licenseurl: str = ""
    rights: str = ""
    possible_copyright_status: str = ""  # NOT_IN_COPYRIGHT, etc.
    copyright_region: str = ""

    # === Scanning/digitization info ===
    scanningcenter: str = ""
    scanner: str = ""
    scandate: str = ""
    ppi: str = ""  # Pixels per inch
    imagecount: str = ""
    camera: str = ""
    operator: str = ""
    repub_state: str = ""
    foldoutcount: str = ""
    bookplateleaf: str = ""

    # === OCR info ===
    ocr: str = ""  # OCR engine used (e.g., "tesseract 5.3.0")
    ocr_module_version: str = ""
    ocr_detected_lang: str = ""
    ocr_detected_lang_conf: str = ""
    ocr_detected_script: str = ""
    ocr_detected_script_conf: str = ""

    # === Upload/modification info ===
    uploader: str = ""
    addeddate: str = ""
    publicdate: str = ""
    updatedate: str = ""
    created: int = 0  # Unix timestamp
    item_last_updated: int = 0  # Unix timestamp

    # === Files ===
    files: list[FileInfo] = field(default_factory=list)
    files_count: int = 0
    item_size: int = 0  # Total bytes

    # === Download stats ===
    downloads: int = 0
    week: int = 0  # Downloads this week
    month: int = 0  # Downloads this month

    # === Server info ===
    server: str = ""  # Primary server (e.g., "ia903408.us.archive.org")
    dir: str = ""  # Path on server (e.g., "/6/items/identifier")
    d1: str = ""  # Primary data node
    d2: str = ""  # Secondary data node
    workable_servers: list[str] = field(default_factory=list)

    # === Raw metadata (for fields not explicitly modeled) ===
    _raw: dict = field(default_factory=dict, repr=False)

    def get_download_url(self, filename: str) -> str:
        """Get download URL for a specific file."""
        return f"{IA_BASE_URL}/download/{self.identifier}/{filename}"

    def get_details_url(self) -> str:
        """Get the archive.org details page URL."""
        return f"{IA_BASE_URL}/details/{self.identifier}"

    def get_best_file(self, preferred_formats: list[str] | None = None) -> FileInfo | None:
        """Get the best file to download based on format preference.

        Args:
            preferred_formats: List of formats in order of preference (e.g., ["pdf", "epub", "djvu"])

        Returns:
            Best matching FileInfo or None if no files
        """
        if not self.files:
            return None

        preferred_formats = preferred_formats or ["pdf", "epub", "djvu", "mobi", "txt"]

        def normalize_format(fmt: str) -> str:
            """Normalize IA format names to standard extensions."""
            fmt = fmt.lower()
            # IA uses descriptive names like "Text PDF", "DjVu", etc.
            if "pdf" in fmt:
                return "pdf"
            if "djvu" in fmt:
                return "djvu"
            if "epub" in fmt:
                return "epub"
            if "mobi" in fmt or "kindle" in fmt:
                return "mobi"
            if fmt in ("txt", "text", "plain text"):
                return "txt"
            if "daisy" in fmt:
                return "daisy"
            if "mp3" in fmt:
                return "mp3"
            if "mp4" in fmt or "mpeg4" in fmt:
                return "mp4"
            if "ogv" in fmt or "ogg video" in fmt:
                return "ogv"
            return fmt

        # First pass: original files in preferred format order
        for fmt in preferred_formats:
            for f in self.files:
                if normalize_format(f.format) == fmt and f.source == "original":
                    return f

        # Second pass: any file in preferred format order (including derivatives)
        for fmt in preferred_formats:
            for f in self.files:
                if normalize_format(f.format) == fmt:
                    return f

        # Fallback: largest original file (excluding metadata files)
        skip_formats = {"metadata", "item tile", "archive bittorrent", "sqlite"}
        originals = [
            f for f in self.files
            if f.source == "original" and f.format.lower() not in skip_formats
        ]
        if originals:
            return max(originals, key=lambda f: f.size)

        # Last resort: largest file (excluding metadata)
        content_files = [
            f for f in self.files
            if f.format.lower() not in skip_formats
        ]
        return max(content_files, key=lambda f: f.size) if content_files else None

    def get_files_by_format(self, format: str) -> list[FileInfo]:
        """Get all files matching a format."""
        format_lower = format.lower()
        return [f for f in self.files if format_lower in f.format.lower()]

    def to_jsonld(self, urn: str, base_url: str = "") -> dict:
        """Convert to JSON-LD format using schema.org + Dublin Core.

        Args:
            urn: The URN identifier (e.g., "urn:ia:taleoftwocities00dick")
            base_url: Optional base URL for resource links

        Returns:
            JSON-LD dict compatible with schema.org and Dublin Core
        """
        # Map mediatype to schema.org types
        schema_type_map = {
            "texts": "Book",
            "audio": "AudioObject",
            "movies": "VideoObject",
            "software": "SoftwareApplication",
            "image": "ImageObject",
            "data": "Dataset",
            "web": "WebPage",
            "collection": "Collection",
        }
        schema_type = schema_type_map.get(self.mediatype, "CreativeWork")

        doc: dict = {
            "@context": {
                "@vocab": "https://schema.org/",
                "dc": "http://purl.org/dc/terms/",
                "dcterms": "http://purl.org/dc/terms/",
                "bibo": "http://purl.org/ontology/bibo/",
            },
            "@type": schema_type,
            "@id": urn,
        }

        # === sameAs: resolvable identifiers ===
        same_as = [self.get_details_url()]

        if self.ark:
            same_as.append(f"https://n2t.net/{self.ark}")
        for isbn in self.isbn:
            same_as.append(f"urn:isbn:{isbn}")
        for oclc_id in self.oclc:
            same_as.append(f"urn:oclc:{oclc_id}")
            same_as.append(f"https://www.worldcat.org/oclc/{oclc_id}")
        for lccn_id in self.lccn:
            same_as.append(f"urn:lccn:{lccn_id}")
        if self.openlibrary_edition:
            same_as.append(f"https://openlibrary.org/books/{self.openlibrary_edition}")
        if self.openlibrary_work:
            same_as.append(f"https://openlibrary.org/works/{self.openlibrary_work}")
        for d in self.doi:
            same_as.append(f"https://doi.org/{d}")

        doc["sameAs"] = same_as

        # === Schema.org fields ===
        if self.title:
            doc["name"] = self.title

        if self.creator:
            creators = self.creator if isinstance(self.creator, list) else [self.creator]
            if len(creators) == 1:
                doc["author"] = {"@type": "Person", "name": creators[0]}
            else:
                doc["author"] = [{"@type": "Person", "name": c} for c in creators]

        if self.publisher:
            doc["publisher"] = {"@type": "Organization", "name": self.publisher}

        if self.date or self.year:
            doc["datePublished"] = self.date or self.year

        if self.description:
            desc = self.description
            if isinstance(desc, list):
                desc = "\n".join(desc)
            doc["description"] = desc

        if self.language:
            doc["inLanguage"] = self.language

        if self.subject:
            doc["keywords"] = self.subject

        if self.licenseurl:
            doc["license"] = self.licenseurl

        # File info
        best_file = self.get_best_file()
        if best_file:
            doc["encodingFormat"] = best_file.format
            if best_file.size:
                doc["contentSize"] = f"{best_file.size} bytes"

        if self.isbn:
            doc["isbn"] = self.isbn[0] if len(self.isbn) == 1 else self.isbn

        # === Dublin Core ===
        if self.title:
            doc["dc:title"] = self.title
        if self.creator:
            doc["dc:creator"] = self.creator
        if self.publisher:
            doc["dc:publisher"] = self.publisher
        if self.date or self.year:
            doc["dc:date"] = self.date or self.year
        if self.language:
            doc["dc:language"] = self.language
        if self.description:
            doc["dc:description"] = self.description if isinstance(self.description, str) else self.description[0]
        if self.rights:
            doc["dc:rights"] = self.rights
        if self.contributor:
            doc["dc:contributor"] = self.contributor

        # DC identifiers
        dc_identifiers = [f"urn:ia:{self.identifier}"]
        if self.ark:
            dc_identifiers.append(self.ark)
        for isbn in self.isbn:
            dc_identifiers.append(f"urn:isbn:{isbn}")
        for oclc_id in self.oclc:
            dc_identifiers.append(f"oclc:{oclc_id}")
        doc["dc:identifier"] = dc_identifiers

        # === Structured identifiers ===
        identifiers = [
            {"@type": "PropertyValue", "propertyID": "ia", "value": self.identifier}
        ]
        if self.ark:
            identifiers.append({"@type": "PropertyValue", "propertyID": "ark", "value": self.ark})
        for isbn in self.isbn:
            identifiers.append({"@type": "PropertyValue", "propertyID": "isbn", "value": isbn})
        for oclc_id in self.oclc:
            identifiers.append({"@type": "PropertyValue", "propertyID": "oclc", "value": oclc_id})
        for lccn_id in self.lccn:
            identifiers.append({"@type": "PropertyValue", "propertyID": "lccn", "value": lccn_id})
        doc["identifier"] = identifiers

        if base_url:
            doc["url"] = f"{base_url}/urn/{urn}"

        return doc


class IAClientError(Exception):
    """Error from Internet Archive API."""
    pass


class ItemNotFoundError(IAClientError):
    """Item not found in Internet Archive (404)."""
    pass


class RateLimitedError(IAClientError):
    """Rate limited by Internet Archive (429/503)."""
    pass


class IAClient:
    """Async client for Internet Archive."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        timeout: float = 30.0,
        access_key: str | None = None,
        secret_key: str | None = None,
    ):
        self._http = http
        self._timeout = timeout
        self._access_key = access_key
        self._secret_key = secret_key

    @classmethod
    def create(
        cls,
        timeout: float = 30.0,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> Self:
        """Create a client with a new HTTP client."""
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        return cls(http, timeout, access_key, secret_key)

    def _auth_headers(self) -> dict[str, str]:
        """Get authentication headers if credentials are configured."""
        if self._access_key and self._secret_key:
            return {"Authorization": f"LOW {self._access_key}:{self._secret_key}"}
        return {}

    async def fetch_metadata(self, identifier: str) -> ItemMetadata:
        """Fetch metadata for an Internet Archive item.

        Args:
            identifier: Internet Archive item identifier

        Returns:
            ItemMetadata with all available fields

        Raises:
            ItemNotFoundError: If item doesn't exist
            RateLimitedError: If rate limited
            IAClientError: For other errors
        """
        url = f"{IA_BASE_URL}/metadata/{identifier}"

        logger.debug("Fetching metadata: %s", url)

        try:
            response = await self._http.get(
                url,
                headers=self._auth_headers(),
                timeout=self._timeout,
            )

            if response.status_code == 404:
                raise ItemNotFoundError(f"Item not found: {identifier}")
            if response.status_code in (429, 503):
                raise RateLimitedError(f"Rate limited fetching {identifier}")

            response.raise_for_status()
            data = response.json()

            # Check for error or empty response
            if data.get("error"):
                raise IAClientError(f"API error: {data['error']}")
            if not data:
                raise ItemNotFoundError(f"Empty response for: {identifier}")

            metadata = data.get("metadata", {})
            files_data = data.get("files", [])

            # Parse files
            files = []
            for f in files_data:
                files.append(FileInfo(
                    name=f.get("name", ""),
                    format=f.get("format", ""),
                    size=int(f.get("size", 0) or 0),
                    md5=f.get("md5", ""),
                    sha1=f.get("sha1", ""),
                    crc32=f.get("crc32", ""),
                    source=f.get("source", ""),
                    original=f.get("original", ""),
                    mtime=f.get("mtime", ""),
                    rotation=f.get("rotation", ""),
                ))

            def as_list(val) -> list:
                """Convert value to list, handling None and single values."""
                if val is None:
                    return []
                if isinstance(val, list):
                    return val
                return [val]

            def as_str(val) -> str:
                """Convert value to string, joining lists."""
                if val is None:
                    return ""
                if isinstance(val, list):
                    return val[0] if val else ""
                return str(val)

            return ItemMetadata(
                # Core bibliographic
                identifier=metadata.get("identifier", identifier),
                title=as_str(metadata.get("title")),
                creator=metadata.get("creator", ""),
                date=as_str(metadata.get("date")),
                year=as_str(metadata.get("year")),
                publisher=as_str(metadata.get("publisher")),
                description=metadata.get("description", ""),

                # Classification
                subject=as_list(metadata.get("subject")),
                collection=as_list(metadata.get("collection")),
                language=as_str(metadata.get("language")),
                mediatype=as_str(metadata.get("mediatype")),

                # External identifiers
                isbn=as_list(metadata.get("isbn")),
                oclc=as_list(metadata.get("oclc-id")),
                lccn=as_list(metadata.get("lccn")),
                issn=as_list(metadata.get("issn")),
                doi=as_list(metadata.get("doi")),
                ark=as_str(metadata.get("identifier-ark")),
                openlibrary_edition=as_str(metadata.get("openlibrary_edition")),
                openlibrary_work=as_str(metadata.get("openlibrary_work")),

                # Contributors
                contributor=metadata.get("contributor", ""),
                sponsor=as_str(metadata.get("sponsor")),

                # Licensing
                licenseurl=as_str(metadata.get("licenseurl")),
                rights=as_str(metadata.get("rights")),
                possible_copyright_status=as_str(metadata.get("possible-copyright-status")),
                copyright_region=as_str(metadata.get("copyright-region")),

                # Scanning info
                scanningcenter=as_str(metadata.get("scanningcenter")),
                scanner=as_str(metadata.get("scanner")),
                scandate=as_str(metadata.get("scandate")),
                ppi=as_str(metadata.get("ppi")),
                imagecount=as_str(metadata.get("imagecount")),
                camera=as_str(metadata.get("camera")),
                operator=as_str(metadata.get("operator")),
                repub_state=as_str(metadata.get("repub_state")),
                foldoutcount=as_str(metadata.get("foldoutcount")),
                bookplateleaf=as_str(metadata.get("bookplateleaf")),

                # OCR info
                ocr=as_str(metadata.get("ocr")),
                ocr_module_version=as_str(metadata.get("ocr_module_version")),
                ocr_detected_lang=as_str(metadata.get("ocr_detected_lang")),
                ocr_detected_lang_conf=as_str(metadata.get("ocr_detected_lang_conf")),
                ocr_detected_script=as_str(metadata.get("ocr_detected_script")),
                ocr_detected_script_conf=as_str(metadata.get("ocr_detected_script_conf")),

                # Upload/modification
                uploader=as_str(metadata.get("uploader")),
                addeddate=as_str(metadata.get("addeddate")),
                publicdate=as_str(metadata.get("publicdate")),
                updatedate=as_str(metadata.get("updatedate")),
                created=int(data.get("created", 0) or 0),
                item_last_updated=int(data.get("item_last_updated", 0) or 0),

                # Files
                files=files,
                files_count=int(data.get("files_count", 0) or 0),
                item_size=int(data.get("item_size", 0) or 0),

                # Download stats
                downloads=int(metadata.get("downloads", 0) or 0),
                week=int(data.get("week", 0) or 0),
                month=int(data.get("month", 0) or 0),

                # Server info
                server=as_str(data.get("server")),
                dir=as_str(data.get("dir")),
                d1=as_str(data.get("d1")),
                d2=as_str(data.get("d2")),
                workable_servers=as_list(data.get("workable_servers")),

                # Raw metadata for anything not explicitly modeled
                _raw=metadata,
            )

        except httpx.HTTPStatusError as exc:
            raise IAClientError(f"HTTP error: {exc}") from exc
        except httpx.RequestError as exc:
            raise IAClientError(f"Request failed: {exc}") from exc

    async def download_file(
        self,
        identifier: str,
        filename: str,
        timeout: float | None = None,
    ) -> tuple[bytes, str]:
        """Download a file from Internet Archive.

        Args:
            identifier: Internet Archive item identifier
            filename: Name of file within the item
            timeout: Optional custom timeout for large files

        Returns:
            Tuple of (content bytes, content-type)

        Raises:
            ItemNotFoundError: If item or file doesn't exist
            RateLimitedError: If rate limited
            IAClientError: For other errors
        """
        url = f"{IA_BASE_URL}/download/{identifier}/{filename}"

        logger.debug("Downloading: %s", url)

        try:
            response = await self._http.get(
                url,
                headers=self._auth_headers(),
                timeout=timeout or self._timeout,
            )

            if response.status_code == 404:
                raise ItemNotFoundError(f"File not found: {identifier}/{filename}")
            if response.status_code in (429, 503):
                raise RateLimitedError(f"Rate limited downloading {identifier}/{filename}")

            response.raise_for_status()

            content_type = response.headers.get("content-type", "application/octet-stream")
            return response.content, content_type

        except httpx.HTTPStatusError as exc:
            raise IAClientError(f"HTTP error: {exc}") from exc
        except httpx.RequestError as exc:
            raise IAClientError(f"Request failed: {exc}") from exc

    async def close(self):
        """Close the HTTP client."""
        await self._http.aclose()
