"""PostgreSQL integration for persisting book metadata to the shared annas-mcp database.

Writes to the same `books` and `book_identifiers` tables defined in
annas-mcp-rs/k8s/init.sql. Connection is optional — if DATABASE_URL is not
configured, all operations silently no-op.

All queries use asyncpg prepared statements ($1, $2, ...) — values are never
string-interpolated into SQL.
"""

import datetime
import json
import logging
import re
from dataclasses import asdict

import asyncpg

from .ia_client import ItemMetadata

logger = logging.getLogger(__name__)

# Identifier types worth persisting
_ID_TYPES = {
    "isbn", "oclc", "lccn", "issn", "doi", "ark",
    "openlibrary_edition", "openlibrary_work",
}

# ISO 639-2/B → ISO 639-3 for the codes that differ.
# IA uses 639-2/B (bibliographic); most are identical to 639-3 but ~20 differ.
_ISO2B_TO_ISO3: dict[str, str] = {
    "alb": "sqi", "arm": "hye", "baq": "eus", "bur": "mya", "chi": "zho",
    "cze": "ces", "dut": "nld", "fre": "fra", "geo": "kat", "ger": "deu",
    "gre": "ell", "ice": "isl", "mac": "mkd", "mao": "mri", "may": "msa",
    "per": "fas", "rum": "ron", "slo": "slk", "tib": "bod", "wel": "cym",
}

# ISO 639-1 (2-letter) → ISO 639-3 (3-letter) — IA sometimes uses these too
_ISO1_TO_ISO3: dict[str, str] = {
    "aa": "aar", "ab": "abk", "af": "afr", "ak": "aka", "am": "amh",
    "an": "arg", "ar": "ara", "as": "asm", "av": "ava", "ay": "aym",
    "az": "aze", "ba": "bak", "be": "bel", "bg": "bul", "bh": "bih",
    "bi": "bis", "bm": "bam", "bn": "ben", "bo": "bod", "br": "bre",
    "bs": "bos", "ca": "cat", "ce": "che", "ch": "cha", "co": "cos",
    "cr": "cre", "cs": "ces", "cu": "chu", "cv": "chv", "cy": "cym",
    "da": "dan", "de": "deu", "dv": "div", "dz": "dzo", "ee": "ewe",
    "el": "ell", "en": "eng", "eo": "epo", "es": "spa", "et": "est",
    "eu": "eus", "fa": "fas", "ff": "ful", "fi": "fin", "fj": "fij",
    "fo": "fao", "fr": "fra", "fy": "fry", "ga": "gle", "gd": "gla",
    "gl": "glg", "gn": "grn", "gu": "guj", "gv": "glv", "ha": "hau",
    "he": "heb", "hi": "hin", "ho": "hmo", "hr": "hrv", "ht": "hat",
    "hu": "hun", "hy": "hye", "hz": "her", "ia": "ina", "id": "ind",
    "ie": "ile", "ig": "ibo", "ii": "iii", "ik": "ipk", "io": "ido",
    "is": "isl", "it": "ita", "iu": "iku", "ja": "jpn", "jv": "jav",
    "ka": "kat", "kg": "kon", "ki": "kik", "kj": "kua", "kk": "kaz",
    "kl": "kal", "km": "khm", "kn": "kan", "ko": "kor", "kr": "kau",
    "ks": "kas", "ku": "kur", "kv": "kom", "kw": "cor", "ky": "kir",
    "la": "lat", "lb": "ltz", "lg": "lug", "li": "lim", "ln": "lin",
    "lo": "lao", "lt": "lit", "lu": "lub", "lv": "lav", "mg": "mlg",
    "mh": "mah", "mi": "mri", "mk": "mkd", "ml": "mal", "mn": "mon",
    "mr": "mar", "ms": "msa", "mt": "mlt", "my": "mya", "na": "nau",
    "nb": "nob", "nd": "nde", "ne": "nep", "ng": "ndo", "nl": "nld",
    "nn": "nno", "no": "nor", "nr": "nbl", "nv": "nav", "ny": "nya",
    "oc": "oci", "oj": "oji", "om": "orm", "or": "ori", "os": "oss",
    "pa": "pan", "pi": "pli", "pl": "pol", "ps": "pus", "pt": "por",
    "qu": "que", "rm": "roh", "rn": "run", "ro": "ron", "ru": "rus",
    "rw": "kin", "sa": "san", "sc": "srd", "sd": "snd", "se": "sme",
    "sg": "sag", "si": "sin", "sk": "slk", "sl": "slv", "sm": "smo",
    "sn": "sna", "so": "som", "sq": "sqi", "sr": "srp", "ss": "ssw",
    "st": "sot", "su": "sun", "sv": "swe", "sw": "swa", "ta": "tam",
    "te": "tel", "tg": "tgk", "th": "tha", "ti": "tir", "tk": "tuk",
    "tl": "tgl", "tn": "tsn", "to": "ton", "tr": "tur", "ts": "tso",
    "tt": "tat", "tw": "twi", "ty": "tah", "ug": "uig", "uk": "ukr",
    "ur": "urd", "uz": "uzb", "ve": "ven", "vi": "vie", "vo": "vol",
    "wa": "wln", "wo": "wol", "xh": "xho", "yi": "yid", "yo": "yor",
    "za": "zha", "zh": "zho", "zu": "zul",
}


# Full English language name → ISO 639-3.
# IA frequently uses these instead of codes (e.g. "Latin", "French", "English").
_LANG_NAME_TO_ISO3: dict[str, str] = {
    "afrikaans": "afr", "albanian": "sqi", "amharic": "amh", "arabic": "ara",
    "armenian": "hye", "azerbaijani": "aze", "basque": "eus", "belarusian": "bel",
    "bengali": "ben", "bosnian": "bos", "breton": "bre", "bulgarian": "bul",
    "burmese": "mya", "catalan": "cat", "cebuano": "ceb", "cherokee": "chr",
    "chinese": "zho", "coptic": "cop", "croatian": "hrv", "czech": "ces",
    "danish": "dan", "dutch": "nld", "english": "eng", "esperanto": "epo",
    "estonian": "est", "faroese": "fao", "finnish": "fin", "french": "fra",
    "galician": "glg", "georgian": "kat", "german": "deu", "greek": "ell",
    "gujarati": "guj", "haitian": "hat", "hausa": "hau", "hebrew": "heb",
    "hindi": "hin", "hungarian": "hun", "icelandic": "isl", "indonesian": "ind",
    "interlingua": "ina", "irish": "gle", "italian": "ita", "japanese": "jpn",
    "javanese": "jav", "kannada": "kan", "kazakh": "kaz", "khmer": "khm",
    "korean": "kor", "kurdish": "kur", "lao": "lao", "latin": "lat",
    "latvian": "lav", "lithuanian": "lit", "luxembourgish": "ltz",
    "macedonian": "mkd", "malay": "msa", "malayalam": "mal", "maltese": "mlt",
    "maori": "mri", "marathi": "mar", "mongolian": "mon", "nepali": "nep",
    "norwegian": "nor", "occitan": "oci", "oriya": "ori", "panjabi": "pan",
    "pashto": "pus", "persian": "fas", "polish": "pol", "portuguese": "por",
    "punjabi": "pan", "romanian": "ron", "romansh": "roh", "russian": "rus",
    "sanskrit": "san", "scottish gaelic": "gla", "serbian": "srp", "sindhi": "snd",
    "sinhala": "sin", "sinhalese": "sin", "slovak": "slk", "slovenian": "slv",
    "somali": "som", "spanish": "spa", "sundanese": "sun", "swahili": "swa",
    "swedish": "swe", "tagalog": "tgl", "tamil": "tam", "tatar": "tat",
    "telugu": "tel", "thai": "tha", "tibetan": "bod", "tigrinya": "tir",
    "turkish": "tur", "turkmen": "tuk", "ukrainian": "ukr", "urdu": "urd",
    "uzbek": "uzb", "vietnamese": "vie", "welsh": "cym", "yiddish": "yid",
    "yoruba": "yor", "zulu": "zul",
    # Common IA variants
    "ancient greek": "grc", "middle english": "enm", "old english": "ang",
    "middle french": "frm", "old french": "fro", "classical greek": "grc",
    "modern greek": "ell",
}


def _to_iso3(lang_str: str) -> list[str]:
    """Convert IA language string to list of ISO 639-3 codes.

    IA stores language as a single string (e.g. "eng", "Latin", "eng,fre")
    or sometimes comma-separated. May use full names, 639-2/B, or 639-1 codes.
    """
    if not lang_str:
        return []
    result = []
    for code in lang_str.replace(";", ",").split(","):
        code = code.strip()
        if not code:
            continue
        lower = code.lower()
        # Try full name first
        if lower in _LANG_NAME_TO_ISO3:
            result.append(_LANG_NAME_TO_ISO3[lower])
        elif len(lower) == 2:
            result.append(_ISO1_TO_ISO3.get(lower, lower))
        elif len(lower) == 3:
            result.append(_ISO2B_TO_ISO3.get(lower, lower))  # 639-2/B → 639-3
        else:
            result.append(lower)
    return result


_UPSERT_BOOK = """\
INSERT INTO books (
    urn, source, title, authors, language, format, year,
    publisher, description, size_bytes, content_type,
    cover_s3_key, added_date, metadata
) VALUES (
    $1, 'ia', $2, $3, $4, $5,
    $6, $7, $8, $9, $10,
    $11, $12, $13
)
ON CONFLICT (urn) DO UPDATE SET
    title       = COALESCE(EXCLUDED.title, books.title),
    authors     = COALESCE(EXCLUDED.authors, books.authors),
    language    = COALESCE(EXCLUDED.language, books.language),
    format      = COALESCE(EXCLUDED.format, books.format),
    year        = COALESCE(EXCLUDED.year, books.year),
    publisher   = COALESCE(EXCLUDED.publisher, books.publisher),
    description = COALESCE(EXCLUDED.description, books.description),
    size_bytes  = COALESCE(EXCLUDED.size_bytes, books.size_bytes),
    content_type = COALESCE(EXCLUDED.content_type, books.content_type),
    cover_s3_key = COALESCE(EXCLUDED.cover_s3_key, books.cover_s3_key),
    added_date  = COALESCE(EXCLUDED.added_date, books.added_date),
    metadata    = COALESCE(EXCLUDED.metadata, books.metadata),
    updated_at  = now()
"""

_UPSERT_ID = """\
INSERT INTO book_identifiers (urn, type, value)
VALUES ($1, $2, $3)
ON CONFLICT DO NOTHING
"""


def _safe_year(val: str) -> int | None:
    if not val:
        return None
    m = re.search(r"\b(\d{4})\b", val)
    if m:
        y = int(m.group(1))
        return y if 0 < y < 3000 else None
    return None


def _safe_date(val: str) -> str | None:
    if not val:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", val)
    return m.group(1) if m else None


class Database:
    """Thin async wrapper around the shared PostgreSQL books table."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
        logger.info("Connected to PostgreSQL")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("Closed PostgreSQL connection pool")

    async def ping(self) -> bool:
        """Check if the database is reachable."""
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def upsert_book(
        self,
        identifier: str,
        meta: ItemMetadata,
        *,
        cover_s3_key: str | None = None,
    ) -> None:
        """Upsert an IA item and its identifiers into PostgreSQL."""
        if not self._pool:
            return

        urn = f"urn:ia:{identifier}"

        # Authors
        creators = meta.creator
        if isinstance(creators, list):
            authors = ", ".join(creators) if creators else None
        else:
            authors = creators or None

        # Description
        desc = meta.description
        if isinstance(desc, list):
            desc = "\n".join(desc)
        description = desc or None

        # Best file format
        best = meta.get_best_file()
        fmt = best.format.lower() if best else None
        # Normalize common IA format names
        if fmt and "pdf" in fmt:
            fmt = "pdf"
        elif fmt and "epub" in fmt:
            fmt = "epub"
        elif fmt and "djvu" in fmt:
            fmt = "djvu"

        # Date
        added_date_str = _safe_date(meta.addeddate or meta.publicdate)
        added_date = datetime.date.fromisoformat(added_date_str) if added_date_str else None

        # Collect identifiers
        id_rows: list[tuple[str, str, str]] = []
        for isbn in meta.isbn:
            if isbn:
                id_rows.append((urn, "isbn", isbn))
        for oclc_id in meta.oclc:
            if oclc_id:
                id_rows.append((urn, "oclc", oclc_id))
        for lccn_id in meta.lccn:
            if lccn_id:
                id_rows.append((urn, "lccn", lccn_id))
        for issn_id in meta.issn:
            if issn_id:
                id_rows.append((urn, "issn", issn_id))
        for doi_id in meta.doi:
            if doi_id:
                id_rows.append((urn, "doi", doi_id))
        if meta.ark:
            id_rows.append((urn, "ark", meta.ark))
        if meta.openlibrary_edition:
            id_rows.append((urn, "ol", meta.openlibrary_edition))
        if meta.openlibrary_work:
            id_rows.append((urn, "ol_work", meta.openlibrary_work))

        # Strip files from metadata blob (too large)
        meta_dict = asdict(meta)
        meta_dict.pop("files", None)
        meta_dict.pop("_raw", None)

        args = (
            urn,                                    # $1
            meta.title or identifier,               # $2
            authors,                                # $3
            _to_iso3(meta.language) or None,         # $4
            fmt,                                    # $5
            _safe_year(meta.year or meta.date),     # $6
            meta.publisher or None,                 # $7
            description,                            # $8
            meta.item_size or None,                 # $9
            meta.mediatype or None,                 # $10
            cover_s3_key,                           # $11
            added_date,                             # $12
            json.dumps(meta_dict),                  # $13
        )

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(_UPSERT_BOOK, *args)
                    if id_rows:
                        await conn.executemany(_UPSERT_ID, id_rows)
            logger.info("Upserted book %s (%s) to PostgreSQL", urn, meta.title)
        except Exception:
            logger.exception("Failed to upsert book %s to PostgreSQL", urn)
