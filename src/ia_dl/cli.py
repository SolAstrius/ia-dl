"""CLI for testing the Internet Archive downloader."""

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def test_download(id: str, format: str = "pdf") -> None:
    """Test downloading an item (no S3, just direct download)."""
    from .ia_client import IAClient
    from .config import get_settings
    from .downloader import download_item
    from .urn import parse_urn, WrongResolverError, InvalidUrnError

    # Parse URN or raw identifier (RFC 8141)
    try:
        parsed = parse_urn(id)
    except WrongResolverError as e:
        print(f"Error: {e}")
        print("  This CLI only handles ia URNs (urn:ia:<identifier>)")
        sys.exit(1)
    except InvalidUrnError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Format priority: r-component > CLI arg
    preferred_format = parsed.format or format

    settings = get_settings()
    client = IAClient.create(
        timeout=settings.download_timeout,
        access_key=settings.ia_access_key,
        secret_key=settings.ia_secret_key,
    )

    print(f"Downloading {parsed.canonical()}")
    print(f"  Identifier: {parsed.identifier}")
    print(f"  Format preference: {preferred_format}")
    if parsed.format:
        print(f"  (format from URN r-component)")

    try:
        result = await download_item(client, settings, parsed.identifier, [preferred_format])
        print(f"Downloaded {result.size_bytes} bytes in {result.duration_ms}ms")
        print(f"  Filename: {result.filename}")
        print(f"  Format: {result.format}")

        # Save to local file for inspection
        with open(result.filename, "wb") as f:
            f.write(result.content)
        print(f"  Saved: {result.filename}")

    finally:
        await client.close()


async def test_metadata(id: str, full: bool = False) -> None:
    """Test fetching metadata for an item."""
    from .ia_client import IAClient
    from .config import get_settings
    from .urn import parse_urn, WrongResolverError, InvalidUrnError

    # Parse URN or raw identifier (RFC 8141)
    try:
        parsed = parse_urn(id)
    except WrongResolverError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except InvalidUrnError as e:
        print(f"Error: {e}")
        sys.exit(1)

    settings = get_settings()
    client = IAClient.create(
        timeout=30.0,
        access_key=settings.ia_access_key,
        secret_key=settings.ia_secret_key,
    )

    print(f"Fetching metadata for {parsed.canonical()}")

    try:
        meta = await client.fetch_metadata(parsed.identifier)

        if full:
            # Complete metadata output
            def p(label: str, value) -> None:
                """Print non-empty values."""
                if value and value != 0:
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    print(f"{label}: {value}")

            print("\n=== Core Bibliographic ===")
            p("Identifier", meta.identifier)
            p("Title", meta.title)
            p("Creator", meta.creator)
            p("Date", meta.date)
            p("Year", meta.year)
            p("Publisher", meta.publisher)
            p("Description", meta.description)

            print("\n=== Classification ===")
            p("Subject", meta.subject)
            p("Collection", meta.collection)
            p("Language", meta.language)
            p("Mediatype", meta.mediatype)

            print("\n=== External Identifiers ===")
            p("ISBN", meta.isbn)
            p("OCLC", meta.oclc)
            p("LCCN", meta.lccn)
            p("ISSN", meta.issn)
            p("DOI", meta.doi)
            p("ARK", meta.ark)
            p("OpenLibrary Edition", meta.openlibrary_edition)
            p("OpenLibrary Work", meta.openlibrary_work)
            if meta.external_identifier:
                print("External Identifiers:")
                for ext_id in meta.external_identifier:
                    print(f"  - {ext_id}")

            print("\n=== Contributors ===")
            p("Contributor", meta.contributor)
            p("Sponsor", meta.sponsor)

            print("\n=== Licensing ===")
            p("License URL", meta.licenseurl)
            p("Rights", meta.rights)
            p("Copyright Status", meta.possible_copyright_status)
            p("Copyright Region", meta.copyright_region)

            print("\n=== Scanning Info ===")
            p("Scanning Center", meta.scanningcenter)
            p("Scanner", meta.scanner)
            p("Scan Date", meta.scandate)
            p("PPI", meta.ppi)
            p("Image Count", meta.imagecount)
            p("Camera", meta.camera)
            p("Operator", meta.operator)
            p("Repub State", meta.repub_state)
            p("Foldout Count", meta.foldoutcount)
            p("Bookplate Leaf", meta.bookplateleaf)

            print("\n=== OCR Info ===")
            p("OCR", meta.ocr)
            p("OCR Module Version", meta.ocr_module_version)
            p("OCR Detected Language", meta.ocr_detected_lang)
            p("OCR Language Confidence", meta.ocr_detected_lang_conf)
            p("OCR Detected Script", meta.ocr_detected_script)
            p("OCR Script Confidence", meta.ocr_detected_script_conf)

            print("\n=== Upload Info ===")
            p("Uploader", meta.uploader)
            p("Added Date", meta.addeddate)
            p("Public Date", meta.publicdate)
            p("Update Date", meta.updatedate)
            p("Created (unix)", meta.created)
            p("Last Updated (unix)", meta.item_last_updated)

            print("\n=== Stats ===")
            p("Downloads (total)", meta.downloads)
            p("Downloads (week)", meta.week)
            p("Downloads (month)", meta.month)
            p("Files Count", meta.files_count)
            p("Item Size (bytes)", meta.item_size)

            print("\n=== Server Info ===")
            p("Server", meta.server)
            p("Dir", meta.dir)
            p("D1", meta.d1)
            p("D2", meta.d2)
            p("Workable Servers", meta.workable_servers)

            # Show all files with full details
            if meta.files:
                print(f"\n=== Files ({len(meta.files)}) ===")
                for f in meta.files:
                    print(f"\n  {f.name}")
                    print(f"    Format: {f.format}")
                    print(f"    Size: {f.size} bytes")
                    if f.source:
                        print(f"    Source: {f.source}")
                    if f.original:
                        print(f"    Original: {f.original}")
                    if f.md5:
                        print(f"    MD5: {f.md5}")
                    if f.sha1:
                        print(f"    SHA1: {f.sha1}")
                    if f.crc32:
                        print(f"    CRC32: {f.crc32}")
                    if f.mtime:
                        print(f"    MTime: {f.mtime}")
                    if f.rotation:
                        print(f"    Rotation: {f.rotation}")

        else:
            # Brief output
            print(f"Title: {meta.title}")
            print(f"Creator: {meta.creator}")
            print(f"Date: {meta.date or meta.year}")
            print(f"Mediatype: {meta.mediatype}")
            print(f"Collections: {', '.join(meta.collection)}")
            print(f"Files: {meta.files_count}")
            print(f"Total size: {meta.item_size} bytes")
            print(f"Downloads: {meta.downloads}")

            # Show available files
            if meta.files:
                print("\nAvailable files:")
                for f in meta.files[:20]:  # Show first 20
                    print(f"  - {f.name} ({f.format}, {f.size} bytes)")
                if len(meta.files) > 20:
                    print(f"  ... and {len(meta.files) - 20} more")

            # Show best file
            best = meta.get_best_file()
            if best:
                print(f"\nBest file for download: {best.name} ({best.format})")

    finally:
        await client.close()


async def test_search(query: str, rows: int = 20, mediatype: str | None = None) -> None:
    """Test searching the Internet Archive."""
    from .ia_client import IAClient
    from .config import get_settings

    settings = get_settings()
    client = IAClient.create(
        timeout=30.0,
        access_key=settings.ia_access_key,
        secret_key=settings.ia_secret_key,
    )

    print(f"Searching: {query}")
    if mediatype:
        print(f"  Mediatype filter: {mediatype}")

    try:
        result = await client.search(
            query=query,
            rows=rows,
            mediatype=mediatype,
        )

        print(f"\nFound {result.total} results (showing {result.count}):\n")

        for item in result.items:
            # Format creators
            creators = ", ".join(item.creator[:2])
            if len(item.creator) > 2:
                creators += f" +{len(item.creator) - 2} more"

            # Format size
            size_mb = item.item_size / 1024 / 1024 if item.item_size else 0

            print(f"  {item.to_urn()}")
            print(f"    Title: {item.title[:70]}{'...' if len(item.title) > 70 else ''}")
            if creators:
                print(f"    Creator: {creators}")
            if item.year:
                print(f"    Year: {item.year}")
            if item.formats:
                print(f"    Formats: {', '.join(item.formats)}")
            print(f"    Downloads: {item.downloads:,}  Size: {size_mb:.1f} MB")
            print()

    finally:
        await client.close()


async def test_api(id: str, format: str = "pdf") -> None:
    """Test the full API endpoint (requires S3 config)."""
    import httpx

    from .config import get_settings
    from .urn import parse_urn, WrongResolverError, InvalidUrnError

    # Parse URN or raw identifier (RFC 8141)
    try:
        parsed = parse_urn(id)
    except WrongResolverError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except InvalidUrnError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Format priority: r-component > CLI arg
    preferred_format = parsed.format or format

    settings = get_settings()
    # Use canonical URN (without r-component) in URL - format goes in body
    url = f"http://{settings.host}:{settings.port}/item/{parsed.canonical()}/download"

    print(f"POST {url}")
    if parsed.format:
        print(f"  (format '{parsed.format}' from URN r-component)")

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            url,
            json={"preferred_formats": [preferred_format]},
        )

        if response.is_success:
            data = response.json()
            print("Success")
            print(f"  ID: {data['id']}")
            print(f"  Identifier: {data['identifier']}")
            print(f"  Title: {data['title']}")
            print(f"  Filename: {data['filename']}")
            print(f"  Format: {data['format']}")
            print(f"  Size: {data['size_bytes']} bytes")
            print(f"  Duration: {data['duration_ms']}ms")
            print(f"  Cached: {data['cached']}")
            print(f"  URL: {data['download_url'][:80]}...")
        else:
            print(f"Error {response.status_code}: {response.text}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Internet Archive download CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ia-dl search "creator:dickens" --mediatype texts
  ia-dl search "title:great expectations" --rows 10
  ia-dl download --id taleoftwocities00dick
  ia-dl download --id taleoftwocities00dick --format epub
  ia-dl download --id 'urn:ia:taleoftwocities00dick?+format=epub'  # RFC 8141
  ia-dl metadata --id taleoftwocities00dick
  ia-dl serve
  ia-dl api --id taleoftwocities00dick

Search query syntax (Lucene):
  creator:dickens              - Search by creator
  title:"great expectations"   - Exact phrase in title
  dickens AND london           - Boolean AND
  year:[1800 TO 1900]          - Year range

URN Format (RFC 8141):
  urn:ia:<identifier>[?+r-component][?=q-component][#f-component]

  r-component: Resolution hints (e.g., ?+format=epub)
  q-component: Query params passed to resource (e.g., ?=page=5)
  f-component: Fragment identifier (e.g., #chapter1)
""",
    )
    parser.add_argument("command", choices=["search", "download", "metadata", "api", "serve"])
    parser.add_argument(
        "--id", "-i",
        help="Item URN (urn:ia:<identifier>) or raw identifier",
        metavar="URN",
    )
    parser.add_argument("--format", "-f", default="pdf", help="Format preference (default: pdf)")
    parser.add_argument("--full", action="store_true", help="Show complete metadata (for metadata command)")
    parser.add_argument("--rows", "-n", type=int, default=20, help="Number of search results (default: 20)")
    parser.add_argument("--mediatype", "-m", default="texts", help="Filter search by mediatype (default: texts)")
    parser.add_argument("query", nargs="?", help="Search query (for search command)")

    args = parser.parse_args()

    if args.command == "serve":
        from .main import main as serve_main
        serve_main()

    elif args.command == "search":
        if not args.query:
            print("Error: search query required")
            print("Usage: ia-dl search 'creator:dickens'")
            sys.exit(1)
        asyncio.run(test_search(args.query, args.rows, args.mediatype))

    elif args.command == "download":
        if not args.id:
            print("Error: --id required (URN or identifier)")
            sys.exit(1)
        asyncio.run(test_download(args.id, args.format))

    elif args.command == "metadata":
        if not args.id:
            print("Error: --id required (URN or identifier)")
            sys.exit(1)
        asyncio.run(test_metadata(args.id, args.full))

    elif args.command == "api":
        if not args.id:
            print("Error: --id required (URN or identifier)")
            sys.exit(1)
        asyncio.run(test_api(args.id, args.format))


if __name__ == "__main__":
    main()
