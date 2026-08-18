"""Download and extract an official PostgreSQL HTML documentation archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tarfile
import tempfile
from urllib.request import Request, urlopen

DEFAULT_VERSION = "18.4"
ARCHIVE_URL = "https://ftp.postgresql.org/pub/source/v{version}/postgresql-{version}-docs.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an already downloaded version.",
    )
    return parser.parse_args()


def download_archive(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "pg-docs-rag/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def extract_html(archive: Path, destination: Path, force: bool = False) -> int:
    existing = list(destination.glob("*.html")) if destination.exists() else []
    if existing and not force:
        raise FileExistsError(
            f"{destination} already contains HTML files; use --force to overwrite."
        )
    destination.mkdir(parents=True, exist_ok=True)

    count = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            member_path = Path(member.name)
            if not member.isfile() or member_path.suffix.lower() != ".html":
                continue
            if tuple(member_path.parts[-5:-1]) != ("doc", "src", "sgml", "html"):
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            # Flatten only verified HTML members; never extract archive paths.
            (destination / member_path.name).write_bytes(source.read())
            count += 1
    return count


def main() -> None:
    args = parse_args()
    version = args.version.strip()
    if not re.fullmatch(r"\d+\.\d+", version):
        raise ValueError("version must use the major.minor form, for example 18.4")
    destination = args.output_root / f"postgresql-{version}"
    url = ARCHIVE_URL.format(version=version)

    print(f"Downloading official PostgreSQL {version} documentation...")
    with tempfile.TemporaryDirectory(prefix="pg-docs-rag-") as temp_dir:
        archive = Path(temp_dir) / f"postgresql-{version}-docs.tar.gz"
        download_archive(url, archive)
        file_count = extract_html(archive, destination, force=args.force)

    manifest = {
        "version": version,
        "source": url,
        "html_files": file_count,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {file_count} HTML files to {destination}")


if __name__ == "__main__":
    main()
