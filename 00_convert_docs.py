"""Convert AsciiDoc (.adoc) and PDF files to Markdown.

Uses pypandoc (pandoc wrapper) for AsciiDoc and docling for PDFs.
Output is either a single merged Markdown file (default) or individual
files written to an output directory.
"""

import os
import sys
import argparse
import glob


SUPPORTED_EXTENSIONS = {".adoc", ".pdf"}


def find_files(input_paths):
    """Resolve input paths into a flat list of supported files.

    Each entry in *input_paths* can be a file or a directory.
    Directories are scanned recursively.
    """
    files = []
    for path in input_paths:
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
            else:
                print(f"Skipping unsupported file: {path}")
        elif os.path.isdir(path):
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(sorted(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True)))
        else:
            print(f"Path not found, skipping: {path}")
    return files


_pandoc_checked = False


def _download_pandoc():
    """Download a pandoc binary from GitHub releases (no system ar needed)."""
    import json
    import platform
    import tarfile
    import tempfile
    import urllib.request

    arch = "arm64" if platform.machine() == "aarch64" else "amd64"
    api_url = "https://api.github.com/repos/jgm/pandoc/releases/latest"

    print("  Fetching latest pandoc release info...")
    with urllib.request.urlopen(api_url) as resp:
        tag = json.loads(resp.read())["tag_name"]

    url = f"https://github.com/jgm/pandoc/releases/download/{tag}/pandoc-{tag}-linux-{arch}.tar.gz"
    target_dir = os.path.join(os.path.expanduser("~"), ".local", "bin")
    os.makedirs(target_dir, exist_ok=True)

    print(f"  Downloading pandoc {tag}...")
    tarball_path = os.path.join(tempfile.gettempdir(), f"pandoc-{tag}.tar.gz")
    urllib.request.urlretrieve(url, tarball_path)

    with tarfile.open(tarball_path) as tar:
        for member in tar.getmembers():
            if member.name.endswith("/bin/pandoc"):
                member.name = "pandoc"
                tar.extract(member, target_dir, filter="data")
                break

    os.remove(tarball_path)
    pandoc_path = os.path.join(target_dir, "pandoc")
    os.chmod(pandoc_path, 0o755)
    os.environ["PATH"] = target_dir + ":" + os.environ.get("PATH", "")
    print(f"  Installed pandoc {tag} to {pandoc_path}")


def _ensure_pandoc():
    """Ensure a pandoc version that supports the asciidoc reader (>= 2.15)."""
    global _pandoc_checked
    if _pandoc_checked:
        return
    import pypandoc
    try:
        version = pypandoc.get_pandoc_version()
        major, minor = (int(x) for x in version.split(".")[:2])
        if major > 2 or (major == 2 and minor >= 15):
            _pandoc_checked = True
            return
        print(f"System pandoc {version} is too old (need >= 2.15 for AsciiDoc).")
    except OSError:
        print("No system pandoc found.")
    _download_pandoc()
    _pandoc_checked = True


def convert_adoc(path):
    """Convert an AsciiDoc file to Markdown via pandoc."""
    import pypandoc
    _ensure_pandoc()
    return pypandoc.convert_file(path, "md", format="asciidoc")


def convert_pdf(path):
    """Convert a PDF file to Markdown via docling."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(path)
    return result.document.export_to_markdown()


def convert_file(path):
    """Dispatch to the correct converter based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".adoc":
        return convert_adoc(path)
    elif ext == ".pdf":
        return convert_pdf(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert AsciiDoc and PDF files to Markdown."
    )
    parser.add_argument("inputs", nargs="+",
                        help="Files or directories to convert (recursively scans directories)")
    parser.add_argument("-o", "--output", type=str, default="corpus.md",
                        help="Output path. If --no-merge, treated as a directory (default: corpus.md)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Write individual .md files instead of a single merged file")
    parser.add_argument("--separator", type=str, default="\n\n---\n\n",
                        help="Separator inserted between documents when merging (default: horizontal rule)")
    args = parser.parse_args()

    files = find_files(args.inputs)
    if not files:
        print("No supported files (.adoc, .pdf) found in the provided paths.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to convert.")

    converted = []
    for i, path in enumerate(files, 1):
        ext = os.path.splitext(path)[1].lower()
        label = "pypandoc" if ext == ".adoc" else "docling"
        print(f"  [{i}/{len(files)}] {path}  ({label})")
        try:
            md_text = convert_file(path)
            converted.append((path, md_text))
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if not converted:
        print("All conversions failed.")
        sys.exit(1)

    if args.no_merge:
        os.makedirs(args.output, exist_ok=True)
        for src_path, md_text in converted:
            basename = os.path.splitext(os.path.basename(src_path))[0] + ".md"
            out_path = os.path.join(args.output, basename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"  -> {out_path}")
        print(f"\nWrote {len(converted)} file(s) to {args.output}/")
    else:
        merged = args.separator.join(md_text for _, md_text in converted)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(merged)
        print(f"\nMerged {len(converted)} document(s) into {args.output}")


if __name__ == "__main__":
    main()
