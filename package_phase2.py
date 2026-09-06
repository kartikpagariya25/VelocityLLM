"""
VelocityLLM - Phase 2 Package Generator
Creates a clean, production-grade distribution zip archive of the repository,
excluding local caches, virtual environments, and git metadata.
"""

import os
import zipfile

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "models",  # HuggingFace weights (~4GB) should be downloaded via setup guide
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
}

EXCLUDE_FILES = {
    "VelocityLLM_Phase2.zip",
    "results.csv",
}


def create_phase2_zip(output_zip_name: str = "VelocityLLM_Phase2.zip"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, output_zip_name)

    print(f"Creating package archive: {zip_path}...")
    total_files = 0
    total_bytes = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for file in files:
                if file in EXCLUDE_FILES:
                    continue
                _, ext = os.path.splitext(file)
                if ext in EXCLUDE_EXTENSIONS:
                    continue

                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, base_dir)

                # Skip any stray cache files
                if "__pycache__" in rel_path or ".pytest_cache" in rel_path:
                    continue

                zipf.write(abs_path, rel_path)
                file_size = os.path.getsize(abs_path)
                total_files += 1
                total_bytes += file_size
                print(f"  + Added: {rel_path} ({file_size:,} bytes)")

    print(f"\nSuccessfully packed {total_files} files ({total_bytes:,} uncompressed bytes) into {output_zip_name}")
    print(f"Archive size: {os.path.getsize(zip_path):,} bytes")
    return zip_path


if __name__ == "__main__":
    create_phase2_zip()
