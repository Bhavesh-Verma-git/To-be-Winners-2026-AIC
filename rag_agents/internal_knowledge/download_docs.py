"""
============================================================
  Agent 3: Internal Knowledge Assistant
  Phase 1 — download_docs.py
  
  What this script does:
  ─────────────────────
  1. Uses git sparse-checkout to clone ONLY the app-service
     folder from the massive azure-docs repo (avoids pulling
     15,000+ files we don't need).
  2. Filters out junk files:
     - redirect stubs (body < 200 chars after frontmatter)
     - non-markdown files (media/, TOC.yml, includes/)
  3. Copies the clean files into our local `documents/` folder
     which is what build_index.py will read.

  Run this ONCE before building the index:
     python rag_agents/internal_knowledge/download_docs.py
============================================================
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── Where to save the downloaded docs ────────────────────────
BASE_DIR   = Path(__file__).parent
DOCS_DIR   = BASE_DIR / "documents"
CLONE_DIR  = BASE_DIR / "azure-docs-sparse"   # Temp clone folder

# ── Patterns we WANT (article categories) ────────────────────
KEEP_PREFIXES = [
    "quickstart-",
    "configure-",
    "deploy-",
    "tutorial-",
    "troubleshoot-",
    "overview-",
    "app-service-",
    "how-to-",
]

# ── Patterns we DISCARD (noise) ───────────────────────────────
SKIP_PREFIXES = [
    "includes/",
    "media/",
    "TOC",
    "index",
    "whats-new",
    "faq",
]


def run_git(args: list, cwd: Path):
    """Run a git command safely, printing output."""
    result = subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] Git command failed: git {' '.join(args)}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def clone_sparse():
    """Clone ONLY the articles/app-service folder using sparse-checkout."""
    if CLONE_DIR.exists():
        print(f"[SKIP] Clone directory already exists: {CLONE_DIR}")
        print("       Delete it and re-run if you want a fresh download.")
        return

    print("[GIT] Cloning azure-docs with sparse-checkout (app-service only)...")
    print("      This downloads ~300 files instead of 15,000+. Please wait...")

    # Step 1: Clone with no blobs (just the tree structure)
    run_git([
        "clone",
        "--depth=1",
        "--filter=blob:none",
        "--sparse",
        "--no-checkout",
        "https://github.com/MicrosoftDocs/azure-docs.git",
        str(CLONE_DIR)
    ], cwd=BASE_DIR)

    # Step 2: Set sparse-checkout to only app-service
    run_git(["sparse-checkout", "set", "articles/app-service"], cwd=CLONE_DIR)

    # Step 3: Checkout the files
    run_git(["checkout"], cwd=CLONE_DIR)

    print("[OK] Sparse clone complete.")


def is_valid_article(filepath: Path) -> bool:
    """
    Returns True if the file is a real documentation article.
    Filters out:
    - Files with body < 200 chars (redirect stubs)
    - Files not matching our keep-prefix list
    """
    name = filepath.name.lower()

    # Must be a markdown file
    if filepath.suffix != ".md":
        return False

    # Must match at least one of our keep patterns
    if not any(name.startswith(p) for p in KEEP_PREFIXES):
        return False

    # Skip known noise patterns
    if any(name.startswith(p) for p in SKIP_PREFIXES):
        return False

    # Read file to check body length (filter out redirect stubs)
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        # Strip YAML frontmatter (everything between first --- and second ---)
        if content.startswith("---"):
            parts = content.split("---", 2)
            body = parts[2] if len(parts) >= 3 else content
        else:
            body = content
        # Reject if the body is too short (it's just a redirect stub)
        if len(body.strip()) < 200:
            return False
    except Exception:
        return False

    return True


def copy_documents():
    """Copy valid article files into our local documents/ folder."""
    source_dir = CLONE_DIR / "articles" / "app-service"

    if not source_dir.exists():
        print(f"[ERROR] App service folder not found: {source_dir}")
        print("        Make sure the git sparse-checkout completed successfully.")
        sys.exit(1)

    # Clean and recreate docs folder
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True)

    all_md_files  = list(source_dir.glob("*.md"))
    valid_files   = [f for f in all_md_files if is_valid_article(f)]
    skipped       = len(all_md_files) - len(valid_files)

    print(f"\n[FILTER] Total .md files found : {len(all_md_files)}")
    print(f"[FILTER] Valid articles kept   : {len(valid_files)}")
    print(f"[FILTER] Junk/stubs skipped    : {skipped}")

    for src_file in valid_files:
        dst_file = DOCS_DIR / src_file.name
        shutil.copy2(src_file, dst_file)

    print(f"\n[OK] {len(valid_files)} clean articles copied to: {DOCS_DIR}")
    return len(valid_files)


if __name__ == "__main__":
    print("=" * 60)
    print("  Agent 3 — Azure App Service Docs Downloader")
    print("=" * 60)

    # Verify git is installed
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Git is not installed or not on PATH.")
        print("        Install Git from: https://git-scm.com/downloads")
        sys.exit(1)

    clone_sparse()
    count = copy_documents()

    print("\n" + "=" * 60)
    print(f"  Download complete! {count} articles ready.")
    print("  Next step: python rag_agents/internal_knowledge/build_index.py")
    print("=" * 60)
