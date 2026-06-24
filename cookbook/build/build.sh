#!/usr/bin/env bash
# =============================================================================
# build.sh — Render the Engram Cookbook in multiple formats
#
# Usage:
#   ./build.sh          # runs all targets (html, pdf, epub); skips any whose
#                       # tool is missing and prints a summary at the end
#   ./build.sh html     # mdBook HTML only
#   ./build.sh pdf      # Pandoc PDF only
#   ./build.sh epub     # Pandoc EPUB only
#
# Prerequisites (install once):
#   mdBook:  cargo install mdbook   — or —  brew install mdbook
#   Pandoc:  brew install pandoc    — or —  apt-get install pandoc
#
#   PDF needs ONE of these engines (build.sh auto-detects, first found wins):
#     LaTeX:       brew install --cask mactex-no-gui   (macOS)
#                  apt-get install texlive-xetex        (Debian/Ubuntu)
#     OR HTML->PDF: brew install weasyprint   — or —   brew install wkhtmltopdf
#
# Note: chmod +x build.sh before running for the first time.
# =============================================================================

set -uo pipefail   # NOTE: no -e — we want `all` to continue past a missing tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_OUT="${SCRIPT_DIR}/out"
mkdir -p "${BUILD_OUT}"

# Chapter list — must match TOC order exactly (used for the Pandoc concatenation)
CHAPTERS=(
  "part-0-big-picture/01-why-finetune.md"
  "part-0-big-picture/02-mental-models.md"
  "part-0-big-picture/03-landscape-when-to-use-what.md"
  "part-1-concepts-primer/04-transformers-in-20-min.md"
  "part-1-concepts-primer/05-tokenization-context-chat-templates.md"
  "part-1-concepts-primer/06-lora-qlora-explained.md"
  "part-1-concepts-primer/07-how-training-works.md"
  "part-2-setup-tools/08-hardware-and-environment.md"
  "part-2-setup-tools/09-the-toolbox.md"
  "part-2-setup-tools/10-choosing-base-model.md"
  "part-3-task-and-data/11-defining-the-task.md"
  "part-3-task-and-data/12-data-format-and-schema.md"
  "part-3-task-and-data/13-synthetic-data-generation.md"
  "part-3-task-and-data/14-data-prep-and-splits.md"
  "part-4-training/15-first-finetune-unsloth.md"
  "part-4-training/16-hyperparameters.md"
  "part-4-training/17-monitoring-training.md"
  "part-5-eval-iteration/18-evaluation.md"
  "part-5-eval-iteration/19-debugging-bad-results.md"
  "part-5-eval-iteration/20-iterating.md"
  "part-6-deploy-beyond/21-saving-merging-exporting.md"
  "part-6-deploy-beyond/22-serving-and-integration.md"
  "part-6-deploy-beyond/23-toward-continual-learning.md"
  "appendices/A-glossary.md"
  "appendices/B-project-layout-and-commands.md"
  "appendices/C-troubleshooting.md"
  "appendices/D-cost-time-and-checklist.md"
)

# Book-content subdirectories (everything mdBook should treat as source)
CONTENT_DIRS=(
  part-0-big-picture part-1-concepts-primer part-2-setup-tools
  part-3-task-and-data part-4-training part-5-eval-iteration
  part-6-deploy-beyond appendices
)

# ---------------------------------------------------------------------------
# Helper: concatenate all chapters into one file for Pandoc (path -> stdout)
# ---------------------------------------------------------------------------
concat_chapters() {
  local out_file="${BUILD_OUT}/book-combined.md"
  echo "--- Concatenating chapters into ${out_file} ---" >&2
  : > "${out_file}"
  for chapter in "${CHAPTERS[@]}"; do
    local full_path="${BOOK_DIR}/${chapter}"
    if [[ -f "${full_path}" ]]; then
      cat "${full_path}" >> "${out_file}"
      printf '\n\n' >> "${out_file}"
    else
      echo "WARNING: chapter not found, skipping: ${full_path}" >&2
    fi
  done
  echo "${out_file}"
}

# ---------------------------------------------------------------------------
# Target: html — mdBook
# We stage ONLY the book content into an isolated src/ dir. The previous
# layout pointed mdBook's src at the whole cookbook/ folder, so it tried to
# copy its own output dir into itself -> infinite path recursion. Staging
# avoids that and keeps .omc/ and build artifacts out of the rendered site.
# ---------------------------------------------------------------------------
build_html() {
  echo "=== Building HTML with mdBook ==="
  if ! command -v mdbook &>/dev/null; then
    echo "SKIP html: mdbook not found (cargo install mdbook | brew install mdbook)" >&2
    return 2
  fi

  local stage src
  stage="$(mktemp -d)"
  src="${stage}/src"
  mkdir -p "${src}"

  # SUMMARY.md is mandatory for mdBook; README.md becomes the intro page.
  cp "${BOOK_DIR}/SUMMARY.md" "${src}/"
  [[ -f "${BOOK_DIR}/README.md" ]] && cp "${BOOK_DIR}/README.md" "${src}/"
  local d
  for d in "${CONTENT_DIRS[@]}"; do
    [[ -d "${BOOK_DIR}/${d}" ]] && cp -R "${BOOK_DIR}/${d}" "${src}/"
  done

  cat > "${stage}/book.toml" <<'TOML'
[book]
title = "The Engram Cookbook"
authors = ["Engram Cookbook"]
language = "en"
src = "src"
TOML

  local html_out="${BUILD_OUT}/html"
  rm -rf "${html_out}"
  # dest is outside the staged src dir, so no self-copy recursion.
  if mdbook build "${stage}" --dest-dir "${html_out}"; then
    echo "HTML output: ${html_out}/index.html"
    rm -rf "${stage}"
    return 0
  fi
  rm -rf "${stage}"
  return 1
}

# ---------------------------------------------------------------------------
# Target: pdf — Pandoc, trying each installed engine until one actually works.
# An engine can be installed but broken (e.g. weasyprint missing system libs),
# so we attempt each in priority order and only skip if none succeed.
# ---------------------------------------------------------------------------
build_pdf() {
  echo "=== Building PDF with Pandoc ==="
  if ! command -v pandoc &>/dev/null; then
    echo "SKIP pdf: pandoc not found (brew install pandoc | apt-get install pandoc)" >&2
    return 2
  fi

  local available=""
  local e
  for e in xelatex lualatex pdflatex tectonic weasyprint wkhtmltopdf; do
    command -v "${e}" &>/dev/null && available="${available} ${e}"
  done
  if [[ -z "${available}" ]]; then
    echo "SKIP pdf: no PDF engine found. Install one of:" >&2
    echo "  LaTeX:      brew install --cask mactex-no-gui   /   apt-get install texlive-xetex" >&2
    echo "  HTML->PDF:  brew install weasyprint             /   brew install wkhtmltopdf" >&2
    return 2
  fi

  local combined; combined="$(concat_chapters)"
  local pdf_out="${BUILD_OUT}/engram-cookbook.pdf"

  for e in ${available}; do
    echo "Trying PDF engine: ${e}"
    set -- "${combined}" --from markdown --to pdf \
           --pdf-engine="${e}" --toc --toc-depth=2 \
           --metadata title="The Engram Cookbook" \
           --metadata author="Engram Cookbook"
    case "${e}" in
      xelatex|lualatex|pdflatex|tectonic)
        set -- "$@" --variable geometry="margin=1in" --variable fontsize=11pt --variable colorlinks=true ;;
    esac
    set -- "$@" --output "${pdf_out}"
    if pandoc "$@" 2>/tmp/pandoc_pdf_err; then
      echo "PDF output: ${pdf_out}  (engine: ${e})"
      return 0
    fi
    echo "  engine ${e} failed; $(tail -1 /tmp/pandoc_pdf_err 2>/dev/null)" >&2
  done

  echo "SKIP pdf: every installed engine failed (see errors above). HTML and EPUB are unaffected." >&2
  echo "  Easiest fix on macOS: brew install --cask mactex-no-gui   (then re-run: ./build.sh pdf)" >&2
  return 2
}

# ---------------------------------------------------------------------------
# Target: epub — Pandoc (no extra engine needed)
# ---------------------------------------------------------------------------
build_epub() {
  echo "=== Building EPUB with Pandoc ==="
  if ! command -v pandoc &>/dev/null; then
    echo "SKIP epub: pandoc not found (brew install pandoc | apt-get install pandoc)" >&2
    return 2
  fi
  local combined; combined="$(concat_chapters)"
  local epub_out="${BUILD_OUT}/engram-cookbook.epub"
  if pandoc "${combined}" --from markdown --to epub --toc --toc-depth=2 \
       --split-level=1 \
       --metadata title="The Engram Cookbook" \
       --metadata author="Engram Cookbook" \
       --output "${epub_out}"; then
    echo "EPUB output: ${epub_out}"
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
TARGET="${1:-all}"

run_one() {  # name -> prints status, returns the target's code
  local fn="$1"
  "${fn}"; return $?
}

case "${TARGET}" in
  html) build_html; exit $? ;;
  pdf)  build_pdf;  exit $? ;;
  epub) build_epub; exit $? ;;
  all)
    # bash 3.2 (macOS default) has no associative arrays — use a flat summary string.
    SUMMARY=""
    HARD_FAIL=0
    for pair in "html:build_html" "pdf:build_pdf" "epub:build_epub"; do
      name="${pair%%:*}"; fn="${pair##*:}"
      "${fn}"; code=$?
      case "${code}" in
        0) state="OK" ;;
        2) state="SKIPPED (tool missing)" ;;
        *) state="FAILED (exit ${code})"; HARD_FAIL=1 ;;
      esac
      SUMMARY="${SUMMARY}$(printf '  %-5s : %s' "${name}" "${state}")
"
      echo ""
    done
    echo "================ BUILD SUMMARY ================"
    printf '%s' "${SUMMARY}"
    echo "  outputs in: ${BUILD_OUT}/"
    echo "==============================================="
    [ "${HARD_FAIL}" -eq 1 ] && exit 1
    exit 0
    ;;
  *)
    echo "Unknown target: ${TARGET}"
    echo "Usage: $0 [html|pdf|epub|all]"
    exit 1 ;;
esac
