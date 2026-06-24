"""
Codal knowledge fetcher.
"""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List


class CodalFetcher:
    """
    Ingests raw standards documents into the codal engine.

    This loader performs real PDF text extraction first, then OCR fallback,
    and raises on missing files so the caller can decide how to proceed.
    """

    PDF_ALIASES = {
        "API_14E.pdf": ("API_4E.pdf",),
        "API_4E.pdf": ("API_14E.pdf",),
    }

    # Common Windows install path for Tesseract from UB Mannheim installer.
    _TESSERACT_FALLBACK_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def __init__(self, storage_path: str = "data/standards"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(__file__).resolve().parents[3]
        # Directory where extracted text is cached as .txt files to avoid
        # re-running expensive OCR or PDF parsing on every startup.
        self.cache_dir = self.storage_path / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_local_pdf(self, path: str) -> str:
        target = Path(path)
        if not target.is_absolute():
            target = self.repo_root / target

        target = self._resolve_existing_target(target)

        print(f"[Codal] Loading standards source: {target.name}...")
        if not target.exists():
            message = f"[Codal] Standards file not found at {target}."
            print(message + " Skipping this source instead of fabricating rules.")
            raise FileNotFoundError(message)

        if target.suffix.lower() == ".pdf":
            # --- Check disk cache first ---
            cached = self._load_cache(target)
            if cached is not None:
                print(f"[Codal] Loaded {len(cached)} characters from cache for {target.name}.")
                return cached

            # --- Stage 1: text-layer extraction (pdfplumber / pypdf) ---
            extracted = self._extract_pdf_text(target)
            if extracted.strip():
                print(f"[Codal] Extracted {len(extracted)} characters from PDF text layer.")
                self._save_cache(target, extracted)
                return extracted

            # --- Stage 2: OCR ---
            print(f"[Codal] PDF text extraction yielded no content for {target.name}. Trying OCR fallback...")
            ocr_text = self._ocr_pdf_text(target)
            if ocr_text.strip():
                print(f"[Codal] OCR extracted {len(ocr_text)} characters from PDF pages.")
                self._save_cache(target, ocr_text)
                return ocr_text

            # --- Stage 3: raw byte decode (last resort) ---
            print(
                f"[Codal] OCR fallback unavailable or empty for {target.name}. "
                "Falling back to raw text decode."
            )
            raw = target.read_text(encoding="utf-8", errors="ignore")
            if raw.strip():
                self._save_cache(target, raw)
            return raw

        return target.read_text(encoding="utf-8", errors="ignore")

    def _resolve_existing_target(self, target: Path) -> Path:
        if target.exists():
            return target

        for alias_name in self.PDF_ALIASES.get(target.name, ()):
            alias_target = target.with_name(alias_name)
            if alias_target.exists():
                print(f"[Codal] Using standards alias {target.name} -> {alias_target.name}.")
                return alias_target

        return target

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, target: Path) -> Path:
        """Derive a stable cache filename from the PDF path."""
        key = hashlib.md5(str(target.resolve()).encode()).hexdigest()[:12]
        return self.cache_dir / f"{target.stem}_{key}.txt"

    def _load_cache(self, target: Path) -> str | None:
        """Return cached text if it exists and is non-empty, else None."""
        cache_file = self._cache_path(target)
        if cache_file.exists():
            text = cache_file.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                return text
        return None

    def _save_cache(self, target: Path, text: str) -> None:
        """Persist extracted text to disk for fast re-use on next run."""
        cache_file = self._cache_path(target)
        try:
            cache_file.write_text(text, encoding="utf-8")
            print(f"[Codal] Cached extracted text to {cache_file.name} ({len(text)} chars).")
        except Exception as exc:
            print(f"[Codal] Warning: could not write cache for {target.name}: {exc}")

    # ------------------------------------------------------------------
    # PDF text-layer extraction
    # ------------------------------------------------------------------

    def _extract_pdf_text(self, target: Path) -> str:
        """
        Extract text from a PDF using text-layer parsers when available.

        Tries pdfplumber first (better layout preservation), then pypdf
        as a secondary attempt.  Both libraries silently return empty
        strings for scanned/image-only PDFs, which triggers the OCR path.
        """
        # --- pdfplumber ---
        try:
            import pdfplumber  # type: ignore
            print(f"[Codal] Attempting pdfplumber text extraction on {target.name}...")
            pages: List[str] = []
            with pdfplumber.open(str(target)) as pdf:
                total = len(pdf.pages)
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(text)
                    if i % 50 == 0 or i == total:
                        print(f"[Codal]   pdfplumber progress: {i}/{total} pages scanned, "
                              f"{len(pages)} pages with text so far.")
            result = "\n".join(pages)
            if result.strip():
                print(f"[Codal] pdfplumber extracted {len(result)} chars from {target.name}.")
                return result
            print(f"[Codal] pdfplumber extracted no text from {target.name} (likely scanned PDF).")
        except ImportError:
            print("[Codal] pdfplumber not available; trying pypdf.")
        except Exception as exc:
            print(f"[Codal] pdfplumber failed for {target.name}: {exc}")

        # --- pypdf ---
        try:
            from pypdf import PdfReader  # type: ignore
            import warnings
            print(f"[Codal] Attempting pypdf text extraction on {target.name}...")
            pages = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress truncation warnings
                reader = PdfReader(str(target), strict=False)
                total = len(reader.pages)
                for i, page in enumerate(reader.pages, 1):
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    if text.strip():
                        pages.append(text)
                    if i % 50 == 0 or i == total:
                        print(f"[Codal]   pypdf progress: {i}/{total} pages scanned, "
                              f"{len(pages)} pages with text so far.")
            result = "\n".join(pages)
            if result.strip():
                print(f"[Codal] pypdf extracted {len(result)} chars from {target.name}.")
                return result
            print(f"[Codal] pypdf extracted no text from {target.name} (likely scanned PDF).")
        except ImportError:
            print("[Codal] pypdf not available; OCR will be attempted.")
        except Exception as exc:
            print(f"[Codal] pypdf failed for {target.name}: {exc}")

        return ""

    def _ocr_pdf_text(self, target: Path) -> str:
        """
        OCR fallback for scanned/image-based standards PDFs.

        Renders each PDF page to an image and runs Tesseract on it.
        Requires: pytesseract + Tesseract binary + PyMuPDF (or poppler).
        """
        ocr_engine = self._load_ocr_engine()
        if ocr_engine is None:
            return ""

        print(f"[Codal] Rendering {target.name} pages to images for OCR...")
        rendered_pages = self._render_pdf_pages(target)
        if not rendered_pages:
            print("[Codal] No rendered pages available; cannot run OCR.")
            return ""

        total = len(rendered_pages)
        print(f"[Codal] Running Tesseract OCR on {total} page(s) from {target.name}...")
        text_chunks: List[str] = []
        for i, image in enumerate(rendered_pages, 1):
            raw_text = ocr_engine(image)
            if raw_text and raw_text.strip():
                text_chunks.append(raw_text)
            if i % 10 == 0 or i == total:
                print(f"[Codal]   OCR progress: {i}/{total} pages done, "
                      f"{len(text_chunks)} pages with text so far.")
        print(f"[Codal] OCR complete: {len(text_chunks)}/{total} pages yielded text.")
        return "\n".join(text_chunks)

    def _load_ocr_engine(self):
        """
        Return a callable(image) -> text when OCR is available, else None.

        Path resolution order:
          1. TESSERACT_PATH environment variable
          2. Hardcoded Windows installer default path
          3. System PATH (Linux / macOS)
        """
        try:
            import pytesseract  # type: ignore
        except ImportError:
            print("[Codal] pytesseract not installed; OCR unavailable.")
            return None

        # Resolve binary path: env var > hardcoded Windows default > system PATH
        env_path = os.environ.get("TESSERACT_PATH", "")
        if env_path and Path(env_path).exists():
            pytesseract.pytesseract.tesseract_cmd = env_path
            print(f"[Codal] Using Tesseract from TESSERACT_PATH env: {env_path}")
        elif Path(self._TESSERACT_FALLBACK_PATH).exists():
            pytesseract.pytesseract.tesseract_cmd = self._TESSERACT_FALLBACK_PATH
            print(f"[Codal] Using Tesseract from default install path: {self._TESSERACT_FALLBACK_PATH}")
        else:
            print("[Codal] Tesseract not found at default path; trying system PATH.")

        try:
            version = pytesseract.get_tesseract_version()
            print(f"[Codal] Tesseract OCR engine loaded (version {version}).")
            return lambda image: pytesseract.image_to_string(image)
        except Exception as exc:
            print(f"[Codal] Tesseract binary not found or failed: {exc}")
            print("[Codal] ACTION REQUIRED: Install Tesseract from "
                  "https://github.com/UB-Mannheim/tesseract/wiki "
                  "and set TESSERACT_PATH env var to the .exe path.")
            return None

    def _render_pdf_pages(self, target: Path):
        """
        Render PDF pages to PIL images for OCR.

        Backend priority:
          1. PyMuPDF (import name: pymupdf or fitz) - pure Python, no system deps
          2. pdf2image + poppler - requires poppler binaries in PATH
        """
        # --- PyMuPDF (pymupdf package) ---
        fitz = None
        try:
            import pymupdf as fitz  # type: ignore  # preferred import name
        except ImportError:
            try:
                import fitz  # type: ignore  # legacy PyMuPDF import name
            except ImportError:
                pass

        if fitz is not None:
            print(f"[Codal] Rendering pages with PyMuPDF (fitz) at 2x resolution...")
            doc = fitz.open(str(target))
            images = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                mode = "RGB" if pix.n < 4 else "RGBA"
                from PIL import Image
                images.append(Image.frombytes(mode, [pix.width, pix.height], pix.samples))
            print(f"[Codal] PyMuPDF rendered {len(images)} page(s).")
            return images

        # --- pdf2image + poppler ---
        try:
            from pdf2image import convert_from_path  # type: ignore
            print(f"[Codal] Rendering pages with pdf2image at 200 DPI...")
            images = convert_from_path(str(target), dpi=200)
            print(f"[Codal] pdf2image rendered {len(images)} page(s).")
            return images
        except ImportError:
            print("[Codal] pdf2image not available.")
        except Exception as exc:
            print(f"[Codal] pdf2image rendering failed: {exc}")
            print("[Codal] ACTION REQUIRED: Install poppler from "
                  "https://github.com/oschwartz10612/poppler-windows/releases "
                  "and add its bin/ folder to your system PATH.")

        print("[Codal] No PDF rendering backend available; OCR cannot proceed.")
        return []

    def fetch_csv_rules(self, path: str) -> List[Dict[str, Any]]:
        import pandas as pd

        df = pd.read_csv(path)
        return df.to_dict("records")

    def ingest_directory(self, dir_path: str) -> List[str]:
        contents = []
        for entry in os.listdir(dir_path):
            full_path = os.path.join(dir_path, entry)
            if entry.endswith(".pdf"):
                contents.append(self.fetch_local_pdf(full_path))
            elif entry.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8", errors="ignore") as file:
                    contents.append(file.read())
        return contents

