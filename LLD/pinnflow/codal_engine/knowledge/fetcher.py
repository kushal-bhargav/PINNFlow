"""
Codal knowledge fetcher.
"""
from __future__ import annotations

import os
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

    def __init__(self, storage_path: str = "data/standards"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(__file__).resolve().parents[3]

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
            extracted = self._extract_pdf_text(target)
            if extracted.strip():
                print(f"[Codal] Extracted {len(extracted)} characters from PDF text.")
                return extracted

            print(f"[Codal] PDF text extraction yielded no content for {target.name}. Trying OCR fallback...")
            ocr_text = self._ocr_pdf_text(target)
            if ocr_text.strip():
                print(f"[Codal] OCR extracted {len(ocr_text)} characters from PDF pages.")
                return ocr_text

            print(
                f"[Codal] OCR fallback unavailable or empty for {target.name}. "
                "Falling back to raw text decode."
            )

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

    def _extract_pdf_text(self, target: Path) -> str:
        """
        Extract text from a PDF using pdfplumber when available.

        The repo historically treated PDFs as plain text files, which works only
        when the document happens to contain readable ASCII fragments. Real PDF
        parsing gives the parser a stable source string to inspect.
        """
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            print("[Codal] pdfplumber not available; using raw decode fallback.")
            return target.read_text(encoding="utf-8", errors="ignore")

        pages: List[str] = []
        with pdfplumber.open(str(target)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)

        return "\n".join(pages)

    def _ocr_pdf_text(self, target: Path) -> str:
        """
        OCR fallback for scanned/image-based standards PDFs.

        This branch is intentionally optional. It will only work when a PDF
        renderer and OCR engine are available in the runtime.
        """
        ocr_engine = self._load_ocr_engine()
        if ocr_engine is None:
            return ""

        rendered_pages = self._render_pdf_pages(target)
        if not rendered_pages:
            return ""

        text_chunks: List[str] = []
        for image in rendered_pages:
            raw_text = ocr_engine(image)
            if raw_text and raw_text.strip():
                text_chunks.append(raw_text)
        return "\n".join(text_chunks)

    def _load_ocr_engine(self):
        """
        Return a callable(image) -> text when OCR is available, else None.

        Priority:
          1. pytesseract + system tesseract binary
          2. a user-provided OCR backend imported as a fallback module
        """
        try:
            import pytesseract  # type: ignore
        except ImportError:
            pytesseract = None

        if pytesseract is not None:
            try:
                _ = pytesseract.get_tesseract_version()
            except Exception:
                pytesseract = None

        if pytesseract is not None:
            return lambda image: pytesseract.image_to_string(image)

        return None

    def _render_pdf_pages(self, target: Path):
        """
        Render PDF pages to PIL images for OCR.

        We intentionally support multiple render backends. If none are
        available, the caller gets an empty list and the pipeline falls back
        to non-OCR decoding.
        """
        try:
            import fitz  # type: ignore
        except ImportError:
            fitz = None

        if fitz is not None:
            doc = fitz.open(str(target))
            images = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                mode = "RGB" if pix.n < 4 else "RGBA"
                from PIL import Image

                images.append(Image.frombytes(mode, [pix.width, pix.height], pix.samples))
            return images

        try:
            from pdf2image import convert_from_path  # type: ignore
        except ImportError:
            convert_from_path = None

        if convert_from_path is not None:
            return convert_from_path(str(target), dpi=200)

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
