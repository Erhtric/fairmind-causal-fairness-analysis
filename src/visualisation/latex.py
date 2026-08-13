from pathlib import Path
import tempfile
from src.llm import LLMReportResult
import os
import shutil
from pdflatex import PDFLaTeX


def compile_report_to_pdf(report: LLMReportResult, output_path: str | Path) -> None:
    """
    Compiles the LaTeX report to a PDF file.

    Args:
        report (LLMReportResult): The report containing LaTeX content.
        output_path (str | Path): The path where the PDF will be saved.
    """
    temp_dir = tempfile.mkdtemp(prefix="latex_report_")

    try:
        temp_tex_path = os.path.join(temp_dir, "report.tex")
        with open(temp_tex_path, "w") as f:
            print(f"Writing LaTeX report to temporary file: {temp_tex_path}")
            f.write(report.latex)

        print(f"Creating PDF from LaTeX file: {temp_tex_path}")
        pdfl = PDFLaTeX.from_texfile(temp_tex_path)
        pdf, _, _ = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)

        # If it exists, remove the existing PDF file before writing the new one
        if os.path.exists(output_path):
            print(f"Removing existing PDF report at: {output_path}")
            os.remove(output_path)

        with open(output_path, "wb") as f:
            print(f"Saving PDF report to: {output_path}")
            f.write(pdf)

        print(f"PDF report generated and saved to: {output_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
