import tempfile
from src.llm import LLMReportResult
import os
import shutil
from pdflatex import PDFLaTeX


def compile_report_to_pdf(report: LLMReportResult, output_path: str) -> None:
    """
    Compiles the LaTeX report to a PDF file.

    Args:
        report (LLMReportResult): The report containing LaTeX content.
        output_path (str): The path where the PDF will be saved.
    """
    temp_dir = tempfile.mkdtemp()

    try:
        temp_tex_path = os.path.join(temp_dir, "report.tex")
        with open(temp_tex_path, "w") as f:
            f.write(report.latex)

        pdfl = PDFLaTeX.from_texfile(temp_tex_path)
        pdf, _, _ = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)

        with open(output_path, "wb") as f:
            f.write(pdf)

        print(f"PDF report generated and saved to: {output_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
