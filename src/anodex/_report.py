
from fpdf import FPDF
import json

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, 'Anomaly Explanation Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_json_table(self, data, title):
        self.chapter_title(title)
        self.set_font('courier', '', 8)
        json_str = json.dumps(data, indent=2)
        self.multi_cell(0, 5, json_str)
        self.ln()

    def add_image_section(self, images, title):
        self.chapter_title(title)
        for image_path in images:
            self.image(image_path, w=self.w - 40, x=20)
            self.ln(5)

def generate_report(meta_path, out_dir, pdf_path):
    """Generates a PDF report from the metadata and artifacts."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    pdf = PDF()
    pdf.add_page()

    pdf.add_json_table(meta['selection'], 'Anomaly Selection')
    if meta.get('counterfactual'):
        pdf.add_json_table(meta['counterfactual'], 'Counterfactual Summary')
    
    image_dir = out_dir / 'figs'
    if image_dir.exists():
        images = sorted(list(image_dir.glob('*.png')))
        if images:
            pdf.add_page()
            pdf.add_image_section(images, 'Figures')

    pdf.output(pdf_path)
