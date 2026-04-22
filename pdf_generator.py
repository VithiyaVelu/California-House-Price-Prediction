"""PDF Report Generator for House Price Predictions"""
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def generate_pdf_report(report_text: str) -> bytes:
    """Generate a PDF report from the text report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=6,
        spaceBefore=6
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=12
    )
    
    elements = []
    elements.append(Paragraph("California House Price Prediction Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    lines = report_text.split('\n')
    for line in lines:
        if line.startswith('==='):
            continue
        elif line.startswith('- '):
            elements.append(Paragraph(f"• {line[2:]}", normal_style))
        elif any(line.startswith(header) for header in ['Prediction Details:', 'Input Features:', 'Engineered Features:', 'What-if Analysis', 'Generated on:']):
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(line, heading_style))
        elif line.strip():
            elements.append(Paragraph(line, normal_style))
        else:
            elements.append(Spacer(1, 0.05*inch))
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
