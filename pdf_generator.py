# pdf_generator.py - FIXED
"""Professional PDF generation with proper error handling"""
import logging
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    KeepTogether
)
from reportlab.lib import colors

logger = logging.getLogger(__name__)


class PDFGenerator:
    """Generate professional proposal documents"""
    
    def __init__(self, config):
        self.config = config
        self.styles = self._create_styles()
    
    def _create_styles(self):
        """Create custom paragraph styles"""
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=20,
            spaceBefore=10,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ))
        
        # Question number style
        styles.add(ParagraphStyle(
            name='QuestionNumber',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666'),
            spaceAfter=5,
            fontName='Helvetica-Bold'
        ))
        
        # Question text style
        styles.add(ParagraphStyle(
            name='QuestionText',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            fontName='Helvetica-Bold',
            leftIndent=0
        ))
        
        # Answer style
        styles.add(ParagraphStyle(
            name='Answer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#34495e'),
            alignment=TA_JUSTIFY,
            leftIndent=0,
            spaceAfter=15,
            leading=14
        ))
        
        # Citation style
        styles.add(ParagraphStyle(
            name='Citation',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#7f8c8d'),
            leftIndent=20,
            spaceAfter=20,
            fontName='Helvetica-Oblique'
        ))
        
        # Error style
        styles.add(ParagraphStyle(
            name='Error',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#e74c3c'),
            leftIndent=20,
            spaceAfter=15,
            fontName='Helvetica-Oblique'
        ))
        
        return styles
    
    def _escape_text(self, text: str) -> str:
        """Escape text for PDF rendering"""
        if not text:
            return ""
        
        text = str(text)
        
        # Escape special XML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Handle newlines
        text = text.replace('\n', '<br/>')
        
        return text
    
    def _format_citations_in_text(self, text: str) -> str:
        """Format [Source N] citations with styling"""
        import re
        
        # Replace [Source N] with styled version
        pattern = r'\[Source (\d+)\]'
        text = re.sub(
            pattern,
            r'<font color="#2980b9"><b>[Source \1]</b></font>',
            text
        )
        
        return text
    
    def generate_qa_pdf(
        self, 
        results: List[Dict], 
        output_path: str,
        title: str = "RFP Response Document"
    ):
        """Generate professional Q&A PDF with citations"""
        
        # Debug logging
        logger.info(f"Generating PDF with {len(results)} results")
        for i, r in enumerate(results, 1):
            logger.info(f"  Result {i}:")
            logger.info(f"    Question: {r.get('question', 'N/A')[:80]}...")
            logger.info(f"    Answer length: {len(r.get('answer', ''))}")
            logger.info(f"    Citations: {len(r.get('citations', []))}")
            
            # Check for errors
            answer = r.get('answer', '')
            if 'Error' in answer or 'error' in answer.lower():
                logger.warning(f"    ⚠ Result {i} contains error text")
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
            title=title
        )
        
        story = []
        
        # Cover page
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Metadata
        date_str = datetime.now().strftime("%B %d, %Y")
        metadata_text = f"Generated: {date_str}<br/>Total Questions: {len(results)}"
        story.append(Paragraph(metadata_text, self.styles['Normal']))
        story.append(Spacer(1, 0.5 * inch))
        
        # Summary
        error_count = sum(1 for r in results if 'error' in r.get('answer', '').lower())
        if error_count > 0:
            summary = f"<font color='red'><b>Note:</b> {error_count} question(s) had errors during processing.</font>"
            story.append(Paragraph(summary, self.styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))
        
        story.append(PageBreak())
        
        # Q&A Content
        for idx, item in enumerate(results, 1):
            qa_section = self._create_qa_section(idx, item)
            story.extend(qa_section)
        
        # Build PDF
        try:
            doc.build(story)
            logger.info(f"✓ PDF generated successfully: {output_path}")
            
            # Verify file was created
            if Path(output_path).exists():
                size = Path(output_path).stat().st_size
                logger.info(f"  File size: {size:,} bytes")
            else:
                logger.error(f"  ✗ PDF file not found after generation!")
                
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_qa_section(self, idx: int, item: Dict) -> List:
        """Create a single Q&A section with citations"""
        elements = []
        
        # Question number and metadata
        q_metadata = item.get('metadata', {})
        q_number_text = f"Question {idx}"
        
        if q_metadata:
            if page := q_metadata.get('page'):
                q_number_text += f" (RFP Page {page}"
                if section := q_metadata.get('section'):
                    q_number_text += f", {section}"
                q_number_text += ")"
        
        elements.append(Paragraph(q_number_text, self.styles['QuestionNumber']))
        
        # Question text
        question_text = self._escape_text(item.get('question', 'No question provided'))
        elements.append(Paragraph(
            f"<b>Q:</b> {question_text}",
            self.styles['QuestionText']
        ))
        
        # Answer
        answer_text = item.get('answer', 'No answer generated')
        
        # Check if it's an error
        if answer_text.startswith('Error'):
            logger.warning(f"Question {idx} has error: {answer_text}")
            elements.append(Paragraph(
                f"<b>Status:</b> {self._escape_text(answer_text)}",
                self.styles['Error']
            ))
            elements.append(Paragraph(
                "<i>This question could not be answered due to a processing error. "
                "Please check the logs for more details.</i>",
                self.styles['Error']
            ))
        else:
            # Normal answer
            answer_text = self._escape_text(answer_text)
            answer_text = self._format_citations_in_text(answer_text)
            
            elements.append(Paragraph(
                f"<b>A:</b> {answer_text}",
                self.styles['Answer']
            ))
        
        # Citations
        include_citations = getattr(self.config, 'include_citations', True)
        citations = item.get('citations', [])
        
        if include_citations and citations:
            citations_text = "<b>Sources:</b><br/>"
            for i, citation in enumerate(citations, 1):
                citation = self._escape_text(citation)
                citations_text += f"[{i}] {citation}<br/>"
            
            elements.append(Paragraph(citations_text, self.styles['Citation']))
        
        elements.append(Spacer(1, 0.3 * inch))
        
        # Keep question and answer together
        return [KeepTogether(elements)]
    
    def generate_simple_pdf(
        self,
        results: List[Dict],
        output_path: str
    ):
        """Generate simple text-based PDF (fallback)"""
        from reportlab.pdfgen import canvas
        
        logger.info(f"Generating simple PDF with {len(results)} results")
        
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        
        margin = inch
        y = height - margin
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "RFP Response Document")
        y -= 30
        
        c.setFont("Helvetica", 10)
        date_str = datetime.now().strftime("%B %d, %Y")
        c.drawString(margin, y, f"Generated: {date_str}")
        y -= 20
        c.drawString(margin, y, f"Total Questions: {len(results)}")
        y -= 40
        
        for idx, item in enumerate(results, 1):
            # Check if new page needed
            if y < 2 * inch:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - margin
            
            # Question
            c.setFont("Helvetica-Bold", 11)
            q_text = f"Q{idx}. {item.get('question', 'N/A')}"
            
            # Wrap text
            for line in self._wrap_text(q_text, 90):
                if y < 1.5 * inch:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 11)
                    y = height - margin
                c.drawString(margin, y, line)
                y -= 15
            
            y -= 5
            
            # Answer
            c.setFont("Helvetica", 10)
            a_text = f"A: {item.get('answer', 'N/A')}"
            
            for line in self._wrap_text(a_text, 95):
                if y < 1.5 * inch:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - margin
                c.drawString(margin + 10, y, line)
                y -= 14
            
            y -= 25
        
        c.save()
        logger.info(f"✓ Simple PDF saved to: {output_path}")
    
    def _wrap_text(self, text: str, max_width: int = 90) -> List[str]:
        """Wrap text to fit within width"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [""]