import fitz


def extract_text(uploaded_pdf):
    try:
        pdf_doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        text = ""
        
        for page in pdf_doc:
            text += page.get_text()
        
        pdf_doc.close()
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

