#Function to convert pdf file to images.

from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

async def convert_pdf_to_images(
    pdf_file: bytes,
    dpi: int = 300,
    output_format: str = "PNG"
) -> List[str]:
    """
    Convert PDF pages to base64 encoded images.
    
    Args:
        pdf_file (bytes): PDF file content in bytes
        dpi (int): Resolution for the output images (default: 300)
        output_format (str): Output image format (default: "PNG")
        
    Returns:
        List[str]: List of base64 encoded images, one per page
    
    Raises:
        ValueError: If the PDF is invalid or empty
        Exception: For other processing errors
    """
    try:
        # Load PDF from bytes
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        
        if pdf_document.page_count == 0:
            raise ValueError("PDF document is empty")
            
        images = []
        zoom = dpi / 72  # PDF standard DPI is 72

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Convert PDF page to image
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to base64
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=output_format)
            img_byte_arr = img_byte_arr.getvalue()
            base64_encoded = base64.b64encode(img_byte_arr).decode('utf-8')
            
            images.append(base64_encoded)
            
        pdf_document.close()
        return images
        
    except fitz.FileDataError:
        raise ValueError("Invalid PDF file")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")