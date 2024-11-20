# file_processor.py

import base64
from typing import List, Optional
from fastapi import UploadFile
import imghdr
from pdf_img_converter import convert_pdf_to_images

async def process_file(file: Optional[UploadFile] = None) -> List[str]:
    """
    Process uploaded file (PDF or image) and return list of base64 encoded images.
    
    Args:
        file (UploadFile, optional): Uploaded file (PDF or image)
        
    Returns:
        List[str]: List of base64 encoded images
        
    Raises:
        ValueError: If file format is invalid
    """
    if not file:
        return []
        
    content_type = file.content_type
    file_content = await file.read()
    
    # Handle PDF
    if content_type == "application/pdf":
        return await convert_pdf_to_images(file_content)
        
    # Handle images
    elif content_type.startswith('image/'):
        # Verify it's actually an image
        img_format = imghdr.what(None, h=file_content)
        if not img_format:
            raise ValueError("Invalid image file")
            
        # Convert to base64
        base64_image = base64.b64encode(file_content).decode('utf-8')
        return [base64_image]
        
    else:
        raise ValueError("Unsupported file type. Please upload PDF or image files only.")