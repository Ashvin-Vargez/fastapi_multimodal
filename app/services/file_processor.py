# file_processor.py

import base64
from typing import List
from fastapi import UploadFile
import imghdr
from pdf_img_converter import convert_pdf_to_images

async def process_files(files: List[UploadFile] = None) -> List[str]:
    """
    Process multiple uploaded files (PDFs and/or images) and return list of base64 encoded images.
    
    Args:
        files (List[UploadFile], optional): List of uploaded files (PDFs and/or images)
        
    Returns:
        List[str]: List of base64 encoded images
        
    Raises:
        ValueError: If file format is invalid
    """
    if not files:
        return []
    
    base64_images = []
    
    for file in files:
        content_type = file.content_type
        file_content = await file.read()
        
        # Handle PDF
        if content_type == "application/pdf":
            pdf_images = await convert_pdf_to_images(file_content)
            base64_images.extend(pdf_images)
            
        # Handle images
        elif content_type.startswith('image/'):
            # Verify it's actually an image
            img_format = imghdr.what(None, h=file_content)
            if not img_format:
                raise ValueError(f"Invalid image file: {file.filename}")
                
            # Convert to base64
            base64_image = base64.b64encode(file_content).decode('utf-8')
            base64_images.append(base64_image)
            
        else:
            raise ValueError(f"Unsupported file type for {file.filename}. Please upload PDF or image files only.")
    
    return base64_images