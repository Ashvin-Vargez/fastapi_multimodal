
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List

app = FastAPI()

@app.post("/convert-pdf/", response_model=List[str])
async def convert_pdf_endpoint(
    file: UploadFile = File(...),
    dpi: int = 300
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    try:
        contents = await file.read()
        images = await app.services.pdf_img_converter.convert_pdf_to_images(contents, dpi=dpi)
        return images
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

