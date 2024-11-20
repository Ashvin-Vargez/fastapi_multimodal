import asyncio
from app.services.pdf_img_converter import convert_pdf_to_images
import base64

async def test_conversion():
    # Replace with your PDF file path
    pdf_path = "app/services/BayColon_pg1_2.pdf"
    
    # Read the PDF file
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    # Convert to images
    images = await convert_pdf_to_images(pdf_content)
    
    # Save each page as an image
    for i, img_base64 in enumerate(images):
        # Decode base64 and save as image
        with open(f"page_{i+1}.png", "wb") as f:
            f.write(base64.b64decode(img_base64))
        print(f"Saved page {i+1} as page_{i+1}.png")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_conversion())