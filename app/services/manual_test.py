import asyncio
from app.services.pdf_img_converter import convert_pdf_to_images

async def test_with_your_pdf():
    with open("your_pdf_file.pdf", "rb") as f:
        pdf_content = f.read()
    
    images = await convert_pdf_to_images(pdf_content)
    
    # Save the first page as an image
    import base64
    with open("output_page1.png", "wb") as f:
        f.write(base64.b64decode(images[0]))

if __name__ == "__main__":
    asyncio.run(test_with_your_pdf())