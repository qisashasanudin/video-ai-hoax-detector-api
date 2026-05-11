import asyncio
from app.media_pipeline import extract_media

async def run():
    try:
        res = await extract_media(
            url="https://www.instagram.com/reels/DYLV39xyLPR/",
            job_id="test_ig_1",
            base_data_dir="./test_ig_data"
        )
        print("Success:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(run())
