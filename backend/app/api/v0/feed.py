import random
from fastapi import APIRouter

router = APIRouter(prefix="/feed", tags=["feed"])

FAKE_FEED = [
    {
        "content_id": "a0",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2022/02/14/107702-678540933_large.mp4",
        "duration": 8.2,
        "creator": "ai_system",
        "tags": ["animal", "nature"],
    },
    {
        "content_id": "a1",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2025/02/12/257851_large.mp4",
        "duration": 11.5,
        "creator": "ai_system",
        "tags": ["music", "human"],
    },
    {
        "content_id": "a2",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2024/09/15/231531_large.mp4",
        "duration": 6.9,
        "creator": "ai_system",
        "tags": ["nature", "animation"],
    },
    {
        "content_id": "a3",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2024/09/13/231156_large.mp4",
        "duration": 1.8,
        "creator": "ai_system",
        "tags": ["nature", "animation"],
    },
    {
        "content_id": "a4",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2019/08/15/26080-357512264_large.mp4",
        "duration": 3.7,
        "creator": "ai_system",
        "tags": ["nature", "low-poly"],
    },
    {
        "content_id": "a5",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2024/08/22/227818_large.mp4",
        "duration": 10.4,
        "creator": "ai_system",
        "tags": ["car", "low-poly"],
    },
    {
        "content_id": "a6",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2024/04/28/209790_large.mp4",
        "duration": 7.1,
        "creator": "ai_system",
        "tags": ["nature", "abstract"],
    },
    {
        "content_id": "a7",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2023/08/18/176534-855914677_large.mp4",
        "duration": 2.2,
        "creator": "ai_system",
        "tags": ["nature", "space"],
    },
    {
        "content_id": "a8",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2015/11/27/1405-147169806_medium.mp4",
        "duration": 14.1,
        "creator": "ai_system",
        "tags": ["human", "dark"],
    },
    {
        "content_id": "a9",
        "type": "video",
        "video_url": "https://cdn.pixabay.com/video/2025/01/27/254934_large.mp4",
        "duration": 4.0,
        "creator": "ai_system",
        "tags": ["abstract", "art"],
    }
]

@router.get("")
def get_feed():
    item = random.choice(FAKE_FEED)
    return {
        **item,
        "score": round(random.random(), 2),
    }
