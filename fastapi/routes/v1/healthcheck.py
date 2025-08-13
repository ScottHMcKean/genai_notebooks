from datetime import datetime, timezone
from fastapi import APIRouter
from typing import Dict

router = APIRouter()


@router.get("/healthcheck")
async def healthcheck() -> Dict[str, str]:
    """Return the API status."""
    return {"status": "OK", "timestamp": datetime.now(timezone.utc).isoformat()}
