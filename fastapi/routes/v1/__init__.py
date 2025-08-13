"""V1 API routes."""

from fastapi import APIRouter

from .healthcheck import router as healthcheck_router

router = APIRouter()

# Include endpoint-specific routers
router.include_router(healthcheck_router)
