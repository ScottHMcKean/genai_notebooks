from fastapi import APIRouter

# Import routers from versioned packages
from .v1 import router as v1_router

# Create a router for the API
api_router = APIRouter()

# Include versioned routers - prefix must have /api for Databricks Apps token-based auth
api_router.include_router(v1_router, prefix="/api/v1")
