from fastapi import FastAPI

from routes import api_router

app = FastAPI(
    title="FastAPI & Databricks Apps",
    description="A simple FastAPI application example for Databricks Apps runtime",
    version="1.0.0",
)

# Router assignment
app.include_router(api_router)
