"""Pydantic models for structured information extraction"""

from typing import List
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """A single extracted entity with metadata"""

    name: str = Field(description="The entity name or key information")
    type: str = Field(
        description="Entity type (e.g., person, organization, date, location)"
    )
    value: str = Field(description="The extracted value or detail")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class StructuredInformation(BaseModel):
    """Structured information extracted from documents"""

    summary: str = Field(description="Concise summary of the information")
    key_points: List[str] = Field(description="List of key points (3-5 items)")
    entities: List[ExtractedEntity] = Field(
        description="Extracted entities with metadata"
    )
    sources: List[str] = Field(description="List of source document titles")

