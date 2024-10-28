from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class RoomType(str, Enum):
    """Enum for room types"""
    ENTIRE_HOME = "Entire home/apt"
    PRIVATE_ROOM = "Private room"
    SHARED_ROOM = "Shared room"


class Neighbourhood(str, Enum):
    """Enum for neighbourhoods"""
    BRONX = "Bronx"
    BROOKLYN = "Brooklyn"
    MANHATTAN = "Manhattan"
    QUEENS = "Queens"
    STATEN_ISLAND = "Staten Island"


class InputData(BaseModel):
    """Model for input data"""
    id: int = Field(..., description="Unique identifier of the listing")
    neighbourhood: Neighbourhood = Field(
        ...,
        description="Neighborhood of the listing"
    )
    room_type: RoomType = Field(..., description="Type of room")
    accommodates: int = Field(
        ...,
        ge=1,
        description="Number of people it can accommodate"
    )
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    bedrooms: int = Field(..., ge=1, description="Number of bedrooms")
    beds: int = Field(..., ge=1, description="Number of beds")
    tv: int = Field(..., ge=0, le=1, description="TV availability (0 or 1)")
    elevator: int = Field(..., ge=0, le=1, description="Elevator availability")
    internet: int = Field(..., ge=0, le=1, description="Internet availability")
    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude coordinate (-90 to 90)"
    )
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude coordinate (-180 to 180)"
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class OutputData(BaseModel):
    """Model for output data"""
    id: int = Field(..., description="Unique identifier of the listing")
    price_category: str = Field(..., description="Predicted price category")


class BatchInputData(BaseModel):
    """Model for batch input data"""
    data: List[InputData] = Field(
        ...,
        description="List of input data for batch prediction"
    )


class BatchOutputData(BaseModel):
    """Model for batch output data"""
    results: List[OutputData] = Field(
        ...,
        description="List of batch prediction results"
    )
