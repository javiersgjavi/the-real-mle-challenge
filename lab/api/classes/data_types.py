from pydantic import BaseModel, Field
from typing import List


class InputData(BaseModel):

    # key parameters

    id: int = Field(..., description="Unique identifier of the listing")
    neighbourhood: str = Field(..., description="Neighborhood of the listing")
    room_type: str = Field(..., description="Type of room")
    accommodates: int = Field(
        ..., ge=1,
        description="Number of people it can accommodate"
    )
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    bedrooms: int = Field(..., ge=1, description="Number of bedrooms")

    beds: int = Field(..., ge=1, description="Number of beds")
    tv: int = Field(..., ge=0, le=1, description="TV availability (0 or 1)")
    elevator: int = Field(..., ge=0, le=1, description="Elevator availability")
    internet: int = Field(..., ge=0, le=1, description="Internet availability")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")


class OutputData(BaseModel):
    id: int = Field(..., description="Unique identifier of the listing")
    price_category: str = Field(..., description="Predicted price category")


class BatchInputData(BaseModel):
    data: List[InputData] = Field(
        ...,
        description="List of input data for batch prediction"
    )


class BatchOutputData(BaseModel):
    results: List[OutputData] = Field(
        ...,
        description="List of batch prediction results"
    )

