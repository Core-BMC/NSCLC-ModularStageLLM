"""Data models for patient information and medical reports."""

import logging
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class MedicalReport(BaseModel):
    """Medical report model with content validation."""

    content: Optional[str] = Field(None, description="Medical report content")

    @field_validator('content')
    def remove_empty_lines_and_convert_to_string(cls, v):
        """Remove empty lines and convert to string."""
        if v is None or pd.isna(v):
            return None
        return '\n'.join(line for line in str(v).split('\n') if line.strip())


class InputData(BaseModel):
    """Input data model for patient case information."""

    case_number: int
    hospital_id: str = Field(..., description="hospital_id")
    pathology: Optional[MedicalReport] = Field(None, description="pathology")
    chest_ct: Optional[MedicalReport] = Field(None, description="chest_ct")
    brain_mr: Optional[MedicalReport] = Field(None, description="brain_mr")
    pet: Optional[MedicalReport] = Field(None, description="pet")
    ebus: Optional[MedicalReport] = Field(None, description="ebus")
    neck_biopsy: Optional[MedicalReport] = Field(
        None, description="neck_biopsy"
    )
    bone_scan: Optional[MedicalReport] = Field(None, description="bone_scan")
    abdomen_pelvis_ct: Optional[MedicalReport] = Field(
        None, description="abdomen_pelvis_ct"
    )
    adrenal_ct: Optional[MedicalReport] = Field(
        None, description="adrenal_ct"
    )

    @field_validator('*')
    def handle_nan(cls, v):
        """Handle NaN values in input data."""
        if pd.isna(v):
            return None
        return v

