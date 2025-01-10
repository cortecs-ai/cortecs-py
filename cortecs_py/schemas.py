from datetime import datetime
from typing import Any

from pydantic import BaseModel

from cortecs_py.utils import convert_model_name


class InstanceStatus(BaseModel):
    status: str
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    init_progress: dict[str, Any] | None = None


class InstanceArgs(BaseModel):
    model_id: str
    hardware_type_id: str | None
    context_length: int | None
    billing_interval: str | None

    class Config:
        protected_namespaces = ()

    @property
    def hf_name(self) -> str:
        """Converts the internal model_id format to the original Hugging Face model name format."""
        return convert_model_name(self.model_id, to_hf_format=True)

    @classmethod
    def from_hf_name(cls, hf_name: str, **kwargs: dict[str, Any]) -> "InstanceArgs":
        """Creates an InstanceArgs object from a Hugging Face name, converting it to the internal format."""
        model_id = convert_model_name(hf_name, to_hf_format=False)
        return cls(model_id=model_id, **kwargs)


class Instance(BaseModel):
    instance_id: str
    base_url: str | None
    instance_args: InstanceArgs
    instance_status: InstanceStatus


class HardwareType(BaseModel):
    hardware_type_id: str
    price_info: dict[str, Any]
    hardware_info: dict[str, Any]


class HardwareConfig(BaseModel):
    hardware_type_id: str
    params: dict[str, Any]
    requirements: dict[str, Any]


class ModelPreview(BaseModel):
    model_id: str
    hf_name: str
    instant_provisioned: bool
    creator: dict[str, str] | None
    description: str | None
    screen_name: str | None
    size: int | None
    hardware_configs: list[str] | None

    class Config:
        protected_namespaces = ()

    @classmethod
    def from_raw_data(cls, raw_data: dict[str, Any]) -> "ModelPreview":  # noqa: N805
        # Set model_id from model_name if it exists
        if "model_name" in raw_data and raw_data["model_name"]:
            raw_data["model_id"] = raw_data["model_name"]

        # Restructure hardware info
        hardware_info = raw_data.get("hardware_info")
        if hardware_info:
            raw_data["hardware_configs"] = hardware_info.get("hardware_configs")

        return cls.model_validate(raw_data)


"""
class Model(BaseModel):
    model_id: str
    instant_provisioned: bool
    created_at: dict[str, datetime]
    screen_name: str
    model_name: str
    hf_name: str
    image_name: str
    license: str | None
    size: int
    creator: dict[str, str]
    quantization: str | None
    description: str
    tags: list[str] | None
    recommended_prompt: str | None
    prompt_example: str | None
    bits: str
    required_vram_gb: float
    required_disk_size: float
    ignore_patterns: list[str] | None
    variants: dict[str, str]
    recommended_variant: bool
    recommended_config: str
    hardware_configs: list[HardwareConfig]
    class Config:
        protected_namespaces = ()
"""
