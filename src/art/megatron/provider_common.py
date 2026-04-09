from typing import Any

from pydantic import BaseModel, ConfigDict

from art.megatron.model_support.spec import ModelSupportSpec


class ProviderBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any
    bridge: Any
    handler: Any
    spec: ModelSupportSpec
