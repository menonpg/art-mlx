"""ART harness Megatron worker entrypoint with CE fusion implementation override."""

from __future__ import annotations

import os
import runpy
from typing import Any

CE_IMPL_ENV = "ART_HARNESS_CROSS_ENTROPY_FUSION_IMPL"
HARNESS_ENTRYPOINT = (
    "/mnt/ws_pvc/ws/projects/art_harness/art_harness/"
    "megatron_train_with_provider_patch.py"
)


def _install_ce_impl_override() -> None:
    impl = os.environ.get(CE_IMPL_ENV, "").strip()
    if not impl:
        return

    import art.megatron.provider as provider_module

    original_prepare_provider_bundle = provider_module.prepare_provider_bundle

    def prepare_provider_bundle_with_ce_impl(*args: Any, **kwargs: Any) -> Any:
        bundle = original_prepare_provider_bundle(*args, **kwargs)
        bundle.provider.cross_entropy_loss_fusion = True
        bundle.provider.cross_entropy_fusion_impl = impl
        return bundle

    provider_module.prepare_provider_bundle = prepare_provider_bundle_with_ce_impl


def main() -> int:
    _install_ce_impl_override()
    runpy.run_path(HARNESS_ENTRYPOINT, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
