from pathlib import Path
from types import SimpleNamespace

import pytest

from art.megatron.client import stream_megatron_job, write_megatron_job
from art.megatron.jobs import (
    MegatronSyncJob,
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
)


@pytest.mark.asyncio
async def test_stream_megatron_job_raises_when_worker_exits(
    tmp_path: Path,
) -> None:
    job_path = tmp_path / "job.json"
    log_path = tmp_path / "job.log"
    job = MegatronSyncJob(
        lora_path="/tmp/lora",
        merged_weight_transfer=MergedWeightTransferSpec(
            init_info=MergedWeightTransferInitInfo(
                master_address="127.0.0.1",
                master_port=12345,
                rank_offset=1,
                world_size=2,
            ),
            vllm_base_url="http://127.0.0.1:8000",
            served_model_name="test@0",
        ),
        log_path=str(log_path),
    )
    write_megatron_job(job, job_path=str(job_path))

    with pytest.raises(RuntimeError, match="Megatron worker exited with code 17"):
        async for _ in stream_megatron_job(
            job,
            job_path=str(job_path),
            process=SimpleNamespace(returncode=17),
            process_log_path="/tmp/megatron-runtime.log",
            poll_interval=0.0,
        ):
            pass
