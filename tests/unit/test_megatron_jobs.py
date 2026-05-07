from art.megatron.runtime.jobs import (
    MegatronMergedTrainingJob,
    MegatronSyncJob,
    MegatronTrainingJob,
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
    dump_megatron_job,
    load_megatron_job,
)
from art.types import TrainConfig


def _merged_weight_transfer_spec() -> MergedWeightTransferSpec:
    return MergedWeightTransferSpec(
        init_info=MergedWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=2345,
            rank_offset=1,
            world_size=2,
        ),
        vllm_base_url="http://127.0.0.1:8000",
        served_model_name="test-model@1",
    )


def test_roundtrip_lora_training_job() -> None:
    job = MegatronTrainingJob(
        lora_path="/tmp/lora",
        optimizer_state_path="/tmp/opt",
        disk_packed_tensors={
            "dir": "/tmp/packed",
            "num_sequences": 2,
            "sequence_length": 128,
        },
        config=TrainConfig(
            learning_rate=1e-5,
            grad_accumulation_sequences=1,
        ),
        experimental_config={},
    )

    loaded = load_megatron_job(dump_megatron_job(job))

    assert isinstance(loaded, MegatronTrainingJob)
    assert loaded.kind == "train_lora"


def test_roundtrip_merged_and_sync_jobs() -> None:
    merged_job = MegatronMergedTrainingJob(
        lora_path="/tmp/lora",
        optimizer_state_path="/tmp/opt",
        disk_packed_tensors={
            "dir": "/tmp/packed",
            "num_sequences": 2,
            "sequence_length": 128,
        },
        config=TrainConfig(
            learning_rate=1e-5,
            grad_accumulation_sequences=1,
        ),
        experimental_config={},
        merged_weight_transfer=_merged_weight_transfer_spec(),
    )
    sync_job = MegatronSyncJob(
        lora_path="/tmp/lora",
        merged_weight_transfer=_merged_weight_transfer_spec(),
    )

    loaded_merged = load_megatron_job(dump_megatron_job(merged_job))
    loaded_sync = load_megatron_job(dump_megatron_job(sync_job))

    assert isinstance(loaded_merged, MegatronMergedTrainingJob)
    assert loaded_merged.kind == "train_merged"
    assert loaded_merged.merged_weight_transfer.served_model_name == "test-model@1"
    assert isinstance(loaded_sync, MegatronSyncJob)
    assert loaded_sync.kind == "sync"
