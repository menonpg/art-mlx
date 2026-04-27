import art.weight_transfer.nccl as nccl


def test_trainer_nccl_unique_id_round_trips_as_raw_bytes() -> None:
    payload = bytes(range(128))
    unique_id = nccl._nccl_unique_id_from_bytes(payload)
    assert nccl._nccl_unique_id_to_bytes(unique_id) == payload
