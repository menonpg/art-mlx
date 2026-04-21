from art.megatron import train as megatron_train


class _FakeProvider:
    def __init__(self) -> None:
        self.hooks: list[object] = []

    def register_pre_wrap_hook(self, hook: object) -> None:
        self.hooks.append(hook)


def test_register_trainable_parameter_mode_base_model_skips_hooks() -> None:
    provider = _FakeProvider()

    megatron_train._register_trainable_parameter_mode(
        provider,
        trainable_parameter_mode="base_model",
    )

    assert provider.hooks == []


def test_register_trainable_parameter_mode_lora_registers_freeze_and_adapter_hooks() -> None:
    provider = _FakeProvider()

    megatron_train._register_trainable_parameter_mode(
        provider,
        trainable_parameter_mode="lora",
    )

    assert provider.hooks[0] is megatron_train.freeze_model
    assert len(provider.hooks) == 2
