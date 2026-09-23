from gutenTAG.tsgen.seeding import derive_instance_seeds


def test_matched_split_parameter_policy_preserves_parameter_draws():
    train = derive_instance_seeds(
        master_seed=42,
        variant_id="sine__mean__p00",
        split="train",
        instance_index=3,
        split_parameter_policy="matched",
    )
    test = derive_instance_seeds(
        master_seed=42,
        variant_id="sine__mean__p00",
        split="test",
        instance_index=3,
        split_parameter_policy="matched",
    )
    assert train["base_params_seed"] == test["base_params_seed"]
    assert train["base_channel_params_seed"] == test["base_channel_params_seed"]
    assert train["base_seed"] != test["base_seed"]


def test_independent_split_parameter_policy_separates_parameter_draws():
    train = derive_instance_seeds(
        master_seed=42,
        variant_id="sine__mean__p00",
        split="train",
        instance_index=3,
        split_parameter_policy="independent",
    )
    test = derive_instance_seeds(
        master_seed=42,
        variant_id="sine__mean__p00",
        split="test",
        instance_index=3,
        split_parameter_policy="independent",
    )
    assert train["base_params_seed"] != test["base_params_seed"]
    assert train["base_channel_params_seed"] != test["base_channel_params_seed"]


def test_split_parameter_policy_rejects_unknown_value():
    try:
        derive_instance_seeds(
            master_seed=42,
            variant_id="sine__mean__p00",
            split="train",
            instance_index=0,
            split_parameter_policy="unknown",
        )
    except ValueError as exc:
        assert "split_parameter_policy" in str(exc)
    else:
        raise AssertionError("unknown split parameter policy must fail")
