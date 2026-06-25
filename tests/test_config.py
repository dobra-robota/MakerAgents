from makeragents.config import AppConfig, load_config


def test_config_loads_defaults_without_api_keys() -> None:
    config = AppConfig.from_env({})

    assert config.openai_api_key is None
    assert config.deepseek_api_key is None
    assert config.default_llm_provider == "openai"
    assert config.default_llm_model is None
    assert config.deepseek_model == "deepseek-chat"
    assert config.brave_search_api_key is None


def test_optional_config_values_treat_whitespace_as_missing() -> None:
    config = AppConfig.from_env({"DEFAULT_LLM_MODEL": "   "})

    assert config.default_llm_model is None


def test_config_loads_expected_environment_variables() -> None:
    config = load_config(
        {
            "OPENAI_API_KEY": "openai-test-key",
            "DEEPSEEK_API_KEY": "deepseek-test-key",
            "DEFAULT_LLM_PROVIDER": "deepseek",
            "DEFAULT_LLM_MODEL": " gpt-placeholder ",
            "DEEPSEEK_MODEL": "deepseek-reasoner",
            "BRAVE_SEARCH_API_KEY": "brave-test-key",
        }
    )

    assert config.openai_api_key == "openai-test-key"
    assert config.deepseek_api_key == "deepseek-test-key"
    assert config.default_llm_provider == "deepseek"
    assert config.default_llm_model == "gpt-placeholder"
    assert config.deepseek_model == "deepseek-reasoner"
    assert config.brave_search_api_key == "brave-test-key"
