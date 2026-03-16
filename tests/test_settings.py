from settings import Settings


def test_settings_loaded():
    settings = Settings()
    assert settings.ENVIRONMENT == "test"
    assert settings.APP_NAME == "TestApp"
    assert settings.API_KEY == "fake-test-api-key-123"


def test_settings_environment_valid():
    settings = Settings()
    assert settings.ENVIRONMENT in ("dev", "test", "prod")
