"""Tests for the Webhook Notifier module.

Tests cover:
- WebhookConfig creation from environment and dictionaries
- WebhookNotifier lifecycle (start/stop)
- Event subscription and handling
- Payload building with proper formatting
- HTTP sending with retries

Issue: #3090
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from framework.runtime.event_bus import AgentEvent, EventBus, EventType
from framework.runtime.webhook import (
    WebhookConfig,
    WebhookNotifier,
    create_webhook_notifier,
)


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WebhookConfig(url="https://example.com/webhook")

        assert config.url == "https://example.com/webhook"
        assert config.events == ["on_start", "on_complete", "on_failure", "on_pause"]
        assert config.timeout_seconds == 10.0
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 1.0
        assert config.headers == {}
        assert config.include_output is False
        assert config.agent_name is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WebhookConfig(
            url="https://slack.com/webhook",
            events=["on_complete", "on_failure"],
            timeout_seconds=5.0,
            retry_count=5,
            retry_delay_seconds=2.0,
            headers={"Authorization": "Bearer token"},
            include_output=True,
            agent_name="Test Agent",
        )

        assert config.url == "https://slack.com/webhook"
        assert config.events == ["on_complete", "on_failure"]
        assert config.timeout_seconds == 5.0
        assert config.retry_count == 5
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.include_output is True
        assert config.agent_name == "Test Agent"


class TestWebhookConfigFromEnv:
    """Tests for WebhookConfig.from_env()."""

    def test_from_env_with_url(self):
        """Test loading config from environment with URL set."""
        with patch.dict(
            os.environ,
            {
                "HIVE_WEBHOOK_URL": "https://test.webhook.com",
                "HIVE_WEBHOOK_EVENTS": "on_complete,on_failure",
                "HIVE_WEBHOOK_TIMEOUT": "15",
                "HIVE_WEBHOOK_INCLUDE_OUTPUT": "true",
            },
        ):
            config = WebhookConfig.from_env(agent_name="EnvAgent")

            assert config is not None
            assert config.url == "https://test.webhook.com"
            assert config.events == ["on_complete", "on_failure"]
            assert config.timeout_seconds == 15.0
            assert config.include_output is True
            assert config.agent_name == "EnvAgent"

    def test_from_env_without_url(self):
        """Test that None is returned when URL not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Make sure HIVE_WEBHOOK_URL is not in environment
            os.environ.pop("HIVE_WEBHOOK_URL", None)
            config = WebhookConfig.from_env()
            assert config is None

    def test_from_env_with_invalid_timeout(self):
        """Test that invalid timeout falls back to default."""
        with patch.dict(
            os.environ,
            {
                "HIVE_WEBHOOK_URL": "https://test.com",
                "HIVE_WEBHOOK_TIMEOUT": "invalid",
            },
        ):
            config = WebhookConfig.from_env()
            assert config is not None
            assert config.timeout_seconds == 10.0

    def test_from_env_include_output_variations(self):
        """Test various truthy values for include_output."""
        for value in ["true", "1", "yes", "TRUE", "Yes"]:
            with patch.dict(
                os.environ,
                {
                    "HIVE_WEBHOOK_URL": "https://test.com",
                    "HIVE_WEBHOOK_INCLUDE_OUTPUT": value,
                },
            ):
                config = WebhookConfig.from_env()
                assert config.include_output is True

        for value in ["false", "0", "no", ""]:
            with patch.dict(
                os.environ,
                {
                    "HIVE_WEBHOOK_URL": "https://test.com",
                    "HIVE_WEBHOOK_INCLUDE_OUTPUT": value,
                },
            ):
                config = WebhookConfig.from_env()
                assert config.include_output is False


class TestWebhookConfigFromDict:
    """Tests for WebhookConfig.from_dict()."""

    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        data = {"url": "https://minimal.webhook.com"}
        config = WebhookConfig.from_dict(data)

        assert config.url == "https://minimal.webhook.com"
        assert config.events == ["on_complete", "on_failure"]  # Default

    def test_from_dict_full(self):
        """Test creating config from full dict."""
        data = {
            "url": "https://full.webhook.com",
            "events": ["on_start", "on_complete"],
            "timeout_seconds": 20.0,
            "retry_count": 2,
            "retry_delay_seconds": 0.5,
            "headers": {"X-Custom": "value"},
            "include_output": True,
            "agent_name": "DictAgent",
        }
        config = WebhookConfig.from_dict(data)

        assert config.url == "https://full.webhook.com"
        assert config.events == ["on_start", "on_complete"]
        assert config.timeout_seconds == 20.0
        assert config.retry_count == 2
        assert config.headers == {"X-Custom": "value"}
        assert config.agent_name == "DictAgent"


class TestWebhookNotifier:
    """Tests for WebhookNotifier class."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh EventBus for testing."""
        return EventBus()

    @pytest.fixture
    def config(self):
        """Create a test webhook config."""
        return WebhookConfig(
            url="https://test.webhook.com",
            events=["on_start", "on_complete", "on_failure"],
            agent_name="TestAgent",
        )

    @pytest.fixture
    def notifier(self, event_bus, config):
        """Create a WebhookNotifier for testing."""
        return WebhookNotifier(event_bus, config)

    def test_notifier_init(self, notifier, event_bus, config):
        """Test notifier initialization."""
        assert notifier._event_bus is event_bus
        assert notifier._config is config
        assert notifier._subscription_id is None
        assert notifier._client is None
        assert notifier.is_running is False

    @pytest.mark.asyncio
    async def test_start_and_stop(self, notifier):
        """Test starting and stopping the notifier."""
        # Start
        await notifier.start()
        assert notifier.is_running is True
        assert notifier._subscription_id is not None
        assert notifier._client is not None

        # Stop
        await notifier.stop()
        assert notifier.is_running is False
        assert notifier._subscription_id is None
        assert notifier._client is None

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self, notifier):
        """Test that calling start twice is safe."""
        await notifier.start()
        subscription_id = notifier._subscription_id

        await notifier.start()  # Should not change anything
        assert notifier._subscription_id == subscription_id

        await notifier.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, notifier):
        """Test that calling stop twice is safe."""
        await notifier.start()
        await notifier.stop()
        await notifier.stop()  # Should not raise

        assert notifier.is_running is False

    def test_event_map(self, notifier):
        """Test event name to EventType mapping."""
        assert notifier.EVENT_MAP["on_start"] == EventType.EXECUTION_STARTED
        assert notifier.EVENT_MAP["on_complete"] == EventType.EXECUTION_COMPLETED
        assert notifier.EVENT_MAP["on_failure"] == EventType.EXECUTION_FAILED
        assert notifier.EVENT_MAP["on_pause"] == EventType.EXECUTION_PAUSED


class TestWebhookNotifierPayload:
    """Tests for payload building."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def config(self):
        return WebhookConfig(
            url="https://test.com",
            events=["on_start", "on_complete", "on_failure"],
            agent_name="PayloadTestAgent",
        )

    @pytest.fixture
    def notifier(self, event_bus, config):
        return WebhookNotifier(event_bus, config)

    def test_build_payload_started(self, notifier):
        """Test payload for started event."""
        event = AgentEvent(
            type=EventType.EXECUTION_STARTED,
            stream_id="test-stream",
            execution_id="exec-123",
        )

        payload = notifier._build_payload(event)

        assert payload["event"] == "execution_started"
        assert "PayloadTestAgent" in payload["message"]
        assert "started" in payload["message"]
        assert "🚀" in payload["message"]
        assert payload["stream_id"] == "test-stream"
        assert payload["execution_id"] == "exec-123"

    def test_build_payload_completed(self, notifier):
        """Test payload for completed event."""
        # First start the timer
        start_event = AgentEvent(
            type=EventType.EXECUTION_STARTED,
            stream_id="test-stream",
            execution_id="exec-456",
        )
        notifier._build_payload(start_event)

        # Then complete
        event = AgentEvent(
            type=EventType.EXECUTION_COMPLETED,
            stream_id="test-stream",
            execution_id="exec-456",
            data={"output": {"result": "success"}},
        )

        payload = notifier._build_payload(event)

        assert payload["event"] == "execution_completed"
        assert "completed successfully" in payload["message"]
        assert "✅" in payload["message"]
        assert "duration_seconds" in payload

    def test_build_payload_failed(self, notifier):
        """Test payload for failed event."""
        event = AgentEvent(
            type=EventType.EXECUTION_FAILED,
            stream_id="test-stream",
            execution_id="exec-789",
            data={"error": "Something went wrong"},
        )

        payload = notifier._build_payload(event)

        assert payload["event"] == "execution_failed"
        assert "failed" in payload["message"]
        assert "❌" in payload["message"]
        assert payload["error"] == "Something went wrong"

    def test_build_payload_with_include_output(self, event_bus):
        """Test that output is included when configured."""
        config = WebhookConfig(
            url="https://test.com",
            events=["on_complete"],
            include_output=True,
        )
        notifier = WebhookNotifier(event_bus, config)

        event = AgentEvent(
            type=EventType.EXECUTION_COMPLETED,
            stream_id="test",
            execution_id="exec",
            data={"output": {"key": "value"}},
        )

        payload = notifier._build_payload(event)
        assert payload["output"] == {"key": "value"}


class TestWebhookNotifierSendWebhook:
    """Tests for HTTP sending with retries."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def config(self):
        return WebhookConfig(
            url="https://test.com",
            retry_count=3,
            retry_delay_seconds=0.01,  # Fast for testing
        )

    @pytest.mark.asyncio
    async def test_send_webhook_success(self, event_bus, config):
        """Test successful webhook send."""
        notifier = WebhookNotifier(event_bus, config)
        await notifier.start()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(notifier._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await notifier._send_webhook({"test": "payload"})

            assert result is True
            mock_post.assert_called_once()

        await notifier.stop()

    @pytest.mark.asyncio
    async def test_send_webhook_retry_on_failure(self, event_bus, config):
        """Test that webhook retries on failure."""
        notifier = WebhookNotifier(event_bus, config)
        await notifier.start()

        # Mock responses: fail twice, then succeed
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.text = "Server Error"

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200

        with patch.object(notifier._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [
                mock_response_fail,
                mock_response_fail,
                mock_response_success,
            ]

            result = await notifier._send_webhook({"test": "payload"})

            assert result is True
            assert mock_post.call_count == 3

        await notifier.stop()

    @pytest.mark.asyncio
    async def test_send_webhook_all_retries_fail(self, event_bus, config):
        """Test that webhook returns False after all retries fail."""
        notifier = WebhookNotifier(event_bus, config)
        await notifier.start()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"

        with patch.object(notifier._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await notifier._send_webhook({"test": "payload"})

            assert result is False
            assert mock_post.call_count == 3  # retry_count

        await notifier.stop()


class TestCreateWebhookNotifier:
    """Tests for the create_webhook_notifier convenience function."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_create_with_url(self, event_bus):
        """Test creating notifier with explicit URL."""
        notifier = await create_webhook_notifier(
            event_bus=event_bus,
            url="https://explicit.webhook.com",
            agent_name="ExplicitAgent",
            events=["on_complete"],
        )

        assert notifier is not None
        assert notifier.is_running is True
        assert notifier._config.url == "https://explicit.webhook.com"
        assert notifier._config.agent_name == "ExplicitAgent"

        await notifier.stop()

    @pytest.mark.asyncio
    async def test_create_without_url_or_env(self, event_bus):
        """Test that None is returned when no URL available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HIVE_WEBHOOK_URL", None)
            notifier = await create_webhook_notifier(event_bus=event_bus)
            assert notifier is None

    @pytest.mark.asyncio
    async def test_create_from_env(self, event_bus):
        """Test creating notifier from environment variable."""
        with patch.dict(
            os.environ,
            {"HIVE_WEBHOOK_URL": "https://env.webhook.com"},
        ):
            notifier = await create_webhook_notifier(event_bus=event_bus)

            assert notifier is not None
            assert notifier.is_running is True
            assert notifier._config.url == "https://env.webhook.com"

            await notifier.stop()


class TestWebhookNotifierIntegration:
    """Integration tests for the full event flow."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_event_triggers_webhook(self, event_bus):
        """Test that publishing an event triggers a webhook call."""
        config = WebhookConfig(
            url="https://test.com",
            events=["on_complete"],
            retry_count=1,
        )
        notifier = WebhookNotifier(event_bus, config)

        # Track webhook calls
        webhook_calls = []

        async def mock_send(payload):
            webhook_calls.append(payload)
            return True

        await notifier.start()

        with patch.object(notifier, "_send_webhook", side_effect=mock_send):
            # Publish event
            await event_bus.emit_execution_completed(
                stream_id="test-stream",
                execution_id="exec-integration",
                output={"result": "done"},
            )

            # Give async handlers time to process
            await asyncio.sleep(0.1)

        await notifier.stop()

        # Verify webhook was called
        assert len(webhook_calls) == 1
        assert webhook_calls[0]["event"] == "execution_completed"
        assert webhook_calls[0]["execution_id"] == "exec-integration"
