"""
Webhook Notifier - Sends HTTP POST requests for agent lifecycle events.

Allows developers to receive notifications when agents start, complete, fail, or pause.
Useful for integrating with Slack, Discord, or custom monitoring systems.

Example:
    # Via environment variable
    export HIVE_WEBHOOK_URL="https://hooks.slack.com/services/xxx/yyy/zzz"

    # Via agent.json
    {
        "webhook": {
            "url": "https://discord.com/api/webhooks/xxx/yyy",
            "events": ["on_complete", "on_failure"],
            "agent_name": "Research Agent"
        }
    }
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from framework.runtime.event_bus import AgentEvent, EventBus, EventType

logger = logging.getLogger(__name__)

# Check for httpx availability (should be present via mcp/litellm deps)
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - webhook notifications disabled")


@dataclass
class WebhookConfig:
    """
    Configuration for webhook notifications.

    Attributes:
        url: Webhook URL to POST to
        events: List of events to notify on. Options:
            - "on_start": When execution begins
            - "on_complete": When execution completes successfully
            - "on_failure": When execution fails
            - "on_pause": When execution pauses for human input
        timeout_seconds: HTTP request timeout
        retry_count: Number of retry attempts on failure
        retry_delay_seconds: Delay between retries
        headers: Additional HTTP headers (e.g., for auth)
        include_output: Whether to include full agent output in payload
        agent_name: Human-readable name for notifications
    """

    url: str
    events: list[str] = field(
        default_factory=lambda: ["on_start", "on_complete", "on_failure", "on_pause"]
    )
    timeout_seconds: float = 10.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    headers: dict[str, str] = field(default_factory=dict)
    include_output: bool = False
    agent_name: str | None = None

    @classmethod
    def from_env(cls, agent_name: str | None = None) -> "WebhookConfig | None":
        """
        Create config from environment variables.

        Environment Variables:
            HIVE_WEBHOOK_URL: Required webhook URL
            HIVE_WEBHOOK_EVENTS: Comma-separated events (default: on_complete,on_failure)
            HIVE_WEBHOOK_TIMEOUT: Timeout in seconds (default: 10)
            HIVE_WEBHOOK_INCLUDE_OUTPUT: Include output (default: false)

        Returns:
            WebhookConfig if URL is set, None otherwise
        """
        url = os.environ.get("HIVE_WEBHOOK_URL")
        if not url:
            return None

        events_str = os.environ.get("HIVE_WEBHOOK_EVENTS", "on_complete,on_failure")
        events = [e.strip() for e in events_str.split(",") if e.strip()]

        timeout_str = os.environ.get("HIVE_WEBHOOK_TIMEOUT", "10")
        try:
            timeout = float(timeout_str)
        except ValueError:
            timeout = 10.0

        include_output_str = os.environ.get("HIVE_WEBHOOK_INCLUDE_OUTPUT", "false")
        include_output = include_output_str.lower() in ("true", "1", "yes")

        return cls(
            url=url,
            events=events,
            timeout_seconds=timeout,
            include_output=include_output,
            agent_name=agent_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WebhookConfig":
        """Create config from dictionary (e.g., from agent.json)."""
        return cls(
            url=data["url"],
            events=data.get("events", ["on_complete", "on_failure"]),
            timeout_seconds=data.get("timeout_seconds", 10.0),
            retry_count=data.get("retry_count", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 1.0),
            headers=data.get("headers", {}),
            include_output=data.get("include_output", False),
            agent_name=data.get("agent_name"),
        )


class WebhookNotifier:
    """
    Subscribes to EventBus and sends webhook notifications on agent lifecycle events.

    Example:
        config = WebhookConfig(
            url="https://hooks.slack.com/services/...",
            events=["on_complete", "on_failure"],
            agent_name="Research Agent",
        )

        notifier = WebhookNotifier(event_bus, config)
        await notifier.start()

        # ... agent runs, webhooks fire automatically ...

        await notifier.stop()
    """

    # Map user-friendly event names to EventType enum
    EVENT_MAP = {
        "on_start": EventType.EXECUTION_STARTED,
        "on_complete": EventType.EXECUTION_COMPLETED,
        "on_failure": EventType.EXECUTION_FAILED,
        "on_pause": EventType.EXECUTION_PAUSED,
    }

    def __init__(
        self,
        event_bus: EventBus,
        config: WebhookConfig,
    ):
        """
        Initialize the webhook notifier.

        Args:
            event_bus: EventBus to subscribe to
            config: Webhook configuration
        """
        self._event_bus = event_bus
        self._config = config
        self._subscription_id: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._start_times: dict[str, float] = {}  # Track execution start times
        self._running = False

    async def start(self) -> None:
        """Start listening for events and sending webhooks."""
        if not HTTPX_AVAILABLE:
            logger.warning("Cannot start WebhookNotifier: httpx not available")
            return

        if self._running:
            return

        # Create HTTP client
        self._client = httpx.AsyncClient(
            timeout=self._config.timeout_seconds,
            headers=self._config.headers,
        )

        # Determine which events to subscribe to
        event_types = [self.EVENT_MAP[e] for e in self._config.events if e in self.EVENT_MAP]

        if not event_types:
            logger.warning("No valid event types configured for webhook")
            return

        # Subscribe to events
        self._subscription_id = self._event_bus.subscribe(
            event_types=event_types,
            handler=self._handle_event,
        )

        self._running = True
        logger.info(f"WebhookNotifier started: {self._config.url} (events: {self._config.events})")

    async def stop(self) -> None:
        """Stop listening and cleanup resources."""
        if not self._running:
            return

        if self._subscription_id:
            self._event_bus.unsubscribe(self._subscription_id)
            self._subscription_id = None

        if self._client:
            await self._client.aclose()
            self._client = None

        self._running = False
        logger.info("WebhookNotifier stopped")

    @property
    def is_running(self) -> bool:
        """Check if notifier is running."""
        return self._running

    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle incoming event and send webhook."""
        try:
            payload = self._build_payload(event)
            success = await self._send_webhook(payload)

            if success:
                logger.debug(f"Webhook sent: {payload.get('message', 'unknown')}")
            else:
                logger.warning(f"Webhook failed for event: {event.type.value}")

        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")

    def _build_payload(self, event: AgentEvent) -> dict[str, Any]:
        """
        Build webhook payload from event.

        The payload is designed to be generic but includes a human-readable
        message that works well for Slack/Discord notifications.
        """
        exec_id = event.execution_id or "unknown"

        # Track timing for duration calculation
        if event.type == EventType.EXECUTION_STARTED:
            self._start_times[exec_id] = time.time()
            duration = None
        else:
            start_time = self._start_times.pop(exec_id, None)
            duration = time.time() - start_time if start_time else None

        # Build human-readable message
        agent_name = self._config.agent_name or event.stream_id
        emoji, status = self._get_status_info(event.type)

        if duration is not None:
            duration_str = f" in {duration:.1f}s"
        else:
            duration_str = ""

        message = f"{emoji} {agent_name} {status}{duration_str}"

        # Build payload
        payload = {
            "event": event.type.value,
            "message": message,
            "agent_name": agent_name,
            "stream_id": event.stream_id,
            "execution_id": exec_id,
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": event.correlation_id,
        }

        # Add duration if available
        if duration is not None:
            payload["duration_seconds"] = round(duration, 2)

        # Add error details for failures
        if event.type == EventType.EXECUTION_FAILED:
            payload["error"] = event.data.get("error", "Unknown error")

        # Optionally include full output
        if self._config.include_output and event.type == EventType.EXECUTION_COMPLETED:
            payload["output"] = event.data.get("output", {})

        return payload

    def _get_status_info(self, event_type: EventType) -> tuple[str, str]:
        """Get emoji and status text for event type."""
        mapping = {
            EventType.EXECUTION_STARTED: ("🚀", "started"),
            EventType.EXECUTION_COMPLETED: ("✅", "completed successfully"),
            EventType.EXECUTION_FAILED: ("❌", "failed"),
            EventType.EXECUTION_PAUSED: ("⏸️", "paused (awaiting input)"),
        }
        return mapping.get(event_type, ("📢", event_type.value))

    async def _send_webhook(self, payload: dict[str, Any]) -> bool:
        """
        Send webhook with retries.

        Returns:
            True if successful, False otherwise
        """
        if not self._client:
            return False

        for attempt in range(self._config.retry_count):
            try:
                response = await self._client.post(
                    self._config.url,
                    json=payload,
                )

                if response.status_code < 300:
                    return True
                else:
                    logger.warning(
                        f"Webhook returned {response.status_code}: {response.text[:200]}"
                    )

            except httpx.TimeoutException:
                logger.warning(
                    f"Webhook timeout (attempt {attempt + 1}/{self._config.retry_count})"
                )
            except httpx.RequestError as e:
                logger.warning(f"Webhook request error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected webhook error: {e}")
                return False  # Don't retry on unexpected errors

            # Wait before retry (except on last attempt)
            if attempt < self._config.retry_count - 1:
                await asyncio.sleep(self._config.retry_delay_seconds)

        return False


# Convenience function for quick setup
async def create_webhook_notifier(
    event_bus: EventBus,
    url: str | None = None,
    agent_name: str | None = None,
    **kwargs,
) -> WebhookNotifier | None:
    """
    Create and start a webhook notifier.

    If url is not provided, attempts to load from HIVE_WEBHOOK_URL environment variable.

    Args:
        event_bus: EventBus to subscribe to
        url: Webhook URL (optional, can use env var)
        agent_name: Human-readable agent name
        **kwargs: Additional WebhookConfig options

    Returns:
        Started WebhookNotifier, or None if no URL configured
    """
    if url:
        config = WebhookConfig(url=url, agent_name=agent_name, **kwargs)
    else:
        config = WebhookConfig.from_env(agent_name=agent_name)

    if config is None:
        return None

    notifier = WebhookNotifier(event_bus, config)
    await notifier.start()
    return notifier
