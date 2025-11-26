"""Unit tests for TransformEngine."""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from loom.core.models import Record, TransformConfig
from loom.core.types import RecordStatus
from loom.core.exceptions import ConfigurationError, TransformError
from loom.engines.transform import TransformEngine
from loom.resilience import CircuitBreakerOpenError


class MockLLMResponse:
    """Mock LLM response."""

    def __init__(self, content: str):
        self.content = content


class TestTransformEngine:
    """Test TransformEngine functionality."""

    def test_init_with_valid_prompt(self):
        """Test engine initialization with valid prompt template."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
            model="gpt-4o-mini",
        )
        engine = TransformEngine(config)

        assert engine.config == config
        assert engine.prompt_template is not None
        assert engine.llm_client is None
        assert engine.circuit_breaker.name == "transform_llm"

    def test_init_with_nonexistent_prompt(self):
        """Test engine initialization fails with nonexistent prompt file."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/nonexistent.txt",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            TransformEngine(config)

        assert "Prompt template not found" in str(exc_info.value)

    def test_prompt_template_loading(self):
        """Test prompt template is loaded correctly."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        # Template should contain placeholder
        template_str = engine.prompt_template.template
        assert "$text" in template_str
        assert "sentiment" in template_str.lower()

    def test_build_prompt_success(self):
        """Test building prompt from template and record data."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        record = Record(
            id="test_1",
            data={"text": "This product is amazing!"},
        )

        prompt = engine._build_prompt(record)

        assert "This product is amazing!" in prompt
        assert "$text" not in prompt  # Variable should be substituted

    def test_build_prompt_missing_variable(self):
        """Test building prompt fails with missing template variable."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        record = Record(
            id="test_1",
            data={"wrong_key": "Some text"},  # Missing 'text' key
        )

        with pytest.raises(TransformError) as exc_info:
            engine._build_prompt(record)

        assert "Missing template variable" in str(exc_info.value)
        assert "text" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_llm_client_initialization(self, mocker):
        """Test LLM client is initialized on first call."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        # Mock LLMManager (imported inside the method)
        mock_client = AsyncMock()

        # Create a mock that returns mock_client when awaited
        async def mock_get_client_func(**kwargs):
            return mock_client

        mock_llm_manager = MagicMock()
        mock_llm_manager.get_client = AsyncMock(side_effect=mock_get_client_func)
        mocker.patch("arbiter_ai.core.llm_client.LLMManager", mock_llm_manager)

        # First call should initialize client
        client1 = await engine._get_llm_client()
        assert client1 == mock_client
        assert engine.llm_client == mock_client
        mock_llm_manager.get_client.assert_called_once_with(model="gpt-4o-mini")

        # Second call should reuse client
        client2 = await engine._get_llm_client()
        assert client2 == mock_client
        assert mock_llm_manager.get_client.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_transform_record_success(self, mocker):
        """Test successful single record transformation."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
            model="gpt-4o-mini",
            temperature=0.7,
            timeout=60.0,
        )
        engine = TransformEngine(config)

        # Mock LLM client
        mock_client = AsyncMock()
        mock_response = MockLLMResponse(content="positive")

        # Make complete an async function that returns the response
        async def mock_complete_func(**kwargs):
            return mock_response
        mock_client.complete = AsyncMock(side_effect=mock_complete_func)

        async def mock_get_client_func(**kwargs):
            return mock_client

        mock_llm_manager = MagicMock()
        mock_llm_manager.get_client = AsyncMock(side_effect=mock_get_client_func)
        mocker.patch("arbiter_ai.core.llm_client.LLMManager", mock_llm_manager)

        record = Record(
            id="test_1",
            data={"text": "This product is amazing!"},
            status=RecordStatus.EXTRACTED,
        )

        result = await engine.transform_record(record)

        assert result.id == "test_1"
        assert result.transformed_data == "positive"
        assert result.status == RecordStatus.TRANSFORMED
        assert result.error is None

        # Verify LLM was called correctly
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_tokens"] is None
        assert len(call_args.kwargs["messages"]) == 1
        assert "This product is amazing!" in call_args.kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_transform_record_with_max_tokens(self, mocker):
        """Test transformation with max_tokens specified."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
            max_tokens=100,
        )
        engine = TransformEngine(config)

        mock_client = AsyncMock()
        mock_response = MockLLMResponse(content="positive")

        async def mock_complete_func(**kwargs):
            return mock_response
        mock_client.complete = AsyncMock(side_effect=mock_complete_func)

        async def mock_get_client_func(**kwargs):
            return mock_client

        mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
        mock_manager.get_client = AsyncMock(side_effect=mock_get_client_func)

        record = Record(id="test_1", data={"text": "Great!"})

        await engine.transform_record(record)

        call_args = mock_client.complete.call_args
        assert call_args.kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_transform_record_timeout(self, mocker):
        """Test transformation fails on timeout."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
            timeout=0.1,  # Very short timeout
        )
        engine = TransformEngine(config)

        # Mock LLM client with slow response
        mock_client = AsyncMock()

        async def slow_complete(*args, **kwargs):
            await asyncio.sleep(1.0)  # Longer than timeout
            return MockLLMResponse(content="positive")

        mock_client.complete = slow_complete

        async def mock_get_client_func(**kwargs):
            return mock_client

        mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
        mock_manager.get_client = AsyncMock(side_effect=mock_get_client_func)

        record = Record(id="test_1", data={"text": "Test"})

        with pytest.raises(TransformError) as exc_info:
            await engine.transform_record(record)

        assert "timeout" in str(exc_info.value).lower()
        assert record.status == RecordStatus.ERROR
        assert record.error is not None

    @pytest.mark.asyncio
    async def test_transform_record_circuit_breaker_open(self, mocker):
        """Test transformation fails when circuit breaker is open."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        # Mock circuit breaker to raise open error
        mock_breaker = mocker.patch.object(engine.circuit_breaker, "call")
        open_until = time.time() + 60.0  # Circuit open for 60 seconds
        mock_breaker.side_effect = CircuitBreakerOpenError(
            "transform_llm", open_until
        )

        mock_client = AsyncMock()

        async def mock_get_client_func(**kwargs):
            return mock_client

        mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
        mock_manager.get_client = AsyncMock(side_effect=mock_get_client_func)

        record = Record(id="test_1", data={"text": "Test"})

        with pytest.raises(TransformError) as exc_info:
            await engine.transform_record(record)

        assert "circuit breaker open" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_transform_record_llm_error(self, mocker):
        """Test transformation handles LLM errors gracefully."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        # Mock LLM client to raise error
        mock_client = AsyncMock()
        mock_client.complete.side_effect = Exception("LLM API error")

        mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
        mock_manager.get_client.return_value = mock_client

        record = Record(id="test_1", data={"text": "Test"})

        with pytest.raises(TransformError) as exc_info:
            await engine.transform_record(record)

        assert "Transform failed" in str(exc_info.value)
        assert record.status == RecordStatus.ERROR
        assert record.error is not None

    @pytest.mark.asyncio
    async def test_transform_batch_all_success(self, mocker):
        """Test batch transformation with all records succeeding."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        # Mock LLM client
        mock_client = AsyncMock()
        mock_responses = [
            MockLLMResponse(content="positive"),
            MockLLMResponse(content="negative"),
            MockLLMResponse(content="neutral"),
        ]

        call_index = [0]
        async def mock_complete_func(**kwargs):
            response = mock_responses[call_index[0]]
            call_index[0] += 1
            return response
        mock_client.complete = AsyncMock(side_effect=mock_complete_func)

        async def mock_get_client_func(**kwargs):
            return mock_client

        mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
        mock_manager.get_client = AsyncMock(side_effect=mock_get_client_func)

        records = [
            Record(id="rec_1", data={"text": "Great!"}),
            Record(id="rec_2", data={"text": "Terrible!"}),
            Record(id="rec_3", data={"text": "It's okay."}),
        ]

        results = await engine.transform_batch(records)

        assert len(results) == 3
        assert results[0].transformed_data == "positive"
        assert results[1].transformed_data == "negative"
        assert results[2].transformed_data == "neutral"
        assert all(r.status == RecordStatus.TRANSFORMED for r in results)
        assert mock_client.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_transform_batch_partial_failure(self, mocker):
        """Test batch transformation with some records failing."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
            timeout=0.5,
        )
        engine = TransformEngine(config)

        # Mock LLM client with mixed success/failure
        mock_client = AsyncMock()
        call_count = [0]

        async def mixed_responses(*args, **kwargs):
            call_count[0] += 1
            # First call succeeds
            if call_count[0] == 1:
                return MockLLMResponse(content="positive")
            # Second call times out
            elif call_count[0] == 2:
                await asyncio.sleep(1.0)
                return MockLLMResponse(content="negative")
            # Third call succeeds
            else:
                return MockLLMResponse(content="neutral")

        mock_client.complete = mixed_responses

        async def mock_get_client_func(**kwargs):
            return mock_client

        mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
        mock_manager.get_client = AsyncMock(side_effect=mock_get_client_func)

        records = [
            Record(id="rec_1", data={"text": "Great!"}),
            Record(id="rec_2", data={"text": "Slow response"}),
            Record(id="rec_3", data={"text": "Okay."}),
        ]

        results = await engine.transform_batch(records)

        assert len(results) == 3

        # First and third should succeed
        assert results[0].status == RecordStatus.TRANSFORMED
        assert results[2].status == RecordStatus.TRANSFORMED

        # Second should have error
        assert results[1].status == RecordStatus.ERROR
        assert results[1].error is not None

    @pytest.mark.asyncio
    async def test_transform_batch_empty(self, mocker):
        """Test batch transformation with empty list."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        results = await engine.transform_batch([])

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        """Test close when no LLM client was created."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        # Should not raise error
        await engine.close()

    @pytest.mark.asyncio
    async def test_close_with_client(self, mocker):
        """Test close properly cleans up LLM client."""
        config = TransformConfig(
            prompt="tests/fixtures/prompts/classify_sentiment.txt",
        )
        engine = TransformEngine(config)

        # Initialize client
        mock_client = AsyncMock()

        async def mock_get_client_func(**kwargs):
            return mock_client

        async def mock_close_func():
            pass

        mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
        mock_manager.get_client = AsyncMock(side_effect=mock_get_client_func)
        mock_manager.close = AsyncMock(side_effect=mock_close_func)

        await engine._get_llm_client()

        # Close should call LLMManager.close()
        await engine.close()
        mock_manager.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_transform_different_models(self, mocker):
        """Test transformation with different model configurations."""
        models = ["gpt-4o-mini", "claude-3-5-sonnet", "gemini-1.5-pro"]

        for model in models:
            config = TransformConfig(
                prompt="tests/fixtures/prompts/classify_sentiment.txt",
                model=model,
            )
            engine = TransformEngine(config)

            mock_client = AsyncMock()
            mock_response = MockLLMResponse(content="positive")

            async def mock_complete_func(**kwargs):
                return mock_response
            mock_client.complete = AsyncMock(side_effect=mock_complete_func)

            async def mock_get_client_func(**kwargs):
                return mock_client

            mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
            mock_manager.get_client = AsyncMock(side_effect=mock_get_client_func)

            record = Record(id="test_1", data={"text": "Great!"})
            await engine.transform_record(record)

            # Verify correct model was requested
            mock_manager.get_client.assert_called_with(model=model)

    @pytest.mark.asyncio
    async def test_transform_different_temperatures(self, mocker):
        """Test transformation with different temperature values."""
        temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]

        for temp in temperatures:
            config = TransformConfig(
                prompt="tests/fixtures/prompts/classify_sentiment.txt",
                temperature=temp,
            )
            engine = TransformEngine(config)

            mock_client = AsyncMock()
            mock_response = MockLLMResponse(content="positive")

            async def mock_complete_func(**kwargs):
                return mock_response
            mock_client.complete = AsyncMock(side_effect=mock_complete_func)

            async def mock_get_client_func(**kwargs):
                return mock_client

            mock_manager = mocker.patch("arbiter_ai.core.llm_client.LLMManager")
            mock_manager.get_client = AsyncMock(side_effect=mock_get_client_func)

            record = Record(id="test_1", data={"text": "Great!"})
            await engine.transform_record(record)

            # Verify correct temperature was used
            call_args = mock_client.complete.call_args
            assert call_args.kwargs["temperature"] == temp
