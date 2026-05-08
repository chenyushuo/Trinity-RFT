"""Utilities for processing multi-modal data (images/videos) for specific vision-language models.

Supported models:
- Qwen2.5-VL, Qwen3-VL series
- Kimi VL series
- GLM VL series

Provides functions to:
1. Parse prompts with media tags (<image>/<video>)
2. Validate multi-modal content in conversations
3. Preprocess media inputs for inference/training
4. Construct model-compatible message formats

Note:
    Only processors with class names containing both ("Qwen", "Kimi" OR "Glm") AND "Processor" are supported.
    Relies on `qwen_vl_utils.process_vision_info` for media extraction.
"""
import asyncio
import re
from typing import Any, Dict, List, Optional, Union

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormat,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict
from vllm.multimodal import MULTIMODAL_REGISTRY

from trinity.utils.log import get_logger


def is_qwen_like_processor(processor: Any) -> bool:
    return re.search(r"(Qwen|Kimi|Glm).*Processor", processor.__class__.__name__) is not None


def build_multi_modal_data(
    processor: Any,
    messages: List[Dict],
) -> Dict[str, Any]:
    """Extract and preprocess vision inputs from multi-modal messages for vLLM inference.

    Processes messages containing image/video placeholders using model-specific vision utilities.
    Returns structured media inputs compatible with vLLM's multi-modal API.

    Args:
        processor: Vision-language processor instance (must have class name containing
                   ("Qwen", "Kimi" OR "Glm") AND "Processor").
        messages: List of conversation messages in model-expected format. Each message's "content"
                  may be a string or list of content items (text/image/video dictionaries).

    Returns:
        Dictionary containing processed media inputs with keys:
        - "image": List of processed image objects (if images exist)
        - "video": List of processed video objects (if videos exist)
        Keys are omitted when no corresponding media is present.

    Raises:
        NotImplementedError: If processor class name doesn't match supported patterns.
        ImportError: If required `qwen_vl_utils` module is unavailable.

    Example:
        >>> messages = [{"role": "user", "content": [{"type": "image", "image": "img.jpg"}, {"type": "text", "text": "Describe this"}]}]
        >>> build_multi_modal_data(processor, messages)
        {"image": [processed_image]}
    """
    processor_class_name = processor.__class__.__name__
    if is_qwen_like_processor(processor):
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        multi_modal_data = {}
        if image_inputs:
            multi_modal_data["image"] = image_inputs
        if video_inputs:
            multi_modal_data["video"] = video_inputs

        return multi_modal_data
    raise NotImplementedError(
        f"Processor '{processor_class_name}' not supported. Only Qwen/Kimi/Glm VL processors are supported."
    )


def build_mm_input_for_training(
    processor: Any, prompt: str, multi_modal_data: Dict[str, List]
) -> Dict[str, Any]:
    """Tokenize prompt and integrate processed media inputs for model training.

    Combines text prompt with preprocessed image/video data into model-ready tensor inputs.
    Handles padding and tensor conversion for training workflows.

    Args:
        processor: Vision-language processor instance (must have class name containing
                   ("Qwen", "Kimi" OR "Glm") AND "Processor").
        prompt: Plain text prompt WITHOUT media tags (e.g., "Describe this image").
                Media placement is handled via `multi_modal_data`, not prompt tags.
        multi_modal_data: Dictionary from `build_multi_modal_data()` containing:
                          {"image": [...], "video": [...]} (keys optional)

    Returns:
        Dictionary of model inputs including:
        - input_ids: Tokenized prompt IDs
        - attention_mask: Attention mask tensor
        - pixel_values: Processed image tensors (if images provided)
        - pixel_values_videos: Processed video tensors (if videos provided)
        All tensors converted to PyTorch format (`return_tensors="pt"`).

    Raises:
        NotImplementedError: If processor class name doesn't match supported patterns.
        ValueError: If media counts mismatch prompt expectations (handled internally by processor).

    Note:
        Prompt should NOT contain <image>/<video> tags here. Media association is managed
        through the structured `multi_modal_data` dictionary.
    """
    processor_class_name = processor.__class__.__name__
    if is_qwen_like_processor(processor):
        inputs = processor(
            text=[prompt],
            images=multi_modal_data.get("image", None),
            videos=multi_modal_data.get("video", None),
            padding=True,
            return_tensors="pt",
        )
        return dict(inputs)
    raise NotImplementedError(
        f"Processor '{processor_class_name}' not supported. Only Qwen/Kimi/Glm VL processors are supported."
    )


def build_mm_message(
    prompt: str, images: List[Union[str, Any]], videos: List[Union[str, Any]]
) -> Dict[str, Any]:
    """Construct multi-modal message by injecting media references at tag positions in prompt.

    Parses prompt for <image>/<video> tags, replaces them with corresponding media references,
    and handles surplus media items. Extra media (beyond tag count) is prepended to content.

    Args:
        prompt: Text containing optional <image> and <video> tags as media placeholders.
                Example: "First <image> then <video> and finally <image>"
        images: List of image references (file paths, URLs, or PIL images) in order of appearance.
        videos: List of video references (file paths, URLs) in order of appearance.

    Returns:
        Message dictionary formatted for VL models:
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ...},  # Surplus media first
                {"type": "video", "video": ...},
                {"type": "text", "text": "First "},
                {"type": "image", "image": ...},  # Tag-replaced media
                ...
            ]
        }

    Raises:
        ValueError: If prompt contains more <image> tags than provided images,
                    or more <video> tags than provided videos.

    Behavior details:
        - Tags are case-sensitive and must be exact: "<image>", "<video>"
        - Empty text segments between tags are omitted
        - Surplus media (images/videos beyond tag count) appears at START of content list
        - Text segments preserve original prompt ordering around tags
    """
    content_list = []
    segments = re.split(r"(<image>|<video>)", prompt)
    img_idx, vid_idx = 0, 0
    for segment in segments:
        if segment == "<image>":
            if img_idx >= len(images):
                raise ValueError("More <image> tags in prompt than images provided.")
            content_list.append({"type": "image", "image": images[img_idx]})
            img_idx += 1
        elif segment == "<video>":
            if vid_idx >= len(videos):
                raise ValueError("More <video> tags in prompt than videos provided.")
            content_list.append({"type": "video", "video": videos[vid_idx]})
            vid_idx += 1
        elif len(segment) == 0:
            continue
        else:
            content_list.append({"type": "text", "text": segment})

    # Prepend surplus media items (not referenced by tags)
    surplus_content = []
    while img_idx < len(images):
        surplus_content.append({"type": "image", "image": images[img_idx]})
        img_idx += 1
    while vid_idx < len(videos):
        surplus_content.append({"type": "video", "video": videos[vid_idx]})
        vid_idx += 1

    content_list = surplus_content + content_list
    if len(content_list) == 1 and content_list[0]["type"] == "text":
        return {"role": "user", "content": content_list[0]["text"]}
    return {"role": "user", "content": content_list}


def has_multi_modal_content(messages: List[Dict]) -> bool:
    """Check if any message contains non-text (image/video) content.

    Inspects message content structure to detect multi-modal elements. Handles both:
    - String content (text-only, returns False)
    - List content (multi-modal candidates)

    Args:
        messages: List of conversation messages. Each message must contain a "content" field.
                  Content may be:
                  - str: Plain text message
                  - List[Dict]: Multi-modal content items (each with "type" key)

    Returns:
        True if any message contains at least one non-text content item (type != "text"),
        False otherwise.

    Example:
        >>> msg = [{"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "image", "image": "..."}]}]
        >>> has_multi_modal_content(msg)
        True
    """
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type", "text") != "text":
                    return True
    return False


class ClientMultiModalProcessor:
    """
    Client-side processor that mirrors vLLM server's multimodal handling.

    This class enables RL training endpoints to extract multimodal data
    with identical processing to the inference server, ensuring consistency.
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        *,
        media_io_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        allowed_local_media_path: str = "",
        allowed_media_domains: Optional[list[str]] = None,
        trust_request_chat_template: bool = False,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the client-side multimodal processor.

        Args:
            model_name: Model identifier (e.g., 'llava-1.5-7b')
            model_path: Path to the model (optional, will use default if not provided)
            media_io_kwargs: Media I/O configuration (mirrors --media-io-kwargs)
            allowed_local_media_path: Path to allowed local media directory
            allowed_media_domains: List of allowed media domains
            trust_request_chat_template: Whether to trust request-provided chat template
            mm_processor_kwargs: Additional processor kwargs for multimodal processing
        """
        self.logger = get_logger(__name__)

        self.model_name = model_name
        self.model_path = model_path
        self.media_io_kwargs = media_io_kwargs or {}
        self.mm_processor_kwargs = mm_processor_kwargs or {}

        # Initialize ModelConfig
        self.model_config = self._create_model_config(
            model_name,
            model_path,
            media_io_kwargs=self.media_io_kwargs,
        )

        # Initialize multimodal processor if the model supports it
        self.mm_processor = None
        if self.model_config.is_multimodal_model:
            try:
                self.mm_processor = MULTIMODAL_REGISTRY.create_processor(self.model_config)
                self.logger.info("Initialized multimodal processor for model: %s", model_name)
            except Exception as e:
                self.logger.warning(
                    "Failed to initialize multimodal processor: %s. "
                    "Some multimodal features may be unavailable.",
                    e,
                )

        # Store media connector configuration
        self._media_connector_config = {
            "media_io_kwargs": self.media_io_kwargs,
            "allowed_local_media_path": allowed_local_media_path,
            "allowed_media_domains": allowed_media_domains or [],
        }

        self.trust_request_chat_template = trust_request_chat_template

    def _create_model_config(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        media_io_kwargs: Optional[dict[str, dict[str, Any]]] = None,
    ) -> ModelConfig:
        """
        Create a ModelConfig instance matching the server configuration.

        This requires the model to be loadable from HuggingFace or local path.
        """
        # Create ModelConfig with multimodal support
        model_config = ModelConfig(
            model=model_name,
            tokenizer=model_name,
            tokenizer_mode="auto",
            trust_remote_code=True,
        )

        return model_config

    def process_messages(
        self,
        messages: list[dict[str, Any]],
        content_format: ChatTemplateContentFormat = "string",
        use_async: bool = False,
    ) -> tuple[
        list[ConversationMessage],
        Optional[MultiModalDataDict],
        Optional[MultiModalUUIDDict],
    ]:
        """
        Process chat messages and extract multimodal data.

        This replicates the server-side parse_chat_messages behavior.

        Args:
            messages: List of chat messages with potential multimodal content
            content_format: Chat template content format ("string" or "openai")
            use_async: Whether to use async processing for media fetching

        Returns:
            Tuple of (conversation, mm_data, mm_uuids) matching server output
        """
        if use_async:
            return asyncio.run(self.process_messages_async(messages, content_format))

        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages=messages,
            model_config=self.model_config,
            content_format=content_format,
            media_io_kwargs=self._media_connector_config["media_io_kwargs"],
            mm_processor_kwargs=self.mm_processor_kwargs,
        )

        return conversation, mm_data, mm_uuids

    async def process_messages_async(
        self,
        messages: list[dict[str, Any]],
        content_format: ChatTemplateContentFormat = "string",
    ) -> tuple[
        list[ConversationMessage],
        Optional[MultiModalDataDict],
        Optional[MultiModalUUIDDict],
    ]:
        """
        Async version of process_messages for concurrent media fetching.
        """
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages=messages,
            model_config=self.model_config,
            content_format=content_format,
            media_io_kwargs=self._media_connector_config["media_io_kwargs"],
            mm_processor_kwargs=self.mm_processor_kwargs,
        )

        return conversation, mm_data, mm_uuids

    def apply_mm_processor(
        self,
        mm_data: Optional[MultiModalDataDict],
        mm_uuids: Optional[MultiModalUUIDDict],
    ) -> dict[str, Any]:
        """
        Apply the multimodal processor to convert raw media to embeddings.

        This step mirrors the server's mm_processor.apply() call.

        Args:
            mm_data: Raw multimodal data dict
            mm_uuids: Multimodal UUIDs for tracking

        Returns:
            Processed multimodal inputs (embeddings, etc.)
        """
        if not self.mm_processor or not mm_data:
            return {}

        try:
            from vllm.multimodal.processing import MMProcessorInputs
            from vllm.renderers.inputs.preprocess import set_default_torch_num_threads

            mm_processor_inputs = MMProcessorInputs(
                prompt="",  # Prompt already tokenized
                mm_data_items=self.mm_processor.info.parse_mm_data(mm_data),
                mm_uuid_items={} if not mm_uuids else mm_uuids,
                hf_processor_mm_kwargs=self.mm_processor_kwargs.copy(),
                tokenization_kwargs={},
            )

            with set_default_torch_num_threads():
                mm_inputs = self.mm_processor.apply(
                    mm_processor_inputs,
                    timing_ctx=None,
                )

            return mm_inputs
        except Exception as e:
            self.logger.error(
                "Failed to apply multimodal processor: %s. " "Returning raw mm_data instead.", e
            )
            return {"mm_data": mm_data, "mm_uuids": mm_uuids}

    def get_mm_data_for_training(
        self,
        messages: list[dict[str, Any]],
        include_processed: bool = True,
    ) -> dict[str, Any]:
        """
        Extract multimodal data in a training-friendly format.

        This is a convenience method that combines all processing steps
        and returns a dict suitable for RL training.

        Args:
            messages: Chat messages with multimodal content
            include_processed: Whether to include processed embeddings

        Returns:
            Dict containing:
                - "conversation": processed conversation messages
                - "mm_data": raw multimodal data
                - "mm_uuids": multimodal identifiers
                - "mm_inputs" (optional): processed multimodal inputs
        """
        conversation, mm_data, mm_uuids = self.process_messages(messages)

        result = {
            "conversation": conversation,
            "mm_data": mm_data,
            "mm_uuids": mm_uuids,
        }

        if include_processed and mm_data:
            mm_inputs = self.apply_mm_processor(mm_data, mm_uuids)
            result["mm_inputs"] = mm_inputs

        return result
