import time
from typing import List, Optional

import qwen_vl_utils
import torch
import transformers

from trinity.common.experience import Experience
from trinity.common.models.mm_utils import ClientMultiModalProcessor
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task


def patch_qwen_vl_utils():
    if getattr(qwen_vl_utils, "_is_patched", False):
        return

    from qwen_vl_utils.vision_process import (
        IMAGE_MAX_TOKEN_NUM,
        IMAGE_MIN_TOKEN_NUM,
        SPATIAL_MERGE_SIZE,
        BytesIO,
        Dict,
        Image,
        Union,
        base64,
        copy,
        requests,
        smart_resize,
        to_rgb,
    )

    def new_fetch_image(
        ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14
    ) -> Image.Image:
        if "image" in ele:
            image = ele["image"]
        else:
            image = ele["image_url"]
            if isinstance(image, dict) and "url" in image:
                image = image["url"]

        image_obj = None
        patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
        if isinstance(image, Image.Image):
            image_obj = image
        elif image.startswith("http://") or image.startswith("https://"):
            with requests.get(image, stream=True) as response:
                response.raise_for_status()
                with BytesIO(response.content) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                with BytesIO(data) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        else:
            image_obj = Image.open(image)
        if image_obj is None:
            raise ValueError(
                f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
            )
        image = to_rgb(image_obj)

        ## resize
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=patch_factor,
            )
        else:
            width, height = image.size
            min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor**2)
            max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor**2)
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        image = image.resize((resized_width, resized_height))
        return image

    qwen_vl_utils.vision_process.fetch_image = new_fetch_image
    qwen_vl_utils._is_patched = True


@WORKFLOWS.register_module("copaw_rl_workflow")
class CoPawRLWorkflow(MultiTurnWorkflow):
    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    def run(self):
        from examples.copaw_rl.workflows.sandbox_utils import (
            get_or_create_sandbox,
            run_workflow,
        )

        patch_qwen_vl_utils()

        start_time = time.time()
        sandbox_id = self.task.workflow_args.get("sandbox_id", None)
        token = self.task.workflow_args["token"]
        domain = self.task.workflow_args["domain"]
        template = self.task.workflow_args["template"]

        sandbox, created = get_or_create_sandbox(sandbox_id, token, domain, template, self.logger)
        sandbox_id = sandbox.sandbox_id

        oss_config = self.task.workflow_args["oss"]
        otel_config = self.task.workflow_args["otel"]
        dashscope_api_key = self.task.workflow_args["dashscope_api_key"]
        task_id = self.task.raw_task["task_id"]
        api_server_url = f"{self.model.api_address}/v1"
        model_path = self.model.model_name
        try:
            dataset = run_workflow(
                sandbox,
                task_id,
                oss_config,
                otel_config,
                dashscope_api_key,
                api_server_url,
                model_path,
                self.logger,
            )
        except Exception as e:
            self.logger.error(f"Error running workflow (ID: {sandbox_id}): {e}")
            raise e
        finally:
            sandbox.kill()

        exps = []
        processor = None
        vllm_processor = ClientMultiModalProcessor(model_name=model_path)
        for data in dataset:
            prompt_token_ids = torch.tensor(data["prompt_token_ids"])
            response_token_ids = torch.tensor(data["token_ids"])
            token_ids = torch.cat([prompt_token_ids, response_token_ids])
            logprobs = torch.tensor(data["logprobs"])
            prompt_length = len(prompt_token_ids)
            action_mask = torch.tensor(data["response_mask"], dtype=torch.int)
            reward = float(data.get("reward", 0.0))
            metrics = {
                "reward": reward,
            }

            messages = data["messages"]
            _, mm_data, _ = vllm_processor.process_messages(messages)
            if mm_data is not None:
                if processor is None:
                    processor = transformers.AutoProcessor.from_pretrained(model_path)
                multi_modal_inputs = {}
                # outputs_kwargs = processor._merge_kwargs(
                #     Qwen3VLProcessorKwargs,
                #     tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
                #     return_tensors="pt",
                # )
                if images := mm_data.get("image", None):
                    images = [img.media for img in images]
                    image_inputs = processor.image_processor(images=images, return_tensors="pt")
                    multi_modal_inputs.update(image_inputs)
                if videos := mm_data.get("video", None):
                    videos = [vid.media for vid in videos]
                    video_inputs = processor.video_processor(videos=videos, return_tensors="pt")
                    multi_modal_inputs.update(video_inputs)
            else:
                multi_modal_inputs = None

            exp = Experience(
                tokens=token_ids,
                logprobs=logprobs,
                prompt_length=prompt_length,
                action_mask=action_mask,
                reward=reward,
                metrics=metrics,
                multi_modal_inputs=multi_modal_inputs,
            )
            exps.append(exp)
        del processor, vllm_processor

        self.logger.info(
            f"Workflow finished in {time.time() - start_time:.2f} seconds. Sandbox {'created' if created else 'connected'} "
            f"(ID: {sandbox_id}). Reward = {reward}. Collected {len(exps)} experiences."
        )

        return exps
