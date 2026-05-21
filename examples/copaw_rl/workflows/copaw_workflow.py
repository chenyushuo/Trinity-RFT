import time
from typing import List, Optional

import torch
import transformers

from trinity.common.experience import Experience
from trinity.common.models.mm_utils import ClientMultiModalProcessor
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task


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
