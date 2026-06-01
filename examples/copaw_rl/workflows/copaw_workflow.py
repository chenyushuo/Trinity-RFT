import json
import time
from typing import List, Optional

import oss2
import torch

from trinity.common.experience import Experience
from trinity.common.models.mm_utils import vLLMMultiModalRender
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
        self.sandbox_token, self.sandbox_template = self.get_sandbox_token_and_template()

    def get_sandbox_token_and_template(self):
        try:
            oss_config = self.task.workflow_args["oss"]
            auth = oss2.Auth(oss_config["access_key_id"], oss_config["access_key_secret"])
            bucket = oss2.Bucket(
                auth,
                oss_config["endpoint"],
                oss_config["bucket_name"],
                region=oss_config.get("region"),
            )
            content = bucket.get_object("env/sandbox_meta.json").read().decode("utf-8")
            meta = json.loads(content)

            token = meta.get("E2B_API_KEY")
            template = meta.get("E2B_TEMPLATE")
            return token, template
        except Exception as e:
            self.logger.warning(
                f"Failed to read env/sandbox_meta.json from OSS, fallback to None: {e}"
            )
            return None, None

    def run(self):
        from examples.copaw_rl.workflows.sandbox_utils import (
            get_or_create_sandbox,
            run_workflow,
        )

        start_time = time.perf_counter()

        sandbox_id = self.task.workflow_args.get("sandbox_id", None)
        token = self.task.workflow_args.get("token", self.sandbox_token) or self.sandbox_token
        domain = self.task.workflow_args["domain"]
        template = (
            self.task.workflow_args.get("template", self.sandbox_template) or self.sandbox_template
        )

        sandbox, created = get_or_create_sandbox(sandbox_id, token, domain, template, self.logger)
        sandbox_id = sandbox.sandbox_id

        oss_config = self.task.workflow_args["oss"]
        otel_config = self.task.workflow_args["otel"]
        dashscope_api_key = self.task.workflow_args["dashscope_api_key"]
        task_id = self.task.raw_task["task_id"]
        api_server_url = f"{self.model.api_address}/v1"
        model_path = self.model.model_name
        try:
            output = run_workflow(
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
        end_time = time.perf_counter()
        duration = end_time - start_time

        exps = []
        render = vLLMMultiModalRender(model_path=model_path)
        dataset = output["dataset"]
        total_steps = output["total_steps"]
        launch_duration_seconds = output["launch_duration_seconds"]
        prepare_duration = output["prepare_duration"]
        call_agent_duration = output["call_agent_duration"]
        extract_duration = output["extract_duration"]
        llm_judge_duration = output["llm_judge_duration"]
        inner_metrics = output["metrics"]
        for step, data in enumerate(dataset):
            prompt_token_ids = torch.tensor(data["prompt_token_ids"])
            response_token_ids = torch.tensor(data["token_ids"])
            token_ids = torch.cat([prompt_token_ids, response_token_ids])
            logprobs = torch.tensor(data["logprobs"])
            prompt_length = len(prompt_token_ids)
            action_mask = torch.tensor(data["response_mask"], dtype=torch.int)
            reward = float(data.get("reward", 0.0))
            metrics = {
                "reward": reward,
                "total_steps": total_steps,
                "sandbox_duration": duration,
                "launch_duration_seconds": launch_duration_seconds,
                "prepare_duration": prepare_duration,
                "call_agent_duration": call_agent_duration,
                "extract_duration": extract_duration,
                "llm_judge_duration": llm_judge_duration,
            }
            metrics.update(inner_metrics)
            multi_modal_inputs = render.build_mm_input_for_training(
                messages=data["messages"],
                input_ids=token_ids.tolist(),
            )

            exp = Experience(
                tokens=token_ids,
                logprobs=logprobs,
                prompt_length=prompt_length,
                action_mask=action_mask,
                reward=reward,
                metrics=metrics,
                multi_modal_inputs=multi_modal_inputs,
            )
            exp.eid.step = step
            exps.append(exp)
        del render

        self.logger.info(
            f"Workflow finished in {time.perf_counter() - start_time:.2f} seconds. "
            f"Sandbox duration = {duration:.2f} seconds. "
            f"Launch duration = {launch_duration_seconds:.2f} seconds. "
            f"Prepare duration = {prepare_duration:.2f} seconds. "
            f"Call agent duration = {call_agent_duration:.2f} seconds. "
            f"Extract duration = {extract_duration:.2f} seconds. "
            f"LLM judge duration = {llm_judge_duration:.2f} seconds. "
            f"Reward = {reward}. Total steps = {total_steps}. Collected {len(exps)} experiences."
        )

        return exps
