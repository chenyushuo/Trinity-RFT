import json
from typing import Dict, List

from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.on_policy_distill_workflow import OnPolicyDistillWorkflow


@WORKFLOWS.register_module("on_policy_distill_message_input_workflow")
class OnPolicyDistillWorkflowMessageInput(OnPolicyDistillWorkflow):
    def format_messages(self) -> List[Dict]:
        return json.loads(self.raw_task[self.task.format_args.messages_key])  # type: ignore
