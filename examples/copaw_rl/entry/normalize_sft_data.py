import argparse
import io
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import oss2
from tqdm import tqdm


def download_from_oss(oss_path):
    access_key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    access_key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = os.environ.get("OSS_ENDPOINT")
    region = os.environ.get("OSS_REGION")

    auth = oss2.Auth(access_key_id, access_key_secret)
    # Extract bucket name and object key from OSS path
    _, _, bucket_name, *object_key_parts = oss_path.split("/")
    object_key = "/".join(object_key_parts)

    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

    # Download the object
    result = bucket.get_object(object_key)
    data = result.read().decode("utf-8")
    f = io.StringIO(data)
    return [json.loads(line) for line in f.readlines()]


def fix_response_message(message):
    reasoning_content = ""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    if isinstance(message["content"], str):
        return message

    for block in message["content"]:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            reasoning_content = block.get("thinking", "") or reasoning_content
        elif btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_id = block.get("id", "")
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        # 这里先保持为 dict，后面统一做 str->dict 修复
                        "arguments": tool_input,
                    },
                },
            )

    response_msg: dict = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else "",
    }
    if reasoning_content:
        response_msg["reasoning_content"] = reasoning_content
    if tool_calls:
        response_msg["tool_calls"] = tool_calls

    return response_msg


# 必须定义在顶层，multiprocessing 才能 pickle
def process_batch(batch, model_path):
    """每个子进程独立加载 tokenizer，处理一批数据"""
    import transformers

    from trinity.buffer.schema.formatter import SFTFormatter
    from trinity.common.config import FormatConfig
    from trinity.common.constants import PromptType

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    formatter_config = FormatConfig(
        prompt_type=PromptType.MESSAGES,
        prompt_key="prompt",
        response_key="response",
        system_prompt_key=None,
        system_prompt=None,
        messages_key="messages",
        tools_key="tools",
        image_key="image_placeholder",
        video_key=None,
        reply_prefix=None,
        workflow_key="",
        reward_fn_key="",
        chosen_key="chosen",
        rejected_key="rejected",
        enable_concatenated_multi_turn=True,
        chat_template=None,
        enable_thinking=True,
    )
    sft_formatter = SFTFormatter(model_path, formatter_config)
    # lengths = []
    subset = []
    lengths = []
    for data in batch:
        if isinstance(data["messages"], list):
            messages = data["messages"]
        else:
            messages = json.loads(data["messages"])

        for msg in messages:
            for tool_call in msg.get("tool_calls", []):
                if isinstance(tool_call["function"]["arguments"], str):
                    # print("tool_call arguments is not a dict")
                    tool_call["function"]["arguments"] = json.loads(
                        tool_call["function"]["arguments"]
                    )
        messages[-1] = fix_response_message(messages[-1])

        data["messages"] = json.dumps(messages, ensure_ascii=False)
        # tools = json.loads(data['tools']) if 'tools' in data else None
        if "tools" in data:
            if isinstance(data["tools"], str):
                tools = json.loads(data["tools"])
            else:
                tools = data["tools"]
                data["tools"] = json.dumps(tools, ensure_ascii=False)
        else:
            data["tools"] = tools = ""

        try:
            # inputs = tokenizer.apply_chat_template(
            #     messages, tools=tools, add_generation_prompt=False
            # )
            exp = sft_formatter.format(data)
        except Exception as e:
            # print(e)
            # print(data)
            # raise e
            print(f"Error processing data: {data}, error: {e}")
            continue

        # lengths.append(len(inputs['input_ids']))
        default_tag = {
            "model": "unknown",
            "version": "unknown",
            "category": "unknown",
            "quality": 0.0,
            "round": 0,
            "max_token_length": 0,
            "tool_call_failure_ratio": 0.0,
        }
        if "tag" in data:
            # default_tag.update(data['tag'])
            for key in default_tag:
                if key in data["tag"]:
                    default_tag[key] = data["tag"][key]
        data["tag"] = default_tag
        if "sample_id" not in data:
            data["sample_id"] = "unknown"
        # if len(inputs['input_ids']) < tokenizer.model_max_length:
        if len(exp.tokens) < tokenizer.model_max_length:
            subset.append(data)
            lengths.append(len(exp.tokens))
    return subset, lengths


def chunkify(lst, n):
    """将列表均匀切分为 n 份"""
    size = max(1, len(lst) // n)
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def normalize_data(model_path, input_data_path, output_data_path):
    if input_data_path.startswith("oss://"):
        dataset = download_from_oss(input_data_path)
    else:
        with open(input_data_path, "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]

    # 将数据切分为多个 batch，分配给各进程
    num_workers = min(cpu_count(), 100, len(dataset))  # 避免进程数过多
    batches = chunkify(dataset, num_workers)

    worker_fn = partial(process_batch, model_path=model_path)

    final_dataset = []
    data_lengths = []
    with Pool(processes=num_workers) as pool:
        # imap 保持顺序，tqdm 跟踪进度（以 batch 为单位）
        for subset, lengths in tqdm(
            pool.imap(worker_fn, batches), total=len(batches), desc=input_data_path
        ):
            final_dataset.extend(subset)
            data_lengths.extend(lengths)

    data_lengths = np.array(data_lengths)
    print(f"mean input length: {np.mean(data_lengths)}")
    print(f"sum input length:  {np.sum(data_lengths)}")
    print(f"max input length:  {np.max(data_lengths)}")
    print(f"{(data_lengths > 262144).sum()=}")

    output_data_path = Path(output_data_path)
    output_dir = output_data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(output_data_path)
    with open(str(output_data_path), "w") as f:
        for data in final_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    from datasets import load_dataset

    dataset = load_dataset("json", data_files={"train": str(output_data_path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-data-path", type=str, nargs="+", required=True)
    parser.add_argument("--output-data-path", type=str, nargs="+", required=True)
    args = parser.parse_args()

    assert len(args.input_data_path) == len(
        args.output_data_path
    ), "Input and output data paths must have the same length."

    for input_data_path, output_data_path in zip(args.input_data_path, args.output_data_path):
        normalize_data(args.model_path, input_data_path, output_data_path)
