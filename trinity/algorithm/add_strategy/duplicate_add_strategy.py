# -*- coding: utf-8 -*-
import asyncio
import copy
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from trinity.algorithm.add_strategy.add_strategy import (
    ADD_STRATEGY,
    AddStrategy,
    group_by,
)
from trinity.buffer import BufferWriter
from trinity.common.experience import Experience
from trinity.utils.monitor import gather_metrics
from trinity.utils.timer import Timer


@ADD_STRATEGY.register_module("duplicate_informative")
class DuplicateInformativeAddStrategy(AddStrategy):
    """An AddStrategy that filters experiences based on reward variance and duplicates them to reach the target size.
    Ref: POLARIS (https://hkunlp.github.io/blog/2025/Polaris)
    """

    def __init__(self, writer: BufferWriter, variance_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(writer)
        self.variance_threshold = variance_threshold
        self.epsilon = 1e-6

    async def add(self, experiences: List[Experience], step: int) -> Tuple[int, Dict]:
        if len(experiences) == 0:
            return 0, {}
        cnt = 0
        metrics = {}
        cnt_tot = len(experiences)
        effective_tasks, effective_experiences = [], []
        with Timer(metrics, "add_strategy_time"):
            grouped_experiences = group_by(experiences, id_type="task")
            for task_id, group_exps in grouped_experiences.items():
                if len(group_exps) < 2:
                    continue
                # check if the rewards are the same
                rewards = [exp.reward for exp in group_exps]
                variance = np.var(rewards)
                if variance <= self.variance_threshold:
                    continue
                cnt += len(group_exps)
                effective_tasks.append(task_id)
                effective_experiences.extend(group_exps)

            if not effective_tasks:
                return 0, metrics

        task_ids_to_add = effective_tasks.copy()
        task_id_offset = len(grouped_experiences)

        metrics["filtered_group_advantages/filtered_proportion"] = (
            1.0 - len(effective_experiences) / cnt_tot
        )
        while cnt < cnt_tot:
            if not task_ids_to_add:
                task_ids_to_add = effective_tasks.copy()
                random.shuffle(task_ids_to_add)
                task_id_offset += len(grouped_experiences)
            task_id = task_ids_to_add.pop()

            copied_exps = copy.deepcopy(grouped_experiences[task_id])

            for exp in copied_exps:
                exp.eid.task += task_id_offset

            cnt += len(copied_exps)
            effective_experiences.extend(copied_exps)

        # await self.writer.write_async(effective_experiences)
        # calculate origin exps group advantage
        exp_groups = self.group_experiences(experiences)
        cnt = 0
        metric_list = []
        for group_id, group_exps in exp_groups.items():
            group_exps, group_metrics = self.calculate_group_advantage(group_id, group_exps)
            metric_list.append(group_metrics)
            cnt += len(group_exps)
        try:
            metrics.update(gather_metrics(metric_list, "group_advantages"))
        except ValueError:
            pass  # empty metric list causes ValueError, ignore it

        # calculate effective_experiences group advantage
        exp_groups = self.group_experiences(effective_experiences)
        cnt = 0
        metric_list = []
        tasks = []
        for group_id, group_exps in exp_groups.items():
            group_exps, group_metrics = self.calculate_group_advantage(group_id, group_exps)
            metric_list.append(group_metrics)
            cnt += len(group_exps)
            if len(group_exps) > 0:
                tasks.append(self.writer.write_async(group_exps))
        if tasks:
            await asyncio.gather(*tasks)
        try:
            metrics.update(gather_metrics(metric_list, "filtered_group_advantages"))
        except ValueError:
            pass  # empty metric list causes ValueError, ignore it

        return cnt, metrics

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)
            for exp in exps:
                score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"variance_threshold": 0.0}


@ADD_STRATEGY.register_module("remove_zero_advantage")
class RemoveZeroAdvantageAddStrategy(AddStrategy):
    """ """

    def __init__(self, writer: BufferWriter, variance_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(writer)
        self.variance_threshold = variance_threshold
        self.epsilon = 1e-6

    async def add(self, experiences: List[Experience], step: int) -> Tuple[int, Dict]:
        if len(experiences) == 0:
            return 0, {}
        cnt = 0
        metrics = {}
        cnt_tot = len(experiences)
        effective_tasks, effective_experiences = [], []
        rewards_mean_counts = defaultdict(int)
        with Timer(metrics, "add_strategy_time"):
            grouped_experiences = group_by(experiences, id_type="task")
            for task_id, group_exps in grouped_experiences.items():
                if len(group_exps) < 2:
                    continue
                # check if the rewards are the same
                rewards = [exp.reward for exp in group_exps]
                rewards_mean_counts[np.mean(rewards)] += len(group_exps)
                variance = np.var(rewards)
                if variance <= self.variance_threshold:
                    continue
                cnt += len(group_exps)
                effective_tasks.append(task_id)
                effective_experiences.extend(group_exps)

            if not effective_tasks:
                return 0, metrics

        # task_ids_to_add = effective_tasks.copy()
        # task_id_offset = len(grouped_experiences)

        range_bins = [
            (0.0, "0"),
            (0.2, "(0,0.2]"),
            (0.4, "(0.2,0.4]"),
            (0.6, "(0.4,0.6]"),
            (0.8, "(0.6,0.8]"),
            (1.0, "(0.8,1)"),
            (1.0, "1"),  # 单独处理 key == 1 的情况
        ]

        # 初始化 metrics 字典中的所有相关键
        for _, label in range_bins:
            metrics[f"filtered_group_advantages/rewards_mean_range/{label}"] = 0

        # 遍历 rewards_mean_counts 并归类
        for key, value in rewards_mean_counts.items():
            normalized_value = value / cnt_tot
            if key == 0 or key == 1:
                metrics[f"filtered_group_advantages/rewards_mean_range/{int(key)}"] += normalized_value
            else:
                # 找到所属区间 (0 < key < 1)
                for threshold, label in range_bins[1:-1]:  # 排除第一个 0 和最后一个 1
                    if key <= threshold:
                        metrics[
                            f"filtered_group_advantages/rewards_mean_range/{label}"
                        ] += normalized_value
                        break

        metrics["filtered_group_advantages/filtered_proportion"] = (
            1.0 - len(effective_experiences) / cnt_tot
        )
        # while cnt < cnt_tot:
        #     if not task_ids_to_add:
        #         task_ids_to_add = effective_tasks.copy()
        #         random.shuffle(task_ids_to_add)
        #         task_id_offset += len(grouped_experiences)
        #     task_id = task_ids_to_add.pop()

        #     copied_exps = copy.deepcopy(grouped_experiences[task_id])

        #     for exp in copied_exps:
        #         exp.eid.task += task_id_offset

        #     cnt += len(copied_exps)
        #     effective_experiences.extend(copied_exps)

        # await self.writer.write_async(effective_experiences)
        # calculate origin exps group advantage
        exp_groups = self.group_experiences(experiences)
        cnt = 0
        metric_list = []
        for group_id, group_exps in exp_groups.items():
            group_exps, group_metrics = self.calculate_group_advantage(group_id, group_exps)
            metric_list.append(group_metrics)
            cnt += len(group_exps)
        try:
            metrics.update(gather_metrics(metric_list, "group_advantages"))
        except ValueError:
            pass  # empty metric list causes ValueError, ignore it

        # calculate effective_experiences group advantage
        exp_groups = self.group_experiences(effective_experiences)
        cnt = 0
        metric_list = []
        tasks = []
        for group_id, group_exps in exp_groups.items():
            group_exps, group_metrics = self.calculate_group_advantage(group_id, group_exps)
            metric_list.append(group_metrics)
            cnt += len(group_exps)
            if len(group_exps) > 0:
                tasks.append(self.writer.write_async(group_exps))
        if tasks:
            await asyncio.gather(*tasks)
        try:
            metrics.update(gather_metrics(metric_list, "filtered_group_advantages"))
        except ValueError:
            pass  # empty metric list causes ValueError, ignore it

        return cnt, metrics

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)
            for exp in exps:
                score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"variance_threshold": 0.0}
