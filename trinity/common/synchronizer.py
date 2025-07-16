"""  """

import asyncio
from typing import List

import ray

from trinity.common.config import Config
from trinity.common.constants import SyncStyle


class Synchronizer:
    def __init__(self, config: Config):
        self.config = config
        self.trainer_status = None
        self.last_trainer_sync_step = 0
        self.explorer_status = None
        self.last_explorer_sync_step = 0
        self.ready_count = 0
        self._ready_condition = asyncio.Condition()

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        explorer = ray.get_actor(self.config.explorer_name)
        await explorer.setup_weight_sync_group.remote(master_address, master_port, state_dict_meta)

    async def need_sync(self, step: int, module: str) -> bool:
        if module == "trainer":
            if self.config.synchronizer.sync_style == SyncStyle.FIXED:
                self.last_trainer_sync_step = step
                return step % self.config.synchronizer.sync_interval == 0
            else:  # dynamic
                if self.config.synchronizer.sync_style == SyncStyle.DYNAMIC_BY_TRAINER:
                    delta = step - self.last_trainer_sync_step
                    if delta >= self.config.synchronizer.sync_interval:
                        self.trainer_status = "waiting_sync"
                if self.ready_count % 2 == 1:
                    return True
                else:
                    return False
        elif module == "explorer":
            if self.config.synchronizer.sync_style == SyncStyle.FIXED:
                delta = step - self.config.synchronizer.sync_offset
                self.last_explorer_sync_step = step
                return delta > 0 and delta % self.config.synchronizer.sync_interval == 0
            else:
                need_sync = False
                if self.config.synchronizer.sync_style == SyncStyle.DYNAMIC_BY_EXPLORER:
                    delta = step - self.last_explorer_sync_step
                    if delta >= self.config.synchronizer.sync_interval:
                        need_sync = True
                else:
                    if self.trainer_status == "waiting_sync":
                        need_sync = True

                return need_sync
        else:
            raise ValueError(f"Invalid module: {module}")

    async def ready_to_sync(self, step: int, module: str):
        async with self._ready_condition:
            try:
                self.ready_count += 1
                if self.ready_count % 2 == 1:
                    await asyncio.wait_for(
                        self._ready_condition.wait_for(
                            lambda: self.ready_count % 2 == 0,
                        ),
                        timeout=self.config.synchronizer.sync_timeout,
                    )
                else:
                    self._ready_condition.notify_all()
                if module == "trainer":
                    self.last_trainer_sync_step = step
                elif module == "explorer":
                    self.last_explorer_sync_step = step
                return True
            except asyncio.TimeoutError:
                another_module = "Trainer" if module == "explorer" else "Explorer"
                self.logger.error(
                    f"{another_module} is not ready for model weight sync in {self.config.synchronizer.sync_timeout} seconds."
                )
                return False
            finally:
                self.trainer_status = None
                self.explorer_status = None
                # self.ready_count -= 1

    @classmethod
    def get_actor(cls, config: Config):
        return (
            ray.remote(cls)
            .options(name="synchronizer", namespace=config.ray_namespace, get_if_exists=True)
            .remote(config)
        )
