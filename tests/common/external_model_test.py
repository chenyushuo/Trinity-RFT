import os
import unittest

import ray

from trinity.common.config import ExternalModelConfig, InferenceModelConfig
from trinity.common.models.allocator import get_external_model_wrapper


class TestExternalModel(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True, namespace="trinity_unittest")

    @classmethod
    def tearDownClass(cls):
        ray.shutdown(_exiting_interpreter=True)

    async def test_external_model_load(self):
        mock_base_url = "https://mock.external.endpoint/"
        mock_api_key = "dummy-api-key"
        base_url_env = "TRINITY_OPENAI_BASE_URL_TEST"
        api_key_env = "TRINITY_OPENAI_API_KEY_TEST"
        os.environ[base_url_env] = mock_base_url
        os.environ[api_key_env] = mock_api_key
        self.addCleanup(os.environ.pop, base_url_env, None)
        self.addCleanup(os.environ.pop, api_key_env, None)
        config = InferenceModelConfig(
            model_path="mock-model-name",
            engine_type="external",
            external_model_config=ExternalModelConfig(
                enable=True,
                base_url_env=base_url_env,
                api_key_env=api_key_env,
            ),
        )

        wrapper = await get_external_model_wrapper(config=config)

        self.assertEqual(wrapper.api_address, mock_base_url.rstrip("/"))
        self.assertEqual(wrapper.api_key, mock_api_key)

        client = wrapper.get_openai_client()
        self.assertEqual(str(client.base_url).rstrip("/"), f"{wrapper.api_address}/v1")
        self.assertEqual(client.api_key, mock_api_key)
