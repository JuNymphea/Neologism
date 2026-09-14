"""Async access to a remote chat model, with a prompt-level disk cache.

The cache is what makes re-running cheap: keys are the prompt text alone, and
the file is `{model}_{tag}_cache.pkl`. Both are kept exactly as they were so
the existing caches under `cached_data/persist_lm_cache/` stay valid -- they
hold hundreds of megabytes of already-paid-for judgements.
"""

import asyncio
import logging
import pickle
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

#: Judge models this code has been used with. The list exists to catch typos
#: in `--model`, not to gatekeep: add the name and it works. It used to be an
#: if/elif chain of `pass` statements, which meant every new judge needed a
#: code edit in the middle of a function.
SUPPORTED_MODELS = (
    "gemini-2.5-flash",
    "gpt-5.2-2025-12-11",
    "gpt-5.4-nano",
)


class LanguageModel(object):
    """Main class abstract async remote language model access"""

    def __init__(self, model, client, dump_dir=None, use_cache=True,
                 cache_level="api", **kwargs):
        self.model = model
        if not any(name in model for name in SUPPORTED_MODELS):
            raise ValueError(
                f"{model} model class is not supported yet. "
                f"Known: {', '.join(SUPPORTED_MODELS)}. "
                f"Add it to SUPPORTED_MODELS in {__file__} if it is a new judge.")
        self.client = client
        self.dump_dir = None
        if dump_dir:
            self.dump_dir = Path(dump_dir) / "lm_cache"
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = kwargs.get("temperature", 1.0)
        self.cache_dir = None
        self.use_cache = use_cache
        self.cache_level = cache_level
        self.cache_in_mem = {}
        self.api_count = {}
        if self.use_cache:
            assert kwargs.get("master_data_dir", None), "master_data_dir is required for cache"
            self.cache_dir = Path(kwargs["master_data_dir"]) / "persist_lm_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # load cache from disk
            if kwargs.get("cache_tag", None):
                self.cache_file = self.cache_dir / f"{self.model}_{kwargs['cache_tag']}_cache.pkl"
            else:
                self.cache_file = self.cache_dir / f"{self.model}_cache.pkl"
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    self.cache_in_mem = pickle.load(f)
                logger.warning(f"loaded {len(self.cache_in_mem)} cached judgements "
                               f"from {self.cache_file}")

    def normalize(self, text):
        return text.strip()

    def _get_cache_key(self, prompt, api_count, api_name):
        if self.cache_level and self.cache_level == "prompt":
            return f"{prompt}"
        return f"{prompt}_____{api_count}_____{api_name}"

    async def chat_completion(self, client, prompt, api_name):
        # check if the prompt is cached
        api_count = self.api_count.get(api_name, 0)
        self.api_count[api_name] = api_count + 1  # increment api count
        if self.use_cache:
            cache_key = self._get_cache_key(prompt, api_count, api_name)
            if cache_key in self.cache_in_mem:
                return (self.cache_in_mem[cache_key], None)
        raw_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model, temperature=self.temperature)
        raw_completion = raw_completion.to_dict()
        completion = self.normalize(raw_completion["choices"][0]["message"]["content"])

        if self.use_cache:
            self.cache_in_mem[cache_key] = completion
        usage = raw_completion['usage']
        return (completion, usage)

    async def chat_completions(self, api_names, prompts, batch_size=32):
        """handling batched async calls with internal batching mechanism"""
        # Ensure api_names is a list of appropriate length
        if not isinstance(api_names, list):
            api_names = [api_names] * len(prompts)

        # Process in batches
        all_completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_api_names = api_names[i:i + batch_size]

            # batched calls
            async_responses = [
                self.chat_completion(self.client, prompt, api_name)
                for prompt, api_name in zip(batch_prompts, batch_api_names)]
            raw_completions = await asyncio.gather(*async_responses)
            # post handling for current batch
            for j, (completion, usage) in enumerate(raw_completions):
                all_completions.append(completion)

        return all_completions

    def save_cache(self):
        if self.use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache_in_mem, f, protocol=pickle.HIGHEST_PROTOCOL)

    async def close(self):
        """Close the underlying HTTP client"""
        await self.client.close()
