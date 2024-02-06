# SETUP
import os
import time
import difflib
from nltk.tokenize import sent_tokenize
from modal import Image, Stub, gpu, method, NetworkFileSystem
import re
import json
from spade_v3.assertion_gen import generate_assertions


# volume = NetworkFileSystem.persisted("huggingface-cache")
MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
GPU_CONFIG = gpu.A100(memory=80, count=2)
TEMPLATES = [
    "You are an Ai asst at that’s an expert at reviewing pull requests. Review the below pull request. \n\n- Only answer on what can be improved and provide the improvement in code. \n- Answer in short form. \n- Adhere to the languages code conventions\n\n\n\n{pr_webhook_payload}",
    "You are an Ai asst at that’s an expert at reviewing pull requests. Review the below pull request. \n\n- Only answer on what can be improved and provide the improvement in code. \n- Answer in short form. \n- Adhere to the languages code conventions\n\n\n\n{pr_webhook_payload}",
    "You are an Ai asst at that’s an expert at reviewing pull requests. Review the below pull request. \n\n- Take into account that you don’t have access to the full code but only the code diff.\n- Only answer on what can be improved and provide the improvement in code. \n- Answer in short form. \n- Adhere to the languages code conventions\n- Make it personal and always show gratitude to the author\n\n\n\n{pr_webhook_payload}",
    "You are an Ai asst at that’s an expert at reviewing pull requests. Review the below pull request. \n\nWorlflow:\n1. Use the get_diff function to fetch the code changes using the the diff_url\n2. Do code review on the content of diff_url by following below instructions\n\nInstructions\n- Take into account that you don’t have access to the full code but only the code diff.\n- Only answer on what can be improved and provide the improvement in code. \n- Answer in short form. \n- Adhere to the languages code conventions\n- Make it personal and always show gratitude to the author\n\n\n\n{pr_webhook_payload}",
    'You are an Ai asst at that’s an expert at reviewing pull requests. Review the below pull request. \n\nWorlflow:\n1. Use the get_diff function to fetch the code changes using the the diff_url\n2. Do code review on the content of diff_url by following below instructions\n\nInstructions\n- Take into account that you don’t have access to the full code but only the code diff.\n- Only answer on what can be improved and provide the improvement in code. \n- Answer in short form. \n- Include code snippets if necessary.\n- Adhere to the languages code conventions.\n- Make it personal and always show gratitude to the author using "@" when tagging.\n\n\n\n{pr_webhook_payload}',
    'You are an AI Assistant that’s an expert at reviewing pull requests. Review the below pull request. \n\nWorlflow:\n1. Use the get_diff function to fetch the code changes using the the diff_url\n2. Do code review on the content of diff_url by following below instructions\n\nInstructions\n- Take into account that you don’t have access to the full code but only the code diff.\n- Only answer on what can be improved and provide the improvement in code. \n- Answer in short form. \n- Include code snippets if necessary.\n- Adhere to the languages code conventions.\n- Make it personal and always show gratitude to the author using "@" when tagging.\n\n\n\n{pr_webhook_payload}',
    'You are an AI Assistant that’s an expert at reviewing pull requests. Review the below pull request that you receive. \n\nInstructions\n- Take into account that you don’t have access to the full code but only the code diff.\n- Only answer on what can be improved and provide the improvement in code. \n- Answer in short form. \n- Include code snippets if necessary.\n- Adhere to the languages code conventions.\n- Make it personal and always show gratitude to the author using "@" when tagging.\n\n\n\n{pr_webhook_payload}',
    'You are an AI Assistant that’s an expert at reviewing pull requests. Review the below pull request that you receive. \n\nInput format\n- The input format follows Github diff format with addition and subtraction of code.\n- The + sign means that code has been added.\n- The - sign means that code has been removed.\n\nInstructions\n- Take into account that you don’t have access to the full code but only the code diff.\n- Only answer on what can be improved and provide the improvement in code. \n- Answer in short form. \n- Include code snippets if necessary.\n- Adhere to the languages code conventions.\n- Make it personal and always show gratitude to the author using "@" when tagging.\n\n\n\n{pr_webhook_payload}',
]
CONCEPT_TEMPLATE = """Here is the diff for my prompt template:

"{prompt_diff}"

Based on the changed lines, I want to write assertions for my LLM pipeline to run on all pipeline responses. Here are some categories of assertion concepts I want to check for:

- Presentation Format: Is there a specific format for the response, like a comma-separated list or a JSON object?
- Example Demonstration: Does theh prompt template include any examples of good responses that demonstrate any specific headers, keys, or structures?
- Workflow Description: Does the prompt template include any descriptions of the workflow that the LLM should follow, indicating possible assertion concepts?
- Count: Are there any instructions regarding the number of items of a certain type in the response, such as “at least”, “at most”, or an exact number?
- Inclusion: Are there keywords that every LLM response should include?
- Exclusion: Are there keywords that every LLM response should never mention?
- Qualitative Assessment: Are there qualitative criteria for assessing good responses, including specific requirements for length, tone, or style?
- Other: Based on the prompt template, are there any other concepts to check in assertions that are not covered by the above categories, such as correctness, completeness, or consistency?

Give me a list of concepts to check for in LLM responses. Each item in the list should contain a string description of a concept to check for, its corresponding category, and the source, or phrase in the prompt template that triggered the concept. For example, if the prompt template is "I am a still-life artist. Give me a bulleted list of colors that I can use to paint <object>.", then a concept might be "The response should include a bulleted list of colors." with category "Presentation Format" and source "Give me a bulleted list of colors".

Your answer should be a JSON list of objects within ```json ``` markers, where each object has the following fields: "concept", "category", and "source". This list should contain as many assertion concepts as you can think of, as long are specific and reasonable."""


def show_diff(template_1: str, template_2: str):
    # Split the templates into sentences
    if isinstance(template_1, list):
        template_1 = str(template_1)
    if isinstance(template_2, list):
        template_2 = str(template_2)

    sent_1 = sent_tokenize(template_1)
    sent_2 = sent_tokenize(template_2)

    diff = list(difflib.unified_diff(sent_1, sent_2))

    # Convert diff to string
    diff = "\n".join(diff)
    return diff


# Define a container image
# create a Modal image which has the model weights pre-saved to a directory.
def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns="*.pt",  # Using safetensors
    )
    move_cache()


vllm_image = (
    Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install(
        "vllm==0.2.5",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
        "nltk",
        "litellm",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder, timeout=60 * 20)
)

stub = Stub("example-vllm-mixtral", image=vllm_image)


# The model class
# __enter__ enables us to load the model into memory
@stub.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    image=vllm_image,
)
class Model:
    def __enter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        if GPU_CONFIG.count > 1:
            # Patch issue from https://github.com/vllm-project/vllm/issues/1116
            import ray

            ray.shutdown()
            ray.init(num_gpus=GPU_CONFIG.count)

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.template = "<s> [INST] {user} [/INST] "

        # Performance improvement from https://github.com/vllm-project/vllm/issues/2073#issuecomment-1853422529
        if GPU_CONFIG.count > 1:
            import subprocess

            RAY_CORE_PIN_OVERRIDE = "cpuid=0 ; for pid in $(ps xo '%p %c' | grep ray:: | awk '{print $1;}') ; do taskset -cp $cpuid $pid ; cpuid=$(($cpuid + 1)) ; done"
            subprocess.call(RAY_CORE_PIN_OVERRIDE, shell=True)

    @method()
    async def completion_stream(self, user_question):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=1024,
            repetition_penalty=1.1,
        )

        t0 = time.time()
        request_id = random_uuid()
        result_generator = self.engine.generate(
            self.template.format(user=user_question),
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        async for output in result_generator:
            if output.outputs[0].text and "\ufffd" == output.outputs[0].text[-1]:
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta

        print(f"Generated {num_tokens} tokens in {time.time() - t0:.2f}s")


def construct_prompt(prompt_diff):
    messages = [
        {
            "content": "You are an expert Python programmer and helping me write assertions for my LLM pipeline. An LLM pipeline accepts an example and prompt template, fills the template's placeholders with the example, and generates a response.",
            "role": "system",
        },
        {
            "content": CONCEPT_TEMPLATE.format(prompt_diff=prompt_diff),
            "role": "user",
        },
    ]
    return messages


# TODO get prompt templates
def get_prompt_templates():
    return TEMPLATES


def generate_prompt(curr_template, prev_template):
    prompt_diff = show_diff(prev_template, curr_template)
    print(f"Prompt diff: {prompt_diff}")

    if len(prompt_diff.strip()) == 0:
        return None

    concept_prompt_part = CONCEPT_TEMPLATE.format(prompt_diff=prompt_diff)

    message = f"You are an expert Python programmer and helping me write assertions for my LLM pipeline. An LLM pipeline accepts an example and prompt template, fills the template's placeholders with the example, and generates a response.\n\n{concept_prompt_part}"

    # construct prompt to LLM
    return message


# Run the model
@stub.local_entrypoint()
def main():
    model = Model()
    prompt_templates = get_prompt_templates()
    for prev_template, curr_template in zip(
        [""] + prompt_templates[:-1], prompt_templates
    ):
        prompt = generate_prompt(curr_template, prev_template)
        print("Sending new request:", prompt)
        reply = list(model.completion_stream.remote_gen(prompt))
        reply = "".join(reply)

        print(f"Raw reply: {reply}")

        # there is a parsing error here, need to experiment/debug
        reply = re.search(r"```json(.*?)```", reply, re.DOTALL).group(1).strip()
        print(f"JSON reply: {reply}")

        # concepts = json.loads(reply)

        # Now generate assertions (don't call this function)
        # assertions = generate_assertions(
        #     curr_template, curr_template, response, concepts
        # )
        # return {"concepts": concepts, "assertions": assertions}
