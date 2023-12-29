from typing import List, Dict, Callable

import asyncio
from litellm import acompletion
import pandas as pd
import os
import inspect
import hashlib


async def execute_pipeline_on_examples(
    prompt_template: str,
    examples: List[Dict],
    dataset_name: str,
    allow_gpt4: bool = True,
):
    # First see if these examples have already been processed
    # Hash the examples
    h = hashlib.sha256(str(examples).encode("utf-8")).hexdigest()
    path_name = f"/Users/shreyashankar/Documents/projects/promptdelta/paper_experiments/{dataset_name}/{h}.csv"
    print(h)
    if os.path.exists(path_name):
        print("Found cached results")
        return pd.read_csv(path_name)

    # Get batches of 5 examples each
    batches = [examples[i : i + 5] for i in range(0, len(examples), 5)]
    replies = []

    for batch in batches:
        gpt_4_tasks = []
        gpt_3_tasks = []
        prompts = []
        for example in batch:
            # example["language"] = "English"
            print(example)

            # make message history
            messages = [
                {
                    "content": prompt_template.format(**example),
                    "role": "user",
                },
            ]

            prompt = "\n\n".join(
                [f"{message['role']}:\n{message['content']}" for message in messages]
            )

            prompts.append(prompt)

            if allow_gpt4:
                # Run with GPT-4
                gpt_4_tasks.append(
                    acompletion(
                        model="azure/gpt-4-2",
                        messages=messages,
                    )
                )
            else:
                gpt_4_tasks.append(asyncio.sleep(0.01))

            # Also run with GPT-3
            gpt_3_tasks.append(
                acompletion(
                    model=f"azure/gpt-35-turbo-16k",
                    messages=messages,
                )
            )

        gpt_4_responses = await asyncio.gather(*gpt_4_tasks, return_exceptions=True)
        gpt_3_responses = await asyncio.gather(*gpt_3_tasks, return_exceptions=True)
        for example, gpt_4_response, gpt_3_response, prompt in zip(
            batch, gpt_4_responses, gpt_3_responses, prompts
        ):
            response_list = (
                [gpt_4_response, gpt_3_response] if allow_gpt4 else [gpt_3_response]
            )
            model_list = ["gpt-4", "gpt-3"] if allow_gpt4 else ["gpt-3"]

            for response, model in zip(response_list, model_list):
                if isinstance(response, Exception):
                    print(f"Error: {response}")
                    replies.append(
                        {
                            "prompt": prompt,
                            "example": example,
                            "response": "N/A",
                            "model": model,
                        }
                    )
                    continue

                try:
                    response = response["choices"][0]["message"]["content"]
                    replies.append(
                        {
                            "prompt": prompt,
                            "example": example,
                            "response": response,
                            "model": model,
                        }
                    )
                except Exception as e:
                    print(e)
                    replies.append(
                        {
                            "prompt": prompt,
                            "example": example,
                            "response": "N/A",
                            "model": model,
                        }
                    )

    # Save replies as df
    df = pd.DataFrame(replies)

    # Cache results
    df.to_csv(path_name, index=False)
    return df


async def execute_candidate_assertions(
    dataset_name: str,
    prompt_template: str,
    examples: List[Dict],
    assertions: List[Callable],
    allow_gpt4: bool = True,
):
    source_codes = str([inspect.getsource(func) for func in assertions])
    # h = hash(str(examples) + prompt_template + str(source_codes))
    h = hashlib.sha256(
        str(examples).encode("utf-8")
        + prompt_template.encode("utf-8")
        + source_codes.encode("utf-8")
    ).hexdigest()
    print(h)
    path_name = f"/Users/shreyashankar/Documents/projects/promptdelta/paper_experiments/{dataset_name}/assertion_res_{h}.csv"
    if os.path.exists(path_name):
        print("Found cached results")
        return pd.read_csv(path_name)

    # Load replies
    reply_df = await execute_pipeline_on_examples(
        prompt_template, examples, dataset_name, allow_gpt4
    )
    all_results = []

    for _, row in reply_df.iterrows():
        # Async run evals
        tasks = []
        rows = []
        for func in assertions:
            if isinstance(row["example"], str):
                example = eval(row["example"])
            else:
                example = row["example"]

            if dataset_name == "codereviews":
                # Evaluate the value
                example = eval(example["pr_webhook_payload"])

            tasks.append(func(example, row["prompt"], row["response"]))
            row_dict = row.to_dict()
            row_dict["function_name"] = func.__name__
            rows.append(row_dict)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for row, result in zip(rows, results):
            if isinstance(result, Exception):
                print(f"Error: {result} from {row['function_name']}")
                row["result"] = f"Error: {result} from {row['function_name']}"
                row["prompt_tokens"] = None
                row["completion_tokens"] = None
                continue

            # if result is a tuple of 3, then it's a result from ask_llm
            if isinstance(result, tuple):
                row["result"] = result[2]
                row["prompt_tokens"] = result[0]
                row["completion_tokens"] = result[1]

            else:
                row["result"] = result
                row["prompt_tokens"] = None
                row["completion_tokens"] = None

        result_df = pd.DataFrame(rows)
        all_results.append(result_df)

        big_result_df = pd.concat(all_results)

        # Sleep for 2 seconds
        print("Sleeping for 2 seconds")
        await asyncio.sleep(2)

    # Cache results
    big_result_df.to_csv(path_name, index=False)
    return big_result_df
