from litellm import acompletion
import re
import json


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for an AI coding expert that tries to determine whether two functions are doing the same thing\n\n{prompt}\n\nHere is the response:\n{response}",
            "role": "system",
        },
        {
            "content": f"{question}\nOnly answer yes or no.",
            "role": "user",
        },
    ]

    response = await acompletion(
        model="azure/gpt-35-turbo",
        messages=messages,
    )

    # get the cost
    completion_tokens = response["usage"]["completion_tokens"]
    prompt_tokens = response["usage"]["prompt_tokens"]

    # get the response
    reply = response["choices"][0]["message"]["content"]

    if "yes" in reply.lower():
        return prompt_tokens, completion_tokens, True

    return prompt_tokens, completion_tokens, False


async def assert_response_json_format(example: dict, prompt: str, response: str):
    """
    Check if the response is formatted as a JSON object within code markers.
    """
    try:
        response = response.strip()
        # Check if response starts and ends with ```json and ``` markers respectively
        if not (response.startswith("```json") and response.endswith("```")):
            return False
        # Attempt to parse the JSON object within the markers
        json_content = response[7:-3].strip()  # remove markers and whitespace
        parsed_json = json.loads(json_content)
        return isinstance(parsed_json, dict)
    except (ValueError, json.JSONDecodeError):
        return False


async def assert_response_json_keys(example: dict, prompt: str, response: str):
    """
    Check if the JSON object includes the keys 'answer' and 'input' with correct values based on 'answer'.
    """
    try:
        response = response.strip(
            "\n` "
        )  # Strips possible newline, backticks and space characters
        json_content = json.loads(
            response[4:-3]
        )  # Excludes the ```json and ``` including wrapping spaces

        # First, check if both keys are present
        if "answer" not in json_content or "input" not in json_content:
            return False

        # Then, check if 'answer' contains a valid value
        answer_value = json_content["answer"]
        if answer_value not in ["yes", "no"]:
            return False

        # Finally, if 'answer' is 'no', 'input' must be None
        if answer_value == "no" and json_content["input"] is not None:
            return False

        return True
    except json.JSONDecodeError:
        return False


async def assert_functions_referenced_correctly(
    example: dict, prompt: str, response: str
):
    """
    Check that the response refers to the functions by their names correctly as `{func_a}` and `{func_b}`.
    """
    func_a_name = example[
        "func_a"
    ]  # Assuming this is how you would extract the function name
    func_b_name = example["func_b"]

    if f"`{func_a_name}`" not in response or f"`{func_b_name}`" not in response:
        return False

    return True


async def assert_answers_quality_assessment(example: dict, prompt: str, response: str):
    """
    Check that the answer includes a correct qualitative assessment about the difference in functions' output.
    """
    question = "Does the LLM provide a correct qualitative assessment about the functions' output differences?"
    # The response from the expert LLM to this question must be interpreted within context.
    return await ask_llm(prompt, response, question)


async def assert_similarity_in_ask_llm_calls(example: dict, prompt: str, response: str):
    """
    Check whether the answer includes a correct assessment of whether both called `ask_llm` functions with similar prompts.
    """
    question = "Does the answer correctly assess the similarity of calls to `ask_llm` functions in both provided functions?"
    # This question is sent to an expert LLM to determine the correctness of the assertion within the context.
    return await ask_llm(prompt, response, question)


async def assert_no_direct_code_in_response(example: dict, prompt: str, response: str):
    """
    Check that the response does not include any direct code from `func_a_src` or `func_b_src`.
    """
    func_a_src = example[
        "func_a_src"
    ]  # Assuming we're getting the source code as a string from the example
    func_b_src = example["func_b_src"]

    if func_a_src in response or func_b_src in response:
        return False

    return True


async def assert_json_format_response(example: dict, prompt: str, response: str):
    """
    Check if the response is formatted as a JSON object within triple backticks.
    """
    return response.strip().startswith("```json") and response.strip().endswith("```")


async def assert_json_keys_answer_response(example: dict, prompt: str, response: str):
    """
    Check that the response JSON object includes the keys 'answer' and 'response'.
    """
    try:
        response_dict = json.loads(response.strip("```json "))
        return all(key in response_dict for key in ["answer", "response"])
    except json.JSONDecodeError:
        return False


async def assert_json_values_types(example: dict, prompt: str, response: str):
    """
    Check the LLM response for boolean 'answer' and string 'response', with 'N/A' for no cases.
    """
    try:
        response_dict = json.loads(response.strip("```json "))
        if "answer" in response_dict and "response" in response_dict:
            answer_ok = isinstance(response_dict["answer"], bool)
            response_ok = isinstance(response_dict["response"], str)
            if not answer_ok or (not response_ok):
                return False
            if response_dict["answer"] is False:
                return response_dict["response"] == "N/A"
            return True
        return False
    except json.JSONDecodeError:
        return False


async def assert_no_numeric_count_required(example: dict, prompt: str, response: str):
    """
    Check if the prompt does not specify numeric count requirements and that the LLM response does not include them.
    """
    # This is a qualitative check, rather than quantitative, may require 'ask_llm'
    question = "Does the prompt specify any numerical count requirements or is it necessary for the LLM response?"
    return not await ask_llm(prompt, response, question)


async def assert_func_b_func_a_disagreement_check(
    example: dict, prompt: str, response: str
):
    """
    Check if there's a claim about a different response such that function func_b returns False and func_a returns True.
    """
    question = "Is there some different response such that function `{func_b}` returns False for while function `{func_a}` returns True?"
    answer = await ask_llm(prompt, response, question)
    if answer == True and "N/A" in response:
        return False

    return True


async def assert_no_duplicate_ask_llm_inputs(example: dict, prompt: str, response: str):
    """
    Check if the LLM response does not include duplicate 'ask_llm' inputs when the functions check for the same thing.
    """
    question = "Do both functions contain `ask_llm` calls to check for the same thing, and if so, is the answer no?"
    return await ask_llm(prompt, response, question)


async def assert_correct_qualitative_assessment(
    example: dict, prompt: str, response: str
):
    """
    Check if the response correctly assesses the functions based on the criteria of different possible outputs for 'ask_llm' calls.
    """
    # This is similar to the assert_func_b_func_a_disagreement_check but focuses on the correctness of the qualitative assessment itself.
    question = "Does the response correctly evaluate whether the functions have different possible outputs for similar 'ask_llm' calls?"
    return await ask_llm(prompt, response, question)


async def assert_represent_disagreement_correctly(
    example: dict, prompt: str, response: str
):
    """
    Check if the LLM response correctly represents whether there is some different response such that `func_b` and `func_a` disagree.
    """
    question = "Does the response correctly represent whether there is some different response such that `func_b` returns False while `func_a` returns True?"
    return await ask_llm(prompt, response, question)


async def assert_json_response_format(example: dict, prompt: str, response: str):
    """
    Check that the response is in the correct JSON format with the specified
    keys 'answer' and 'response'.
    """
    try:
        response_json = json.loads(response)
        return (
            "answer" in response_json
            and response_json["answer"] in {"yes", "no"}
            and "response" in response_json
            and (response_json["answer"] == "no" and response_json["response"] == "N/A")
        )
    except json.JSONDecodeError:
        return False


async def assert_no_ask_llm_calls(example: dict, prompt: str, response: str):
    """
    Check that the response does not contain 'ask_llm' calls for the same check
    in both functions.
    """
    return "ask_llm" not in response


async def assert_correct_implication_relationship(
    example: dict, prompt: str, response: str
):
    """
    Ask the LLM to evaluate whether the response correctly interprets the implication
    relationship between the two functions.
    """
    question = (
        "Is the response correctly interpreting the implication relationship "
        "between the two functions according to the given example and prompt?"
    )
    return await ask_llm(prompt, response, question)


async def assert_example_implication_logic(example: dict, prompt: str, response: str):
    """
    Check that the response correctly determines if there exists an example such that function
    B returns False while function A returns True.
    """
    question = (
        "Does the example demonstrate a case where function B would return False "
        "while function A returns True?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_inclusion_consistency(
    example: dict, prompt: str, response: str
):
    """
    Evaluate if the response is consistent with the prompt change regarding the implication
    of function A to function B.
    """
    question = (
        "Is the response consistent with the prompt change from asking for different "
        "response possibilities to asking the implication relationship?"
    )
    return await ask_llm(prompt, response, question)


async def assert_json_response_enclosure(example: dict, prompt: str, response: str):
    """
    Check that the response JSON is enclosed within triple backticks.
    """
    return response.strip().startswith("```json") and response.strip().endswith("```")


ALL_FUNCTIONS = [
    assert_response_json_format,
    assert_response_json_keys,
    assert_functions_referenced_correctly,
    assert_answers_quality_assessment,
    assert_similarity_in_ask_llm_calls,
    assert_no_direct_code_in_response,
    assert_json_format_response,
    assert_json_keys_answer_response,
    assert_json_values_types,
    assert_no_numeric_count_required,
    assert_func_b_func_a_disagreement_check,
    assert_no_duplicate_ask_llm_inputs,
    assert_correct_qualitative_assessment,
    assert_represent_disagreement_correctly,
    assert_json_response_format,
    assert_no_ask_llm_calls,
    assert_correct_implication_relationship,
    assert_example_implication_logic,
    assert_response_inclusion_consistency,
    assert_json_response_enclosure,
]
