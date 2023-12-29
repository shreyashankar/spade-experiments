from litellm import acompletion
import re


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for a large language model pipeline that turns exercise video texts into markdown programs of step by step instructions:\n\n{prompt}\n\nHere is the response:\n{response}",
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


async def assert_as_markdown_format(
    example: dict, formatted_prompt: str, response: str
) -> bool:
    """
    Check if the response is in markdown format.
    """
    patterns = [
        r"^#{1,6}\s",  # Header
        r"^\s*[-*]\s",  # Unordered list
        r"^\s*\d+\.\s",  # Ordered list
        r"\[.+\]\(.+\)",  # Link
        r"\*\*.+\*\*",  # Bold text
        r"\*.+\*",  # Italic text
        r"`[^`]+`",  # Inline code
        r"^```\n[\s\S]*?\n```",  # Fenced code block
    ]

    # Check if any pattern matches
    for pattern in patterns:
        if re.search(pattern, response, re.MULTILINE):
            return True

    return False


async def assert_name_present_in_routine(
    example: dict, formatted_prompt: str, response: str
) -> bool:
    """
    Check if the response contains a section about the name of the routine.
    """
    return f"### \"{example['vid_name']}\" Routine:" in response


async def assert_name_without_follow_along(
    example: dict, formatted_prompt: str, response: str
) -> bool:
    """
    Verify that the routine name does not contain 'follow along'.
    """
    return "follow along" not in response.lower()


async def assert_summary_suitability_and_presence(
    example: dict, formatted_prompt: str, response: str
) -> bool:
    """
    Check that there is a summary section and it's not too long.
    """
    if "**Summary:**" not in response:
        return False
    summary_content = response.split("**Summary:**")[-1]
    summary_sentences = summary_content.split(".")
    return (
        0 < len(summary_sentences) <= 3
    )  # Assumes a couple of sentences can mean two or three.


async def assert_equipment_section_availability(
    example: dict, formatted_prompt: str, response: str
) -> bool:
    """
    Verify that the response contains a section listing the necessary equipment.
    """
    return "**Equipment Needed:**" in response


async def assert_post_exercise_repetitions(
    example: dict, formatted_prompt: str, response: str
) -> bool:
    """
    Check that the number of repetitions or the time is placed after the exercise name.
    """
    # Typically, this can be achieved by pattern matching, but we assume ask_llm is a fallback.
    return not any(
        exercise_time in desc
        for exercise_time in response.split("\n")
        if "(" in exercise_time
        for desc in response.split("\n")
        if ":" in desc
    )


async def assert_deduplication_side_exercises(
    example: dict, formatted_prompt: str, response: str
) -> bool:
    """
    Check that exercises performed on both sides mention 'on each side' without repetition in the list.
    """
    return "on each side" in response and response.count(
        "on each side"
    ) == response.count("(")


async def assert_non_splitable_routine_check(
    example: dict, formatted_prompt: str, response: str
) -> bool:
    """
    Check if the response indicates the absence of a follow-along routine when applicable.
    """
    question = "Does the response correctly indicate if the text does not contain a follow-along routine that can be split into exercises?"
    return await ask_llm(formatted_prompt, response, question)


async def assert_response_markdown_format(example: dict, prompt: str, response: str):
    """
    Check if the response is in markdown format.
    """
    return response.startswith("# ") and "\n## " in response and "\n### " in response


async def assert_routine_name_response_start(example: dict, prompt: str, response: str):
    """
    Check if the response starts with the routine name.
    """
    routine_name = example["vid_name"].replace("follow along", "").strip()
    return response.lstrip("# ").startswith(routine_name)


async def assert_absence_of_follow_along_in_name(
    example: dict, prompt: str, response: str
):
    """
    Check if the 'follow along' text is removed from the routine name.
    """
    return "follow along" not in response.split("\n", 1)[0].lower()


async def assert_summary_section_target_correctness(
    example: dict, prompt: str, response: str
):
    """
    Check if there is a summary section that describes the routine and the target audience.
    """
    question = "Is there a separate summary section below the routine name that describes the routine and its target audience, in no more than a couple of sentences?"
    return await ask_llm(prompt, response, question)


async def assert_equipment_section_existence(example: dict, prompt: str, response: str):
    """
    Check if there is a separate equipment section in the response.
    """
    if "## Equipment" not in response:
        return False
    equipment_section_index = response.index("## Equipment")
    return response[equipment_section_index:].count("\n") > 1


async def assert_exercises_with_time_or_reps_post_name(
    example: dict, prompt: str, response: str
):
    """
    Check if the exercises are listed with reps or time directly after the exercise name.
    """
    exercises_section = response.split("## Exercises\n", 1)[-1]
    for exercise in exercises_section.strip().split("\n\n"):
        if not exercise.startswith("### ") or "(" not in exercise:
            return False
    return True


async def assert_no_repeat_exercises_both_sides(
    example: dict, prompt: str, response: str
):
    """
    Check that exercises done on both sides are not repeated and have 'on each side' included.
    """
    question = "Are there any exercises listed for the right side and left side separately, instead of mentioning once with 'on each side'?"
    # The LLM will return True if there is an error; we need to invert that result
    return not await ask_llm(prompt, response, question)


async def assert_no_text_below_exercise_list(example: dict, prompt: str, response: str):
    """
    Check that there is no additional text below the exercises list.
    """
    exercises_ending = "### Cool-Down and Stretch:"
    return exercises_ending in response and not response.strip().endswith(
        exercises_ending
    )


async def assert_alternative_for_no_routine(example: dict, prompt: str, response: str):
    """
    Check for an alternative response when the text does not contain a follow-along routine that can be split into exercises.
    """
    question = "If the text does not contain a follow-along routine, does the response indicate so?"
    return await ask_llm(prompt, response, question)


async def assert_completeness_and_correctness_of_response(
    example: dict, prompt: str, response: str
):
    """
    Check the response for correctness and completeness, including summaries, equipment, and exercise lists.
    """
    question = "Does the response include a complete and correct summary, equipment list, and exercises list while adhering to the specified format?"
    return await ask_llm(prompt, response, question)


async def assert_markdown_format(example: dict, prompt: str, response: str):
    """
    Check if the response is in markdown format by looking for markdown headers (e.g., '#', '##', '###').
    """
    expected_headers = ["# ", "## ", "### "]
    return all(header in response for header in expected_headers)


async def assert_presence_headers_summary_equipment_exercises(
    example: dict, prompt: str, response: str
):
    """
    Check if the response contains specific headers 'Summary', 'Equipment', and 'Exercises'.
    """
    required_headers = ["## Summary", "## Equipment", "## Exercises"]
    return all(header in response for header in required_headers)


async def assert_no_text_post_exercises_list(example: dict, prompt: str, response: str):
    """
    Check that there is no text below the exercises list in the response.
    """
    exercises_end_index = response.rfind("###") + response.rsplit("###", 1)[-1].rfind(
        "\n"
    )
    return exercises_end_index == len(response.strip())


async def assert_descriptions_under_exercise_header(
    example: dict, prompt: str, response: str
):
    """
    Check that exercise descriptions appear under the 'Exercises' header.
    """
    try:
        exercises_start_index = response.index("## Exercises")
        exercises_end_index = response.index("### ", exercises_start_index)
        return (
            response[exercises_start_index:exercises_end_index].strip()
            == "## Exercises"
        )
    except Exception as e:
        return False


async def assert_no_duplicate_side_exercise(example: dict, prompt: str, response: str):
    """
    Verify that if an exercise is performed on both sides, it is stated only once with 'on each side'.
    """
    exercises_segment = response.split("## Exercises")[-1]
    return (
        "on each side" in exercises_segment
        and "left side" not in exercises_segment
        and "right side" not in exercises_segment
    )


async def assert_no_redundant_information_in_response(
    example: dict, prompt: str, response: str
):
    """
    Check if the response does not repeat the same information twice, especially within the exercise instructions.
    """
    exercises_content = response.split("## Exercises")[-1]
    unique_content = set()
    for line in exercises_content.split("\n"):
        if line.strip() and line.strip() in unique_content:
            return False
        unique_content.add(line.strip())
    return True


async def assert_follow_along_routine_description(
    example: dict, prompt: str, response: str
):
    """
    Use ask_llm to determine if the given text contains a follow-along routine that one can split into exercises.
    """
    question = (
        "Does the text contain a follow-along routine that can be split into exercises?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_completeness_and_correctness(
    example: dict, prompt: str, response: str
):
    """
    Use ask_llm to determine if the response demonstrates correctness and completeness by accurately representing the details of the routine.
    """
    question = "Does the response accurately represent the details of the given mobility routine, including the correct exercise names and descriptions?"
    return await ask_llm(prompt, response, question)


ALL_FUNCTIONS = [
    assert_as_markdown_format,
    assert_name_present_in_routine,
    assert_name_without_follow_along,
    assert_summary_suitability_and_presence,
    assert_equipment_section_availability,
    assert_post_exercise_repetitions,
    assert_deduplication_side_exercises,
    assert_non_splitable_routine_check,
    assert_response_markdown_format,
    assert_routine_name_response_start,
    assert_absence_of_follow_along_in_name,
    assert_summary_section_target_correctness,
    assert_equipment_section_existence,
    assert_exercises_with_time_or_reps_post_name,
    assert_no_repeat_exercises_both_sides,
    assert_no_text_below_exercise_list,
    assert_alternative_for_no_routine,
    assert_completeness_and_correctness_of_response,
    assert_markdown_format,
    assert_presence_headers_summary_equipment_exercises,
    assert_no_text_post_exercises_list,
    assert_descriptions_under_exercise_header,
    assert_no_duplicate_side_exercise,
    assert_no_redundant_information_in_response,
    assert_follow_along_routine_description,
    assert_response_completeness_and_correctness,
]
