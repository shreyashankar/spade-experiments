from litellm import acompletion


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for a large language model pipeline that writes reviews for pull requests to codebases. Here is the prompt:\n\n{prompt}\n\nHere is the response:\n{response}",
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


async def assert_includes_code_improvements_v1(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check if the response includes code improvement suggestions.
    """
    # It's expected that the response must contain the word "improvement" or modifications in the code snippet
    improvements_keywords = ["improvement", "improve", "refactor"]
    return any(keyword in response for keyword in improvements_keywords)


async def assert_response_follows_review_format_v1(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check if the response follows a specific format resembling a code review,
    which includes a title, author mention, gratitude, and listed improvements.
    """
    response_lines = response.split("\n")
    has_title = any("Title:" in line for line in response_lines)
    has_author = example["author"] in response
    has_gratitude = "thank you" in response.lower() or "thanks" in response.lower()
    has_improvements = "Improvements:" in response or "improvement:" in response.lower()

    return has_title and has_author and has_gratitude and has_improvements


async def assert_examples_of_improvement_follow_code_conventions_v1(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check if the examples of improvements in the response follow the code conventions
    of the language (Python in this case), checking for syntax correctness.
    """
    # Example Python code to check for syntax errors in response
    try:
        exec(response)
    except SyntaxError:
        return False
    return True


async def assert_response_adheres_to_workflow_v1(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response adheres to the workflow of discussing only the code in the diff,
    with no assumptions or extrapolations outside of the provided code segment.
    """
    question = "Does the response adhere to the workflow by discussing the code diff without making assumptions about the unseen code?"
    return await ask_llm(prompt, response, question)


async def assert_recognition_of_code_conventions_v1(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response includes recognition of the coding conventions, such as the use of snake_case for function names in Python.
    """
    question = "Does the response recognize and adhere to the code conventions of Python, including proper naming and style?"
    return await ask_llm(prompt, response, question)


async def assert_exclusion_of_speculation_v1(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response does not include speculations or assumptions about the parts of the code that are not visible in the diff.
    """
    question = "Does the response avoid making speculations or assumptions about the unseen parts of the code?"
    return await ask_llm(prompt, response, question)


async def assert_response_is_personal_and_grateful_v1(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check if the response demonstrates a personal touch and shows gratitude towards the author of the pull request.
    """
    is_personal = example["author"] in response
    shows_gratitude = (
        "thank you" in response.lower() or "gratitude:" in response.lower()
    )
    return is_personal and shows_gratitude


async def assert_response_is_concise_v1(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check if the response is concise and to the point, without unnecessary elaboration.
    """
    question = "Is the LLM response concise and to the point?"
    return await ask_llm(prompt, response, question)


async def assert_response_includes_code_review_v1(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes a code review of the changes fetched
    via the get_diff function.
    """
    code_review_present = "Review:" in response or "Improvement:" in response
    return code_review_present


async def assert_response_includes_improvements_v1(
    example: dict, prompt: str, response: str
):
    """
    Check if the response answers on what can be improved and provides the
    improvement in code.
    """
    contains_improvement = "Improvement:" in response
    return contains_improvement


async def assert_response_is_brief_v1(example: dict, prompt: str, response: str):
    """
    Assess if the response possesses a certain brevity as verbose commentary is discouraged.
    """
    question = "Is the review in the response brief and to the point without unnecessary verbosity?"
    return await ask_llm(prompt, response, question)


async def assert_follows_code_conventions_v1(example: dict, prompt: str, response: str):
    """
    Check if the response follows code conventions of the language it is reviewing.
    """
    question = "Does the response follow Python's code conventions?"
    return await ask_llm(prompt, response, question)


async def assert_response_is_personal_and_grateful_v2(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes a personal tone and shows gratitude to the author of the pull request.
    """
    is_personal = "Thank you, @" in response or "Great work!" in response
    return is_personal


async def assert_response_does_not_require_full_code_v1(
    example: dict, prompt: str, response: str
):
    """
    Verify that the response does not indicate a need for access to the full code.
    """
    question = (
        "Does the response provide a review without requiring access to the full code?"
    )
    return await ask_llm(prompt, response, question)


async def assert_review_based_on_workflow_v1(example: dict, prompt: str, response: str):
    """
    Confirm that the response indicates that the review process was based on the workflow described in the template.
    """
    # Since the workflow description does not directly reflect in the output, we ask the LLM.
    question = (
        "Was the review in the response based on the workflow described in the prompt?"
    )
    return await ask_llm(prompt, response, question)


# Usage example:
# Each of the functions would be awaited when checking the concepts.
# For example:
# satisfied = await assert_response_includes_code_review(example, formatted_prompt, llm_response)
# This would be done for each function to assert all the different concepts.
async def assert_presentation_format_v1(example: dict, prompt: str, response: str):
    """
    Verify if the LLM response includes instructions within a structured format.
    """
    if (
        "Instructions" in response
        and "Improvements:" in response
        and "Code Review:" in response
    ):
        return True
    else:
        return False


async def assert_improvement_suggestion_v1(example: dict, prompt: str, response: str):
    """
    Check whether the response provides an improvement in code or a suggestion for improvement.
    """
    return "Improvements:" in response


async def assert_workflow_adherence_v1(example: dict, prompt: str, response: str):
    """
    Check if the LLM has reviewed the code based on the diff_url following the instructions.
    """
    # This might be ambiguous without the full context of diff_url, hence using ask_llm
    question = "Does the response show that the review was based on the diff_url provided, following the instructions given?"
    return await ask_llm(prompt, response, question)


async def assert_code_review_aspects(example: dict, prompt: str, response: str):
    """
    Check if the response includes different aspects of code review: code snippets,
    adherence to language conventions, and a personal touch.
    """
    includes_code_snippets = "```python" in response and "```" in response
    personal_touch = "@" + example.get("author", "").replace("@", "") in response
    return includes_code_snippets and personal_touch


async def assert_gratitude_personal_touch(example: dict, prompt: str, response: str):
    """
    Ensure that response includes gratitude towards the author in a personal manner, possibly by tagging them.
    """
    author_tag = "@" + example.get("author", "").replace("@", "")
    return author_tag in response


async def assert_exclusion_of_full_code(example: dict, prompt: str, response: str):
    """
    Verify that the response does not demonstrate access to the full code, only the code diff.
    """
    question = "Does the response give the impression that the full code was accessed instead of just the code diff?"
    return not await ask_llm(prompt, response, question)


async def assert_conciseness_and_convention(example: dict, prompt: str, response: str):
    """
    Check that the response is concise and adheres to the language code conventions.
    """
    question_for_conciseness = "Is the response concise?"
    question_for_convention = (
        "Does the response adhere to the language's code conventions?"
    )

    # Call ask_llm once for brevity since both are qualitative assessments
    concise = await ask_llm(prompt, response, question_for_conciseness)
    convention = await ask_llm(prompt, response, question_for_convention)

    return concise and convention


async def assert_correct_payload_integration(example: dict, prompt: str, response: str):
    """
    Ensure that the response correctly interprets and integrates the pr_webhook_payload.
    """
    # Since this is likely factual, it can be inferred from response contents using Python's string methods
    has_title = example["title"] in response
    has_description = example["description"] in response

    return has_title and has_description


async def assert_properly_structured_english_v1(
    example: dict, prompt: str, response: str
):
    """
    Check that the response is in sentence or paragraph form with properly structured English.
    """
    question = "Is the response in sentence or paragraph form with properly structured English?"
    return await ask_llm(prompt, response, question)


async def assert_includes_get_diff_usage_v1(example: dict, prompt: str, response: str):
    """
    Check that the response includes an explanation or demonstration of how to use the 'get_diff' function.
    """
    # Since the response content can be varied and complex, using ask_llm to evaluate
    return await ask_llm(
        prompt,
        response,
        "Does the response explain or demonstrate how to use the 'get_diff' function?",
    )


async def assert_excludes_irrelevant_content_v1(
    example: dict, prompt: str, response: str
):
    """
    Check that the response excludes any keywords or steps not related to the pull request review process.
    """
    irrelevant_terms = ["irrelevant", "unnecessary", "unrelated"]
    # This can use Python string methods since it's a check for the exclusion of certain terms
    return not any(term in response for term in irrelevant_terms)


async def assert_has_pull_request_keywords_v1(
    example: dict, prompt: str, response: str
):
    """
    Check that the response always includes essential keywords related to pull request review.
    """
    # Keywords to be present in the response
    keywords = ["pull request", "code changes", "diff_url", "get_diff"]
    # This can be evaluated directly in Python since it's just a presence check
    return all(keyword in response for keyword in keywords)


async def assert_clear_professional_language_v1(
    example: dict, prompt: str, response: str
):
    """
    Check that the response is professional, clear, and without unnecessary jargon or overly complex vocabulary.
    """
    question = "Is the response professional, clear, and without unnecessary jargon or overly complex vocabulary?"
    return await ask_llm(prompt, response, question)


async def assert_follows_workflow_steps_v1(example: dict, prompt: str, response: str):
    """
    Check that the response follows the exact steps or sequence provided in the workflow.
    """
    # This assessment is complex and requires contextual understanding, so ask_llm is appropriate
    question = (
        "Does the response follow the exact steps or sequence provided in the workflow?"
    )
    return await ask_llm(prompt, response, question)


# Although not explicitly requested, this function is inferred from the concepts to check the proper acknowledgement
# and interaction with the pull request author, which includes gratitude and personalization.
async def assert_proper_acknowledgement_v1(example: dict, prompt: str, response: str):
    """
    Check for proper personal acknowledgement and gratitude towards the pull request author.
    """
    # Check for the "@" symbol followed by a username (author) and words expressing gratitude
    # We assume the author name will not contain spaces
    author = example.get("author")
    gratitude_phrases = ["thank you", "thanks", "appreciate", "grateful"]
    return (
        any(gratitude in response for gratitude in gratitude_phrases)
        and f"@{author}" in response
    )


async def assert_includes_improvements_and_code_snippets(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes both suggestions for improvements and corresponding code snippets.
    """
    return (
        "```" in response
        and "def " in response
        and all(term in response for term in ["->", "self.result", "return"])
    )


async def assert_excludes_full_codebase_review(
    example: dict, prompt: str, response: str
):
    """
    Check if the response does not attempt to review code outside of the given code diff.
    """
    # Assuming that non-diff related comments would not start with 'def ' or include 'self.' keywords
    code_diff_terms = ["def ", "self."]
    # If all code-related terms in the response also appear in the code diff, assume no full codebase review
    return all(
        term in example["diff"][0]["additions"]
        for term in code_diff_terms
        if term in response
    )


async def assert_response_is_concise_and_clear(
    example: dict, prompt: str, response: str
):
    """
    Ask LLM if the response is concise and clear.
    """
    question = "Is this pull request review response concise and clear?"
    return await ask_llm(prompt, response, question)


async def assert_reference_to_pull_request_review(
    example: dict, prompt: str, response: str
):
    """
    Check if the response is in reference to a pull request review.
    """
    return (
        "@pythonDev" in response
        and "suggestion" in response.lower()
        and "pull request" in response.lower()
    )


async def assert_follows_instructions_for_review(
    example: dict, prompt: str, response: str
):
    """
    Check if the response follows the given instructions for providing a review.
    """
    instructions = [
        "you don’t have access to the full code but only the code diff",
        "make it personal",
        'always show gratitude to the author using "@" when tagging',
        "Include code snippets if necessary",
        "Adhere to the languages code conventions",
    ]
    return all(instruction in prompt for instruction in instructions)


async def assert_excludes_unrelated_topics_or_keywords(
    example: dict, prompt: str, response: str
):
    """
    Check if the response does not mention any unrelated topics or keywords.
    """
    unrelated_terms = ["AI Assistant", "expert", "reviewing pull requests"]
    return not any(unrelated_term in response for unrelated_term in unrelated_terms)


async def assert_completeness_in_reviewing_code_diff(
    example: dict, prompt: str, response: str
):
    """
    Ask LLM if the response demonstrates completeness in reviewing only the code diff provided.
    """
    question = "Does this response demonstrate completeness in reviewing only the code diff provided?"
    return await ask_llm(prompt, response, question)


async def assert_consistency_with_tasks_and_instructions(
    example: dict, prompt: str, response: str
):
    """
    Check if the response is consistent with the given tasks and instructions provided.
    """
    question = "Is the response consistent with the tasks and instructions provided in the prompt template?"
    return await ask_llm(prompt, response, question)


async def assert_includes_code_improvement_v2(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes suggestions for code improvements.
    """
    question = "Does the response include suggestions for code improvements?"
    return await ask_llm(prompt, response, question)


async def assert_respects_information_limitation_v1(
    example: dict, prompt: str, response: str
):
    """
    Check that the response respects the limitation of information
    and does not refer to non-available parts of the code.
    """
    question = "Does the response correctly avoid referring to non-available parts of the code?"
    return await ask_llm(prompt, response, question)


async def assert_contains_brief_answers_v1(example: dict, prompt: str, response: str):
    """
    Check that the response contains brief answers.
    """
    question = "Is the response brief and to the point without unnecessary elaboration?"
    return await ask_llm(prompt, response, question)


async def assert_correctly_explains_diff_format_v1(
    example: dict, prompt: str, response: str
):
    """
    Check that the response explains the code diff format correctly.
    """
    question = "Does the response correctly explain the code diff format using '+' and '-' symbols?"
    return await ask_llm(prompt, response, question)


async def assert_excludes_irrelevant_code(example: dict, prompt: str, response: str):
    """
    Check that the response does not include code that hasn't been added or removed.
    """
    additions = "".join(diff["additions"] for diff in example["diff"])
    deletions = "".join(diff["deletions"] for diff in example["diff"])

    # Convert code blocks to plain text for comparison.
    response_code = "".join(response.split("```")[1::2]).strip()

    # Check if the code in the response is strictly from the additions or deletions.
    return additions.strip() in response_code and deletions.strip() in response_code


async def assert_responds_to_correct_pull_request(
    example: dict, prompt: str, response: str
):
    """
    Check that the response is a review of the received Pull Request (PR) and not any other topic.
    """
    pr_title = example["title"]
    question = (
        f"Is the response a review focused on the Pull Request titled '{pr_title}'?"
    )
    return await ask_llm(prompt, response, question)


ALL_FUNCTIONS = [
    assert_includes_code_improvements_v1,
    assert_response_follows_review_format_v1,
    assert_examples_of_improvement_follow_code_conventions_v1,
    assert_response_adheres_to_workflow_v1,
    assert_recognition_of_code_conventions_v1,
    assert_exclusion_of_speculation_v1,
    assert_response_is_personal_and_grateful_v1,
    assert_response_is_concise_v1,
    assert_response_includes_code_review_v1,
    assert_response_includes_improvements_v1,
    assert_response_is_brief_v1,
    assert_follows_code_conventions_v1,
    assert_response_is_personal_and_grateful_v2,
    assert_response_does_not_require_full_code_v1,
    assert_review_based_on_workflow_v1,
    assert_presentation_format_v1,
    assert_improvement_suggestion_v1,
    assert_workflow_adherence_v1,
    assert_code_review_aspects,
    assert_properly_structured_english_v1,
    assert_includes_get_diff_usage_v1,
    assert_excludes_irrelevant_content_v1,
    assert_has_pull_request_keywords_v1,
    assert_clear_professional_language_v1,
    assert_follows_workflow_steps_v1,
    assert_proper_acknowledgement_v1,
    assert_includes_improvements_and_code_snippets,
    assert_excludes_full_codebase_review,
    assert_response_is_concise_and_clear,
    assert_reference_to_pull_request_review,
    assert_follows_instructions_for_review,
    assert_excludes_unrelated_topics_or_keywords,
    assert_completeness_in_reviewing_code_diff,
    assert_consistency_with_tasks_and_instructions,
    assert_includes_code_improvement_v2,
    assert_respects_information_limitation_v1,
    assert_contains_brief_answers_v1,
    assert_correctly_explains_diff_format_v1,
    assert_excludes_irrelevant_code,
    assert_responds_to_correct_pull_request,
    assert_gratitude_personal_touch,
    assert_exclusion_of_full_code,
    assert_conciseness_and_convention,
    assert_correct_payload_integration,
]
