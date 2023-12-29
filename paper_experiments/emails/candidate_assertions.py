from litellm import acompletion


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for a large language model pipeline that writes SaaS User On-Boarding Emails following the Pain-Agitate-Solution strategy. Here is the prompt:\n\n{prompt}\n\nHere is the response:\n{response}",
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


async def assert_structured_format_followed_1(
    example: dict, prompt: str, response: str
) -> bool:
    required_headers = [
        "Subject Line",
        "Body",
        "Pain",
        "Agitate",
        "Solution",
        "Sub-Closing",
        "Closing",
        "Salutation Format",
    ]
    return all(header in response for header in required_headers)


async def assert_workflow_followed_2(example: dict, prompt: str, response: str) -> bool:
    question = "Does the response include an introduction of a problem, elaboration on the problem, a presented solution, encouragement for the user, and a salutation?"
    return await ask_llm(prompt, response, question)


async def assert_placeholder_elements_included_3(
    example: dict, prompt: str, response: str
) -> bool:
    return (f"{example['topic']}" in response) and (f"{example['context']}" in response)


async def assert_proper_ending_with_cta_4(
    example: dict, prompt: str, response: str
) -> bool:
    question = "Does the response end with a call to action, a sub-closing, a closing, and a formal salutation?"
    return await ask_llm(prompt, response, question)


async def assert_word_count_within_limit_5(
    example: dict, prompt: str, response: str
) -> bool:
    max_word_count = example["word_count"]
    word_count_response = len(response.split())
    return word_count_response <= max_word_count


async def assert_curiosity_and_action_driven_6(
    example: dict, prompt: str, response: str
) -> bool:
    question = "Does the response foster curiosity and drive action?"
    return await ask_llm(prompt, response, question)


async def assert_user_onboarding_practices_included_7(
    example: dict, prompt: str, response: str
) -> bool:
    question = "Does the response utilize user onboarding best practices to encourage users to return to the platform and discover the value in the service?"
    return await ask_llm(prompt, response, question)


async def assert_encouragement_to_contact_company_8(
    example: dict, prompt: str, response: str
) -> bool:
    contact_phrases = [
        "reach out",
        "don't hesitate to contact",
        "looking forward to hearing from you",
        "if you have any questions",
        "need help getting started",
    ]
    return any(phrase in response for phrase in contact_phrases)


async def assert_has_specific_headers_9(example: dict, prompt: str, response: str):
    headers = [
        "Pain:",
        "Agitate:",
        "Solution:",
        "Sub-Closing:",
        "Closing:",
        "Call to Action:",
    ]
    return all(header in response for header in headers)


async def assert_includes_placeholders_10(example: dict, prompt: str, response: str):
    return "[topic]" in response and "[context]" in response


async def assert_workflow_follows_ordered_steps_11(
    example: dict, prompt: str, response: str
):
    workflow_steps = ["Pain", "Agitate", "Solution", "Sub-Closing", "Closing"]
    workflow_indices = [response.find(step) for step in workflow_steps]
    return all(
        workflow_indices[i] < workflow_indices[i + 1]
        for i in range(len(workflow_indices) - 1)
    )


async def assert_minimum_core_components_present_12(
    example: dict, prompt: str, response: str
):
    sections = [
        "Pain:",
        "Agitate:",
        "Solution:",
        "Sub-Closing:",
        "Aim",
        "Closing:",
        "Salutation Format",
    ]
    return len([section for section in sections if section in response]) >= 7


async def assert_includes_call_to_action_13(example: dict, prompt: str, response: str):
    return "Call to Action:" in response


async def assert_excludes_forbidden_words_14(example: dict, prompt: str, response: str):
    forbidden_words = ["Feature", "Religion"]
    return not any(word in response for word in forbidden_words)


async def assert_response_has_adequate_length_and_tone_15(
    example: dict, prompt: str, response: str
):
    question = "Is the response of adequate length to discuss each section and does it maintain a professional tone?"
    return await ask_llm(prompt, response, question)


async def assert_consistent_correct_placeholder_usage_16(
    example: dict, prompt: str, response: str
):
    placeholders_used_correctly = "[topic]" in response and "[context]" in response
    if placeholders_used_correctly:
        topic_placeholder_index = response.find("[topic]")
        context_placeholder_index = response.find("[context]")
        return topic_placeholder_index < context_placeholder_index
    return False


async def assert_has_email_structure_17(example: dict, prompt: str, response: str):
    is_subject_present = "Subject Line:" in response
    is_body_present = "Body:" in response
    return is_subject_present and is_body_present


async def assert_demonstrates_common_problem_18(
    example: dict, prompt: str, response: str
):
    question = "Does the response demonstrate a common, real-world problem or challenge related to the topic?"
    return await ask_llm(prompt, response, question)


async def assert_follows_pain_agitate_solution_19(
    example: dict, prompt: str, response: str
):
    question = "Does this email follow the Pain-Agitate-Solution strategy effectively?"
    return await ask_llm(prompt, response, question)


async def assert_correct_length_20(example: dict, prompt: str, response: str):
    word_count = example["word_count"]
    response_word_count = len(response.split())
    return response_word_count <= int(word_count)


async def assert_includes_required_components_21(
    example: dict, prompt: str, response: str
):
    question = "Does the email include pain, agitate, and solution components and relate them to the context?"
    return await ask_llm(prompt, response, question)


async def assert_excludes_new_user_onboarding_22(
    example: dict, prompt: str, response: str
):
    questioning_new_user = "Is the email targeting new users to onboard them, or does it focus on existing users?"
    return not await ask_llm(prompt, response, questioning_new_user)


async def assert_compelling_subject_line_23(example: dict, prompt: str, response: str):
    question = "Is the subject line of the email compelling and aligned with the email content?"
    return await ask_llm(prompt, response, question)


async def assert_consistent_professional_tone_24(
    example: dict, prompt: str, response: str
):
    question = "Does the tone of the email maintain the consistent role of a 'professional marketing copywriter'?"
    return await ask_llm(prompt, response, question)


ALL_FUNCTIONS = [
    assert_structured_format_followed_1,
    assert_workflow_followed_2,
    assert_placeholder_elements_included_3,
    assert_proper_ending_with_cta_4,
    assert_word_count_within_limit_5,
    assert_curiosity_and_action_driven_6,
    assert_user_onboarding_practices_included_7,
    assert_encouragement_to_contact_company_8,
    assert_has_specific_headers_9,
    assert_includes_placeholders_10,
    assert_workflow_follows_ordered_steps_11,
    assert_minimum_core_components_present_12,
    assert_includes_call_to_action_13,
    assert_excludes_forbidden_words_14,
    assert_response_has_adequate_length_and_tone_15,
    assert_consistent_correct_placeholder_usage_16,
    assert_has_email_structure_17,
    assert_demonstrates_common_problem_18,
    assert_follows_pain_agitate_solution_19,
    assert_correct_length_20,
    assert_includes_required_components_21,
    assert_excludes_new_user_onboarding_22,
    assert_compelling_subject_line_23,
    assert_consistent_professional_tone_24,
]
