from litellm import acompletion


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are a helpful assistant. Here is the prompt:\n\n{prompt}\n\nHere is the response:\n{response}",
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


async def assert_response_bullet_list_format(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if the response is formatted as a bullet list.
    """
    return response.strip().startswith("-") and all(
        line.strip().startswith("-") for line in response.splitlines()
    )


async def assert_response_bullet_key_value_format(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if each bullet in the response is formatted with `key: value.`
    """
    bullets = [line.strip() for line in response.splitlines()]
    return all(":" in bullet and bullet.endswith(".") for bullet in bullets)


async def assert_response_has_required_keys_with_na(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if the response contains the required keys with `N/A` for missing values.
    """
    required_keys = [
        "chief complaint:",
        "history of present illness:",
        "physical examination:",
        "symptoms experienced by the patient:",
        "new medications prescribed or changed, including dosages:",
        "follow-up instructions:",
    ]
    return all(key in response for key in required_keys) or "N/A" in response


async def assert_no_pii_in_response(example: dict, prompt: str, response: str) -> bool:
    """
    Checks if the response does not contain any personal identifiable information (PII).
    """
    pii_keywords = ["name", "age", "gender", "ID"]
    return not any(pii in response for pii in pii_keywords)


async def assert_use_of_the_patient_instead_of_name(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if 'the patient' is used instead of personal names in the response.
    """
    patient_name = example["document"].split("Patient Name")[1]

    # Get everything up to newline
    patient_name = patient_name.split("\n")[0].strip().lower()

    return patient_name not in response and "the patient" in response.lower()


async def assert_inclusion_of_specific_keys(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if the response contains extracted values for specific keys: Chief complaint, History of present illness,
    Physical examination, Symptoms, New medications, and Follow-up instructions.
    """
    required_keys = [
        "chief complaint:",
        "history of present illness:",
        "physical examination:",
        "symptoms experienced by the patient:",
        "new medications prescribed or changed, including dosages:",
        "follow-up instructions:",
    ]
    return all(key in response for key in required_keys)


async def assert_response_under_word_limit(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if the response is under 100 words.
    """
    word_count = len(response.split())
    return word_count <= 100


async def assert_adheres_to_workflow(example: dict, prompt: str, response: str) -> bool:
    """
    Checks if the LLM response adheres to the workflow of extracting insights from medical records that include a
    medical note and a doctor-patient dialogue.
    """
    workflow_description = "extracting insights from medical records that include a medical note and a doctor-patient dialogue"
    return workflow_description in prompt


async def assert_adherence_to_extraction_without_demos(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if the LLM response adheres to the instructions of extracting values without providing any examples or demonstrations.
    """
    question = "Does the response properly extract values without providing examples or demonstrations?"
    return await ask_llm(prompt, response, question)


async def assert_adherence_to_formal_clinical_tone(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if the response maintains a formal and clinical tone appropriate for medical record analysis.
    """
    question = "Does the response maintain a formal and clinical tone appropriate for medical record analysis?"
    return await ask_llm(prompt, response, question)


async def assert_bullet_list_correct_format(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Verifies if the response is formatted as a bullet list with each bullet point following the
    format `key: value.`.
    """
    bullets = [line.strip() for line in response.split("\n") if line.strip()]
    return all(
        bullet.startswith("- ") and ": " in bullet and bullet.endswith(".")
        for bullet in bullets
    )


async def assert_headers_correspondence_to_prompt(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if each bullet in the response has headers corresponding to the data points to be
    extracted as per the prompt.
    """
    required_headers = [
        "chief complaint:",
        "history of present illness:",
        "physical examination:",
        "symptoms experienced by the patient:",
        "new medications prescribed or changed, including dosages:",
        "follow-up instructions:",
    ]
    response_lines = response.lower().split("\n")
    return all(
        any(header in line for line in response_lines) for header in required_headers
    )


async def assert_placeholder_usage_correctly(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Confirms that placeholders are used for the patient's name in the response.
    """
    patient_name = example["document"].split("Patient Name")[1]

    # Get everything up to newline
    patient_name = patient_name.split("\n")[0].strip().lower()
    return patient_name not in response.lower()


async def assert_na_usage_for_missing_data(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if 'N/A' is used appropriately for data points that are not available.
    """
    response = response.lower()
    is_meds_present = "new medications prescribed or changed" in response
    is_followup_present = "follow-up instructions" in response

    meds_index = response.find("new medications prescribed or changed")
    followup_index = response.find("follow-up instructions")

    check_meds = (
        "n/a" in response[meds_index : response.find("\n", meds_index)]
        if not is_meds_present
        else True
    )
    check_followup = (
        "n/a" in response[followup_index : response.find("\n", followup_index)]
        if not is_followup_present
        else True
    )

    return check_meds and check_followup


async def assert_adjusted_word_count(example: dict, prompt: str, response: str) -> bool:
    """
    Verifies if the response is around 150 words, to ensure summarization when necessary.
    """
    word_count = len(response.split())
    return 100 <= word_count <= 200  # Little flexibility given the approximation


async def assert_no_pii_present(example: dict, prompt: str, response: str) -> bool:
    """
    Verifies that no personal identifiable information (PII) is present in the response.
    """
    return await ask_llm(
        prompt,
        response,
        f"In the response, is there any personal identifiable information (PII) such as name, age, gender, or ID?",
    )


async def assert_neutral_clinical_tone(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Determines if the tone of the response is neutral, professional, and objective.
    """
    question = "Is the tone of the response neutral, professional, and objective?"
    return await ask_llm(prompt, response, question)


async def assert_explicit_na_if_no_meds_changed(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if the response explicitly states 'N/A' for dosages when no new medications are prescribed or changed.
    """
    if "new medications prescribed or changed" not in response.lower():
        return "n/a" in response.lower()
    return True  # Should return true if the segment is present, to not be consequential


async def assert_explicit_na_if_no_follow_up(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Checks if the response explicitly states 'N/A' for follow-up instructions when no follow-up details are provided.
    """
    if "follow-up instructions" not in response.lower():
        return "n/a" in response.lower()
    return True  # Should return true if the segment is present, to not be consequential


async def assert_correct_header_order_maintained(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Verifies the response adheres to the header order as requested in the prompt.
    """
    required_order = [
        "chief complaint",
        "history of present illness",
        "physical examination",
        "symptoms experienced by the patient",
        "new medications prescribed or changed, including dosages",
        "follow-up instructions",
    ]
    headers_found = []
    for line in response.lower().split("\n"):
        for header in required_order:
            if header in line:
                headers_found.append(header)
                break
    return headers_found == required_order


async def assert_data_points_filled_completely(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Ensures that each data point extracted is complete, and no relevant details are missing.
    """
    question = "Are all relevant details included for each section listed, with nothing significant omitted?"
    return await ask_llm(prompt, response, question)


# Include a variable called ALL_FUNCTIONS with a list of function names
ALL_FUNCTIONS = [
    assert_response_bullet_list_format,
    assert_response_bullet_key_value_format,
    assert_response_has_required_keys_with_na,
    assert_no_pii_in_response,
    assert_use_of_the_patient_instead_of_name,
    assert_inclusion_of_specific_keys,
    assert_response_under_word_limit,
    assert_adheres_to_workflow,
    assert_adherence_to_extraction_without_demos,
    assert_adherence_to_formal_clinical_tone,
    assert_bullet_list_correct_format,
    assert_headers_correspondence_to_prompt,
    assert_placeholder_usage_correctly,
    assert_na_usage_for_missing_data,
    assert_adjusted_word_count,
    assert_no_pii_present,
    assert_neutral_clinical_tone,
    assert_explicit_na_if_no_meds_changed,
    assert_explicit_na_if_no_follow_up,
    assert_correct_header_order_maintained,
    assert_data_points_filled_completely,
]
