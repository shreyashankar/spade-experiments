from litellm import acompletion


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for a large language model chatbot that writes summaries of video transcripts:\n\n{prompt}\n\nHere is the response:\n{response}",
            "role": "system",
        },
        {
            "content": f"{question}\nOnly answer yes or no.",
            "role": "user",
        },
    ]

    response = await acompletion(
        model="azure/gpt-35-turbo-16k",
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


async def assert_response_begins_with_outline(
    example: dict, prompt: str, response: str
):
    """
    Check if the response begins with an outline planning the execution of the summary.
    """
    return response.strip().startswith("Title:") or response.strip().startswith(
        "Outline:"
    )


async def assert_response_includes_context(example: dict, prompt: str, response: str):
    """
    Check if the response includes context at the beginning with the title, speaker/author,
    and overarching theme.
    """
    return "Title:" in response and "Speaker:" in response and "Subject:" in response


async def assert_response_lists_key_points_with_bullets(
    example: dict, prompt: str, response: str
):
    """
    Check if the response lists 3-5 key points using bullet points.
    """
    key_points_start = response.find("Key Points:")
    key_points_end = response.find("Central Thesis:", key_points_start)
    key_points_text = response[key_points_start:key_points_end].strip()

    if key_points_start == -1 or key_points_end == -1:
        return False

    bullet_points = key_points_text.count(
        "-"
    )  # Assume bullet points are marked with hyphens
    return 3 <= bullet_points <= 5


async def assert_response_articulates_central_thesis(
    example: dict, prompt: str, response: str
):
    """
    Check whether the response articulates the central thesis of the discussion.
    """
    return "Central Thesis:" in response


async def assert_response_includes_significant_details(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes significant details that support the key points.
    """
    return "Significant Details:" in response


async def assert_response_summarizes_implications(
    example: dict, prompt: str, response: str
):
    """
    Check if the response summarizes the overall implications and actionable insights.
    """
    return "Conclusions and Takeaways:" in response


async def assert_response_mentions_personal_insights(
    example: dict, prompt: str, response: str
):
    """
    Check if the response briefly mentions personal insights or criticisms and is clearly
    marked as not part of the original content.
    """
    return "Personal Insights or Criticisms:" in response


async def assert_response_word_count_within_limits(
    example: dict, prompt: str, response: str
):
    """
    Check if the response is within the 200-400 word count range.
    """
    word_count = len(response.split())
    return 200 <= word_count <= 400


async def assert_response_excludes_text_outside_content(
    example: dict, prompt: str, response: str
):
    """
    Check if the response excludes any text outside the provided original content.
    """
    outside_text = example["text"]
    return outside_text not in response


async def assert_response_includes_relevant_dates_and_settings(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes relevant dates and settings if applicable.
    """
    # This is a bit trickier since the presence of dates and settings can highly vary.
    # Let's ask the LLM to validate whether the response correctly includes or excludes
    # dates and settings based on context from the example.
    question = "Does the response include the relevant dates and settings, if they were applicable?"
    return await ask_llm(prompt, response, question)


async def assert_response_clear_distinction_between_content_and_insights(
    example: dict, prompt: str, response: str
):
    """
    Check if the response upholds a clear distinction between original content and personal
    insights or criticisms.
    """
    insights_section_start = response.find("Personal Insights or Criticisms:")
    if insights_section_start == -1:
        return False
    insights_section = response[insights_section_start:]
    return (
        "*Be very clear this is not a part of the original content.*"
        in insights_section
    )


async def assert_response_structure_for_usefulness_and_clarity(
    example: dict, prompt: str, response: str
):
    """
    Check if the response ensures that the requested structure is followed for usefulness and clarity.
    """
    required_sections = [
        "Title:",
        "Speaker:",
        "Subject:",
        "Key Points:",
        "Central Thesis:",
        "Significant Details:",
        "Conclusions and Takeaways:",
        "Personal Insights or Criticisms:",
    ]
    for section in required_sections:
        if section not in response:
            return False
    return True


async def assert_contains_outline(example: dict, prompt: str, response: str) -> bool:
    """
    Check that the response begins with an outline planning the summary's execution.
    """
    return response.lower().startswith("outline:")


async def assert_contains_significant_details(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response contains significant details using data, anecdotes, or examples.
    """
    return "significant details:" in response.lower()


async def assert_contains_conclusions_takeaways(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response includes a section on conclusions and takeaways.
    """
    return "conclusions and takeaways:" in response.lower()


async def assert_summary_within_word_count(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response is a summary within the 200-400 word count range.
    """
    word_count = len(response.split())
    return 200 <= word_count <= 400


async def assert_response_ends_with_word_count(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response ends with the word count being mentioned.
    """
    return response.strip().endswith("word count: {}".format(len(response.split())))


async def assert_response_proper_summary_structure(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response accomplishes the parts of the task successfully according to the instructed structure.
    """
    question = "Does the response accomplish all parts of the task successfully according to the instructed structure, while providing a useful, clear, and succinct summary that captures the essence of the discussion?"
    return await ask_llm(prompt, response, question)


async def assert_response_maintains_essence_and_brevity(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response maintains the essence of the discussion and creates a succinct summary.
    """
    question = "Does the response maintain the essence of the discussion and create a succinct summary?"
    return await ask_llm(prompt, response, question)


async def assert_response_free_from_personal_opinions(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response is free from opinions, as the section for personal insights or criticisms is removed.
    """
    question = "Does the response include any personal insights or criticisms, contradicting the requirement to be free from opinions?"
    # We expect a False response here, so we negate the response from the LLM.
    return not await ask_llm(prompt, response, question)


async def assert_response_clearly_a_summary(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response clearly indicates it is a summary and not the original content.
    """
    question = "Is it clear from the response that it is a summary and not the original content?"
    return await ask_llm(prompt, response, question)


async def assert_context_includes_title_author_subject(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes context information: title, speaker/author, and subject/theme.
    """
    return (
        ("Title:" in response)
        and ("Author:" in response)
        and ("Theme:" in response or "Subject:" in response)
    )


async def assert_inclusion_of_date_and_setting_if_applicable(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes date and setting when applicable. Since applicability is subjective,
    leverage LLM if the title implies a specific event or setting.
    """
    if "lecture" in response or "discussion" in response:
        question = "Does the summary unnecessarily omit any applicable date or setting details?"
        return not await ask_llm(prompt, response, question)
    return True  # Assume correct if no evident relation to specific events or settings.


async def assert_minimum_key_points_listed(example: dict, prompt: str, response: str):
    """
    Check that the response lists at least 3 key points.
    """
    return (
        len(
            [
                point
                for point in response.split("-")
                if "Key Points:" in point or "point:" in point
            ]
        )
        >= 3
    )


async def assert_brief_and_succinct_preservation_of_essence(
    example: dict, prompt: str, response: str
):
    """
    Check that the response is brief and succinct while preserving the essence of the discussion.
    """
    question = "Is the summary brief and succinct while preserving the essence of the discussion?"
    return await ask_llm(prompt, response, question)


async def assert_no_excessive_details(example: dict, prompt: str, response: str):
    """
    Assert that the summary does not dwell into excessive details and maintains a focus on key points.
    """
    question = (
        "Does the summary include excessive details beyond the necessary key points?"
    )
    return not await ask_llm(prompt, response, question)


async def assert_accuracy_in_summary_comprehension(
    example: dict, prompt: str, response: str
):
    """
    Check that the response demonstrates comprehension skills by summarizing the text accurately.
    """
    question = "Does this summary accurately represent the provided text, reflecting comprehension of the material?"
    return await ask_llm(prompt, response, question)


async def assert_correct_summary_structure(example: dict, prompt: str, response: str):
    """
    Check that the response follows the outlined structure: context first, followed by key points, etc.
    """
    structure_parts = [
        "Title:",
        "Author:",
        "Theme:",
        "Key Points:",
        "Central Thesis:",
        "Significant Details:",
        "Conclusions and Takeaways:",
    ]
    structure_order = [
        response.index(part) for part in structure_parts if part in response
    ]

    # A correctly structured response will have indices in increasing order
    return all(
        earlier < later for earlier, later in zip(structure_order, structure_order[1:])
    )


async def assert_no_unnecessary_urgency(example: dict, prompt: str, response: str):
    """
    Ensure that the response does not reflect an unnecessary sense of urgency that could compromise information quality.
    """
    question = "Does the summary reflect an unnecessary urgency that compromises the information quality?"
    return not await ask_llm(prompt, response, question)


async def assert_clarity_and_format_of_structure(
    example: dict, prompt: str, response: str
):
    """
    Verify that the response is clearly structured with the specified formatting and arrangement of information.
    """
    question = (
        "Is the summary clearly structured and formatted as directed in the prompt?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_includes_date_and_setting_if_applicable(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes the date and setting of the discussion or lecture when applicable.
    """
    if "date" in example and "setting" in example:
        return (
            f"Date: {example['date']}" in response
            and f"Setting: {example['setting']}" in response
        )
    return True  # Assume it's fine if not applicable.


async def assert_glossary_in_response_1(example: dict, prompt: str, response: str):
    """
    Check if the response contains a glossary of important terms with their definitions and context of use.
    """
    glossary_heading = "Glossary of Important Terms:"
    return (
        glossary_heading in response
    )  # For more detail, regex or further string checks could be used


async def assert_minimum_number_of_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check if there are at least 3-5 key points listed in the response.
    """
    count = response.count("-")  # Assuming bullet points start with "-"
    return 3 <= count <= 5


async def assert_key_points_are_bullet_points(
    example: dict, prompt: str, response: str
):
    """
    Check if key points are formatted as bullet points for easy readability.
    """
    key_points_heading = "Key Points:"
    key_points_index = response.find(key_points_heading)
    return (
        response[key_points_index:].count("\n- ") >= 3
    )  # Looking for newline followed by "- "


async def assert_significant_details_present(example: dict, prompt: str, response: str):
    """
    Check if the response includes significant details related to the key points.
    """
    significant_details_heading = "Significant Details:"
    return significant_details_heading in response


async def assert_conclusions_and_takeaways(example: dict, prompt: str, response: str):
    """
    Check if the conclusion and takeaways of the talk are summarized in the response.
    """
    conclusions_heading = "Conclusions and Takeaways:"
    return conclusions_heading in response


async def assert_response_format_markdown(example: dict, prompt: str, response: str):
    """
    Check if the response format is written in Markdown.
    """
    # This check assumes that Markdown response must contain specific Markdown elements like headings or lists
    return response.count("# ") > 0 or response.count("- ") > 0


async def assert_no_word_count_requirement(example: dict, prompt: str, response: str):
    """
    Check if there's no specific word count requirement mentioned in the response.
    """
    question = "Is there a specific word count mentioned in the response?"
    return not await ask_llm(prompt, response, question)


async def assert_task_execution(example: dict, prompt: str, response: str):
    """
    Check if the response faithfully executes the given task to create a summary of the provided text.
    """
    question = "Does the response accurately represent a summary of the text provided in the task?"
    return await ask_llm(prompt, response, question)


async def assert_has_context(example: dict, prompt: str, response: str) -> bool:
    """
    Check that the LLM response begins with context including the title, speaker or author,
    and overarching subject or theme of the discussion or lecture.
    """
    return response.lower().startswith("context:")


async def assert_has_glossary_of_terms(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response contains a glossary of important terms with definitions.
    """
    return "glossary of important terms:" in response.lower()


async def assert_has_central_thesis(example: dict, prompt: str, response: str) -> bool:
    """
    Check that the response clearly articulates the central thesis or main argument of the discussion.
    """
    return "central thesis:" in response.lower()


async def assert_has_key_points(example: dict, prompt: str, response: str) -> bool:
    """
    Check that the response lists at least 3-5 key points using bullet points.
    """
    key_points_index = response.lower().find("key points:")
    list_start_index = response[key_points_index:].find("-")
    bullets = response[key_points_index + list_start_index :].count("-")
    return bullets >= 3 and bullets <= 5


async def assert_has_conclusions_and_takeaways(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response includes conclusions and takeaways summarizing the overall implications
    of the discussion and any actionable insights.
    """
    return "conclusions and takeaways:" in response.lower()


async def assert_is_clear_succinct_and_essence_captured(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the summary is clear, succinct, and captures the essence of the discussion using `ask_llm`.
    """
    question = "Is the summary clear, succinct, and does it capture the essence of the discussion?"
    return await ask_llm(prompt, response, question)


async def assert_is_formatted_in_markdown(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response is formatted in Markdown.
    """
    return response.strip().startswith("#")


async def assert_no_unnecessary_details(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response does not include any unnecessary details that do not aid in understanding
    the main points using `ask_llm`.
    """
    question = "Does the summary include any unnecessary details that do not aid in understanding the main points?"
    # We ask the LLM the opposite question; if it returns True, we include unnecessary details, so the result should be False.
    return not await ask_llm(prompt, response, question)


async def assert_includes_significant_details(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response includes any additional details crucial for understanding the key points using `ask_llm`.
    """
    question = "Does the summary exclude crucial details that are significant for understanding the key points?"
    # We ask the LLM the opposite question; if it returns True, we are missing crucial details, so the result should be False.
    return not await ask_llm(prompt, response, question)


async def assert_response_begins_with_context_2(
    example: dict, prompt: str, response: str
):
    """
    Check that the response begins with context, including the title, the speaker or author,
    and the overarching subject or theme.
    """
    return (
        response.startswith("**Title:**")
        and ("**Author:**" in response)
        and ("**Subject:**" in response)
    )


async def assert_response_contains_central_thesis(
    example: dict, prompt: str, response: str
):
    """
    Check that the response contains a clearly articulated central thesis or main argument.
    """
    return "**Thesis:**" in response


async def assert_response_has_at_least_3_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check that the response lists at least 3 key points in a bulleted format.
    """
    try:
        key_points_section = response.split("**Key Points:**")[1]
        key_points = [
            point for point in key_points_section.split("\n") if point.startswith("-")
        ]
        return len(key_points) >= 3
    except:
        return False


async def assert_conclusions_and_takeaways_present(
    example: dict, prompt: str, response: str
):
    """
    Check that the response summarizes the overall implications of the discussion
    and any actionable insights in the conclusions and takeaways section.
    """
    return "**Conclusions and Takeaways:**" in response


async def assert_glossary_included(example: dict, prompt: str, response: str):
    """
    Check that the response includes a glossary of important terms with definitions
    and context usage if applicable.
    """
    return "**Glossary of Important Terms:**" in response


async def assert_correct_summary_headers(example: dict, prompt: str, response: str):
    """
    Check that each section of the summary has clearly defined Markdown headers.
    """
    headers = [
        "**Title:**",
        "**Author:**",
        "**Subject:**",
        "**Thesis:**",
        "**Key Points:**",
        "**Conclusions and Takeaways:**",
        "**Glossary of Important Terms:**",
    ]
    return all(header in response for header in headers)


async def assert_exclusion_of_irrelevant_information(
    example: dict, prompt: str, response: str
):
    """
    Use an LLM to determine if the response excludes any information not relevant to the summary,
    such as personal opinions or information outside of the original discussion.
    """
    question = "Does the response exclude any information not relevant to the summary, such as personal opinions or information outside the boundaries of the original discussion?"
    return await ask_llm(prompt, response, question)


async def assert_text_quotation_and_workflow_description(
    example: dict, prompt: str, response: str
):
    """
    Check that the text to summarize is quoted or delineated to separate it from the summary response.
    """
    return "BEGIN TEXT" in prompt and "END TEXT" in prompt


async def assert_comprehension_of_main_ideas_and_details(
    example: dict, prompt: str, response: str
):
    """
    Use an LLM to determine if the response exhibits high comprehension skills by distinguishing between main ideas and supporting details.
    """
    question = "Does the response exhibit high comprehension skills by distinguishing between main ideas and supporting details?"
    return await ask_llm(prompt, response, question)


async def assert_brevity_and_essence_maintenance(
    example: dict, prompt: str, response: str
):
    """
    Use an LLM to determine if the response is succinct while maintaining the essence of the discussion.
    """
    question = (
        "Is the response succinct while maintaining the essence of the discussion?"
    )
    return await ask_llm(prompt, response, question)


async def assert_clear_central_thesis(example: dict, prompt: str, response: str):
    """
    Check that the response clearly articulates the central thesis or main argument
    of the discussion.
    """
    question = "Does the response clearly articulate the central thesis or main argument of the discussion?"
    return await ask_llm(prompt, response, question)


async def assert_bullet_points_key_points(example: dict, prompt: str, response: str):
    """
    Check that the key points in the response are enumerated using bullet points.
    """
    return "- " in response


async def assert_minimum_key_points_count(example: dict, prompt: str, response: str):
    """
    Check that the response includes at least 3-5 key points.
    """
    bullet_points = response.count("- ")
    return 3 <= bullet_points <= 5


async def assert_conclusion_and_actionable_insights(
    example: dict, prompt: str, response: str
):
    """
    Check that the response summarizes the overall implications and actionable insights,
    indicating a conclusion.
    """
    question = (
        "Does the response summarize the overall implications and actionable insights, "
        "indicating a conclusion?"
    )
    return await ask_llm(prompt, response, question)


async def assert_consistent_detail_level_for_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check that the response maintains the same level of detail when listing the key points,
    without introducing new line indents or headers.
    """
    question = (
        "Does the response maintain the same level of detail throughout the key points "
        "without any new line indents or headers, beyond the bullet points?"
    )
    return await ask_llm(prompt, response, question)


async def assert_no_extra_headers_or_sections(
    example: dict, prompt: str, response: str
):
    """
    Check that the response does not introduce new headers or sections outside the provided template.
    """
    prompted_headers = [
        "Context",
        "Central Thesis",
        "Key Points",
        "Conclusions and Takeaways",
        "Glossary of Important Terms",
    ]
    response_headers = [
        line
        for line in response.splitlines()
        if line.startswith("**") and line.endswith("**")
    ]
    return all(header.strip("**") in prompted_headers for header in response_headers)


async def assert_rich_content_in_thesis_and_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check that the response contains detailed elaboration for the thesis and key points, implying richness in content.
    """
    question = (
        "Does the response contain detailed elaboration for the thesis and key points, "
        "implying richness in content?"
    )
    return await ask_llm(prompt, response, question)


async def assert_understanding_of_date_and_setting(
    example: dict, prompt: str, response: str
):
    """
    Check that the response reflects an understanding of the date and setting if applicable,
    suggesting completeness.
    """
    question = (
        "Does the response reflect an understanding of the date and setting when mentioned, "
        "suggesting completeness of the summary?"
    )
    return await ask_llm(prompt, response, question)


async def assert_grammatical_correctness(example: dict, prompt: str, response: str):
    """
    Check that the response avoids any grammatical or spelling errors, upholding correctness.
    """
    question = "Is the response free of grammatical or spelling errors?"
    return await ask_llm(prompt, response, question)


async def assert_markdown_format(example: dict, prompt: str, response: str):
    """
    Check that the response is formatted in Markdown.
    """
    is_markdown = response.startswith("**Context:**") and "- " in response
    return is_markdown


async def assert_context_includes_title_speaker_theme(
    example: dict, prompt: str, response: str
):
    """
    Check if the 'Context' section includes the title, speaker or author, subject or theme,
    with optional date and setting if applicable.
    """
    context_start = response.find("**Context:**")
    context_end = (
        response.find("**Central Thesis:**")
        if "**Central Thesis:**" in response
        else len(response)
    )
    context_section = response[context_start:context_end]
    includes_title = "Title:" in context_section
    includes_speaker = "Speaker:" in context_section or "Author:" in context_section
    includes_theme = "Theme:" in context_section
    return includes_title and includes_speaker and includes_theme


async def assert_central_thesis_articulated(example: dict, prompt: str, response: str):
    """
    Check if the response articulates the 'Central Thesis' or main argument.
    """
    thesis_start = response.find("**Central Thesis:**")
    thesis_end = (
        response.find("**Key Points:**")
        if "**Key Points:**" in response
        else len(response)
    )
    central_thesis_section = response[thesis_start:thesis_end]
    has_central_thesis = (
        len(central_thesis_section.split()) > 5
    )  # Arbitrarily assumes thesis should be longer than 5 words
    return has_central_thesis


async def assert_key_points_with_bullet_points(
    example: dict, prompt: str, response: str
):
    """
    Check that the 'Key Points' section includes at least 3 bullet points.
    """
    key_points_start = response.find("**Key Points:**")
    key_points_end = (
        response.find("**Conclusions and Takeaways:**")
        if "**Conclusions and Takeaways:**" in response
        else len(response)
    )
    key_points_section = response[key_points_start:key_points_end]
    bullet_points = key_points_section.count("- ")
    return bullet_points >= 3


async def assert_conclusions_and_takeaways_summary(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes a 'Conclusions and Takeaways' summary.
    """
    conclusions_start = response.find("**Conclusions and Takeaways:**")
    conclusions_end = (
        response.find("**Glossary of Important Terms:**")
        if "**Glossary of Important Terms:**" in response
        else len(response)
    )
    conclusions_section = response[conclusions_start:conclusions_end]
    has_conclusions = (
        len(conclusions_section.split()) > 5
    )  # Arbitrarily assumes conclusions should be longer than 5 words
    return has_conclusions


async def assert_glossary_with_definitions(example: dict, prompt: str, response: str):
    """
    Check if the response includes a 'Glossary of Important Terms' with definitions.
    """
    glossary_start = response.find("**Glossary of Important Terms:**")
    glossary_section = response[glossary_start:] if glossary_start != -1 else ""
    return "List" in glossary_section and ":" in glossary_section


async def assert_non_bias_and_journalistic_tone(
    example: dict, prompt: str, response: str
):
    """
    Check if the response maintains a non-bias and journalistic tone.
    """
    question = "Does the summary maintain a non-bias and journalistic tone?"
    return await ask_llm(prompt, response, question)


async def assert_distillation_and_brevity(example: dict, prompt: str, response: str):
    """
    Check if the summary distills the discussion into a concise format while maintaining the essence.
    """
    question = "Does the summary distill the discussion into a concise format while maintaining the essence?"
    return await ask_llm(prompt, response, question)


async def assert_usefulness_clarity_and_succinctness(
    example: dict, prompt: str, response: str
):
    """
    Check if the response is useful, clear, and succinct.
    """
    question = "Is the summary useful, clear, and succinct?"
    return await ask_llm(prompt, response, question)


async def assert_response_structure_followed(example: dict, prompt: str, response: str):
    """
    Check if the workflow follows the specified structure of context, thesis, key points, conclusions, and glossary.
    """
    has_context = "**Context:**" in response
    has_thesis = "**Central Thesis:**" in response
    has_key_points = "**Key Points:**" in response
    has_conclusions = "**Conclusions and Takeaways:**" in response
    has_glossary = "**Glossary of Important Terms:**" in response
    return (
        has_context
        and has_thesis
        and has_key_points
        and has_conclusions
        and has_glossary
    )


async def assert_excludes_irrelevant_information(
    example: dict, prompt: str, response: str
):
    """
    Check if the response excludes any bias, personal opinions, or irrelevant information.
    """
    question = "Does the summary include any bias, personal opinions, or irrelevant information?"
    # Negate the result because we want to ensure these things are excluded
    return not await ask_llm(prompt, response, question)


async def assert_response_uses_markdown_formatting(
    example: dict, prompt: str, response: str
):
    """
    Check that the response is formatted using Markdown syntax.
    """
    return response.startswith("# ") and "- " in response


async def assert_response_uses_bullet_points_for_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check that the response rewrites the essential points using bullet points.
    """
    return "- " in response and "**Key Points:**" in response


async def assert_response_includes_3_to_5_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes at least 3-5 key points.
    """
    key_points_section = response.split("**Key Points:**")[-1].split("**")[0]
    key_points_list = [
        line.strip()
        for line in key_points_section.split("\n")
        if line.strip().startswith("-")
    ]
    return 3 <= len(key_points_list) <= 5


async def assert_response_lists_pivotal_terms_with_definitions(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes a list of pivotal terms with their definitions.
    """
    glossary_section = response.split("**Glossary of Important Terms:**")[-1].split(
        "**"
    )[0]
    return "- " in glossary_section and ":" in glossary_section


async def assert_response_does_not_make_up_definitions(
    example: dict, prompt: str, response: str
):
    """
    Check that the response does not make up definitions for terms.
    """
    question = (
        "Does the response make up definitions for terms it is supposed to explain?"
    )
    # The answer needs to be False for LLM to not made up definitions
    return not await ask_llm(prompt, response, question)


async def assert_response_includes_unknown_definition_clause(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes the clause 'definition unknown' for terms without known meanings.
    """
    glossary_section = response.split("**Glossary of Important Terms:**")[-1].split(
        "**"
    )[0]
    return "definition unknown" in glossary_section


async def assert_response_summarizes_implications_and_insights(
    example: dict, prompt: str, response: str
):
    """
    Check that the response summarizes the overall implications and any actionable insights.
    """
    conclusions_section = response.split("**Conclusions and Takeaways:**")[-1].split(
        "**"
    )[0]
    return conclusions_section.strip() != ""


async def assert_response_adheres_to_provided_structure(
    example: dict, prompt: str, response: str
):
    """
    Check that the response adheres to the structure provided in the prompt to ensure it is helpful, clear, and concise.
    """
    question = "Does the response adhere to the structure provided in the prompt, providing a helpful, clear, and concise summary?"
    return await ask_llm(prompt, response, question)


import re


async def assert_articulates_central_thesis_2(
    example: dict, prompt: str, response: str
):
    """
    Check that the response articulates the central thesis or main argument of the discussion.
    """
    return (
        "Central Thesis:" in response
        and len(response.split("Central Thesis:")[1].split("\n")[0].strip()) > 0
    )


async def assert_reiterates_essential_points(example: dict, prompt: str, response: str):
    """
    Check that the response reiterates the essential points, facts, or arguments made during the talk.
    """
    return (
        "Key Points:" in response
        and len(response.split("Key Points:")[1].split("\n")[0].strip()) > 0
    )


async def assert_identifies_key_points(example: dict, prompt: str, response: str):
    """
    Check that the response identifies at least 3-5 key points.
    """
    key_points_text = response.split("Key Points:")[1].split("\n\n")[0].strip()
    key_points = [
        point for point in key_points_text.split("\n") if point.startswith("-")
    ]
    return 3 <= len(key_points) <= 5


async def assert_uses_bullet_points_for_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check that the response uses bullet points for listing key points.
    """
    key_points_text = response.split("Key Points:")[1].split("\n\n")[0].strip()
    return all(
        line.strip().startswith("-")
        for line in key_points_text.strip().split("\n")
        if line.strip()
    )


async def assert_includes_time_stamps_if_applicable(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes time stamps if they are applicable.
    """
    # This is a more complex check and might need context to decide if
    # timestamps were applicable. This function assumes that if the
    # example text contains phrases that are indicative of a video or
    # audio recording (e.g., timestamps, "at minute 5", etc.), then
    # timestamps should be included in the response.
    indicative_phrases = ["minute", "second", "hour", "at time"]
    should_have_timestamps = any(
        phrase in example["text"].lower() for phrase in indicative_phrases
    )

    has_timestamps = re.search(r"\b\d{1,2}:\d{2}\b", response)
    return not should_have_timestamps or (should_have_timestamps and has_timestamps)


async def assert_includes_date_and_setting_if_applicable(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes the date and setting if applicable.
    """
    # This is also more complex and might need context to decide if the date
    # and setting were included and if they were applicable.
    includes_date_setting = "Context:" in response and "Title:" in response
    question = "Was the inclusion of the date and setting applicable and correctly included in the response?"
    return includes_date_setting and await ask_llm(prompt, response, question)


async def assert_excludes_off_topic_information(
    example: dict, prompt: str, response: str
):
    """
    Check that the response does not mention any off-topic or irrelevant information relative to the prompt template guidelines.
    """
    question = "Does the response mention any off-topic or irrelevant information not aligned with the prompt template guidelines?"
    return not await ask_llm(prompt, response, question)


async def assert_coherent_and_professional_tone(
    example: dict, prompt: str, response: str
):
    """
    Check that the response maintains a coherent and professional tone as implied by the structured and clear instructions.
    """
    question = "Does the response maintain a coherent and professional tone?"
    return await ask_llm(prompt, response, question)


async def assert_response_completeness(example: dict, prompt: str, response: str):
    """
    Check that the response is complete by covering all specified elements in the prompt template.
    """
    required_elements = [
        "Context:",
        "Central Thesis:",
        "Key Points:",
        "Conclusions and Takeaways:",
        "Glossary of Important Terms:",
    ]
    return all(element in response for element in required_elements)


async def assert_markdown_formatting(example: dict, prompt: str, response: str) -> bool:
    """
    Check if the response is formatted in Markdown.
    """
    is_markdown = response.strip().startswith("# ") or any(
        line.strip().startswith("## ") for line in response.splitlines()
    )
    return is_markdown


async def assert_summary_follows_task_directive(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response contains a summary section following the 'TASK:' directive.
    """
    task_index = prompt.find("TASK:")
    summary_index = response.find("## Context:")
    return summary_index > task_index


async def assert_glossary_in_response_2(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response includes a glossary of important terms if such terms exist.
    """
    return "**Glossary of Important Terms:**" in response


async def assert_definitions_provided_or_unknown(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that definitions of terms are provided when terms or jargon are used,
    and the phrase 'definition unknown' is used when the meaning of a term is not known
    or cannot be implied from the text.
    """
    if "**Glossary of Important Terms:**" not in response:
        return True  # If no glossary present, pass by default.

    glossary_section_start = response.index("**Glossary of Important Terms:**")
    glossary_section = response[glossary_section_start:]

    # Split the glossary section into lines and check each term and definition
    for line in glossary_section.splitlines()[1:]:  # Skip the title line
        if ":" not in line:
            continue  # This may be a definition or continuation of a term
        term, definition = line.split(":", 1)
        if "definition unknown" in definition:
            continue
        if not definition.strip():
            return False  # A definition must follow the term
    return True


async def assert_no_made_up_definitions(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Assert that the response adheres to the instruction of not making up definitions.
    """
    # This function is highly ambiguous and domain-specific, so we rely on ask_llm
    question = "Does the response include any made-up definitions for terms?"
    return not await ask_llm(prompt, response, question)


async def assert_helpful_clear_concise_summary(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response helps provide a helpful, clear, and concise summary.
    """
    # Since this is a qualitative assessment, we will need to ask the LLM
    question = "Is the summary helpful, clear, and concise?"
    return await ask_llm(prompt, response, question)


async def assert_summary_within_text_markers(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the summary begins and ends within 'BEGIN TEXT' and 'END TEXT' markers.
    """
    begin_text = prompt.find("BEGIN TEXT")
    end_text = prompt.find("END TEXT")
    # The summary should start after BEGIN TEXT and end before END TEXT
    return response.find("Title:") > begin_text and response.find("END TEXT") < end_text


async def assert_neutral_professional_tone(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response maintains a professional and neutral tone.
    """
    # Since this is a qualitative assessment, we will need to ask the LLM
    question = "Does the response maintain a professional and neutral tone?"
    return await ask_llm(prompt, response, question)


async def assert_correct_markdown_application(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response includes markdown formatting as indicated in the template changes
    with correct application of plus signs.
    """
    # This is an ambiguous assertion, and it's not clear what "template changes"
    # refer to in the given context, so we ask the LLM as a placeholder
    question = "Does the response apply markdown format correctly according to the prompt template?"
    return await ask_llm(prompt, response, question)


async def assert_no_removed_changes_in_response(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response does not include changes that were removed in the prompt template,
    as indicated by the minus sign.
    """
    # As the concept talks about template changes with minus sign there is no
    # concrete example to base this check on, so we ask the LLM as a placeholder
    question = (
        "Does the response omit changes that were removed in the prompt template?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_has_correct_headers(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes specific headers such as 'Central Thesis',
    'Key Points', and 'Conclusions and Takeaways'.
    """
    required_headers = ["Central Thesis", "Key Points", "Conclusions and Takeaways"]
    headers_present = all(header in response for header in required_headers)
    return headers_present


async def assert_key_points_are_summarized(example: dict, prompt: str, response: str):
    """
    Check that the key points are followed up with summarized content in the response.
    """
    start = response.find("**Key Points:**")
    end = response.find("**Conclusions and Takeaways:**")
    if start == -1 or end == -1:
        return False  # Sections missing

    key_points_section = response[start:end]
    # Assuming that key points are summarized if there is content following each bullet point
    return key_points_section.count("-") > 0 and len(
        key_points_section.splitlines()
    ) > key_points_section.count("-")


async def assert_bullet_points_for_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check if 'Key Points' section in the response uses bullet points for easy readability.
    """
    start = response.find("**Key Points:**")
    end = response.find("**Conclusions and Takeaways:**", start)
    key_points_section = response[start:end]

    # Assuming bullet points are marked by asterisks (*) or hyphens (-)
    return "*" in key_points_section or "-" in key_points_section


async def assert_key_points_include_time_stamps_if_applicable(
    example: dict, prompt: str, response: str
):
    """
    Check if the 'Key Points' section includes time stamps where applicable.
    """
    # This is a potentially ambiguous check as not all text will have explicit time stamps.
    # We could ask the LLM if time stamps were applicable and if they are present.
    question = "Are time stamps required and present in the 'Key Points' section?"
    return await ask_llm(prompt, response, question)


async def assert_adequate_number_of_key_points(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes at least 3-5 key points.
    """
    start = response.find("**Key Points:**")
    end = response.find("**Conclusions and Takeaways:**", start)
    key_points_section = response[start:end]
    # Count the number of bullet points, assuming one per key point
    bullet_counts = key_points_section.count("-")
    return 3 <= bullet_counts <= 5


async def assert_use_of_specific_keywords(example: dict, prompt: str, response: str):
    """
    Check if there are keywords that should always be included in the response related to the headers.
    """
    # This would be challenging without context to what the "keywords" are,
    # so here we would be using 'ask_llm' to determine if the keywords related
    # to the headers like 'Central Thesis' are included in the response.
    keywords_related_to_headers = [
        "Central Thesis",
        "Key Points",
        "Conclusions and Takeaways",
    ]
    for keyword in keywords_related_to_headers:
        if keyword not in response:
            return False
    return True


async def assert_informative_tone_for_summary(
    example: dict, prompt: str, response: str
):
    """
    Evaluate if the tone of the response is informative and captures the essence of a summary.
    """
    question = "Is the tone of the LLM response informative, reflecting the essence of a summary?"
    return await ask_llm(prompt, response, question)


async def assert_clear_articulation_of_central_thesis(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response clearly articulates the central thesis or main argument.
    """
    # Since the central thesis' clarity might be subjective, using 'ask_llm' is appropriate.
    question = "Does the response clearly articulate the central thesis or main argument of the discussion?"
    return await ask_llm(prompt, response, question)


async def assert_summary_of_implications_and_insights(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response summarizes the overall implications of the discussion and any actionable insights.
    """
    expected_section = "**Conclusions and Takeaways:**"
    section_exists = expected_section in response
    if not section_exists:
        return False

    # Get the content after the expected section
    content_start = response.find(expected_section) + len(expected_section)
    content = response[content_start:].strip()

    # Make sure that at least something non-trivial is written after the section header
    return len(content) > 10


async def assert_exclusion_of_specific_template_keys(
    example: dict, prompt: str, response: str
):
    """
    Ensure that template placeholder keys or subjects indicated by '-' are not present in the response.
    """
    # This assertion assumes that template placeholder keys are not present in the final response
    # For example, "BEGIN TEXT" and "END TEXT" should not appear in the LLM's summary
    placeholder_keys = ["BEGIN TEXT", "END TEXT"]
    return not any(key in response for key in placeholder_keys)


async def assert_contains_context_section(example: dict, prompt: str, response: str):
    """
    Check if the LLM response begins with a 'Context' section, including title, speaker/author, and subject/theme.
    """
    return response.startswith("**Context:**")


async def assert_articulates_central_thesis_1(
    example: dict, prompt: str, response: str
):
    """
    Check if the 'Central Thesis' section is articulated in the LLM response.
    """
    question = "Does the response clearly articulate the central thesis or main argument of the discussion?"
    return await ask_llm(prompt, response, question)


async def assert_key_points_present_and_summarized(
    example: dict, prompt: str, response: str
):
    """
    Check if 'Key Points' are listed and if they are followed by a summarized version of the actual speaker content.
    """
    if "**Key Points:**" not in response:
        return False
    question = "Are the key points followed by a summarized version of the actual speaker content?"
    return await ask_llm(prompt, response, question)


async def assert_key_points_use_bullet_points(
    example: dict, prompt: str, response: str
):
    """
    Check if 'Key Points' section uses bullet points for easy readability.
    """
    key_points_index = response.find("**Key Points:**")
    subsequent_text = response[key_points_index:]
    return "- " in subsequent_text


async def assert_conclusions_and_takeaways_section(
    example: dict, prompt: str, response: str
):
    """
    Check if the summary concludes with 'Conclusions and Takeaways' encapsulating implications and insights.
    """
    return "**Conclusions and Takeaways:**" in response


async def assert_brevity_and_essence_preserved(
    example: dict, prompt: str, response: str
):
    """
    Check if the response maintains brevity while preserving the essence of the discussion.
    """
    question = "Does the response maintain brevity while preserving the essence of the discussion?"
    return await ask_llm(prompt, response, question)


async def assert_non_biased_journalistic_tone(
    example: dict, prompt: str, response: str
):
    """
    Check if the response maintains a non-biased and journalistic tone.
    """
    question = "Does the response maintain a non-biased and journalistic tone?"
    return await ask_llm(prompt, response, question)


async def assert_response_structure_with_specific_headers(
    example: dict, prompt: str, response: str
):
    """
    Check if the response is structured with specific headers such as 'Context', 'Central Thesis', 'Key Points', 'Conclusions and Takeaways'.
    """
    required_headers = [
        "**Context:**",
        "**Central Thesis:**",
        "**Key Points:**",
        "**Conclusions and Takeaways:**",
    ]
    return all(header in response for header in required_headers)


async def assert_key_points_goal_achieved(example: dict, prompt: str, response: str):
    """
    Check if the goal of the 'Key Points' section is achieved for the reader's comprehension without full reading.
    """
    question = "Does the 'Key Points' section enable the reader to obtain the most critical information without requiring the full reading of the text?"
    return await ask_llm(prompt, response, question)


ALL_FUNCTIONS = [
    assert_response_begins_with_outline,
    assert_response_includes_context,
    assert_response_lists_key_points_with_bullets,
    assert_response_includes_significant_details,
    assert_response_summarizes_implications,
    assert_response_mentions_personal_insights,
    assert_response_word_count_within_limits,
    assert_response_excludes_text_outside_content,
    assert_response_includes_relevant_dates_and_settings,
    assert_response_clear_distinction_between_content_and_insights,
    assert_response_structure_for_usefulness_and_clarity,
    assert_contains_outline,
    assert_contains_context_section,
    assert_contains_significant_details,
    assert_contains_conclusions_takeaways,
    assert_summary_within_word_count,
    assert_response_ends_with_word_count,
    assert_response_proper_summary_structure,
    assert_response_maintains_essence_and_brevity,
    assert_response_free_from_personal_opinions,
    assert_response_clearly_a_summary,
    assert_context_includes_title_author_subject,
    assert_inclusion_of_date_and_setting_if_applicable,
    assert_minimum_key_points_listed,
    assert_brief_and_succinct_preservation_of_essence,
    assert_no_excessive_details,
    assert_accuracy_in_summary_comprehension,
    assert_correct_summary_structure,
    assert_no_unnecessary_urgency,
    assert_clarity_and_format_of_structure,
    assert_response_includes_date_and_setting_if_applicable,
    assert_glossary_in_response_1,
    assert_minimum_number_of_key_points,
    assert_key_points_are_bullet_points,
    assert_significant_details_present,
    assert_conclusions_and_takeaways,
    assert_response_format_markdown,
    assert_no_word_count_requirement,
    assert_task_execution,
    assert_has_context,
    assert_has_glossary_of_terms,
    assert_has_central_thesis,
    assert_has_key_points,
    assert_has_conclusions_and_takeaways,
    assert_is_clear_succinct_and_essence_captured,
    assert_is_formatted_in_markdown,
    assert_no_unnecessary_details,
    assert_includes_significant_details,
    assert_response_begins_with_context_2,
    assert_response_contains_central_thesis,
    assert_response_has_at_least_3_key_points,
    assert_conclusions_and_takeaways_present,
    assert_glossary_included,
    assert_correct_summary_headers,
    assert_exclusion_of_irrelevant_information,
    assert_text_quotation_and_workflow_description,
    assert_comprehension_of_main_ideas_and_details,
    assert_brevity_and_essence_maintenance,
    assert_clear_central_thesis,
    assert_bullet_points_key_points,
    assert_minimum_key_points_count,
    assert_conclusion_and_actionable_insights,
    assert_consistent_detail_level_for_key_points,
    assert_no_extra_headers_or_sections,
    assert_rich_content_in_thesis_and_key_points,
    assert_understanding_of_date_and_setting,
    assert_grammatical_correctness,
    assert_markdown_format,
    assert_context_includes_title_speaker_theme,
    assert_central_thesis_articulated,
    assert_key_points_with_bullet_points,
    assert_conclusions_and_takeaways_summary,
    assert_glossary_with_definitions,
    assert_non_bias_and_journalistic_tone,
    assert_distillation_and_brevity,
    assert_usefulness_clarity_and_succinctness,
    assert_response_structure_followed,
    assert_excludes_irrelevant_information,
    assert_response_uses_markdown_formatting,
    assert_response_articulates_central_thesis,
    assert_response_uses_bullet_points_for_key_points,
    assert_response_includes_3_to_5_key_points,
    assert_response_lists_pivotal_terms_with_definitions,
    assert_response_does_not_make_up_definitions,
    assert_response_includes_unknown_definition_clause,
    assert_response_summarizes_implications_and_insights,
    assert_response_adheres_to_provided_structure,
    assert_articulates_central_thesis_2,
    assert_reiterates_essential_points,
    assert_identifies_key_points,
    assert_uses_bullet_points_for_key_points,
    assert_includes_time_stamps_if_applicable,
    assert_excludes_off_topic_information,
    assert_coherent_and_professional_tone,
    assert_response_completeness,
    assert_markdown_formatting,
    assert_summary_follows_task_directive,
    assert_glossary_in_response_2,
    assert_definitions_provided_or_unknown,
    assert_no_made_up_definitions,
    assert_helpful_clear_concise_summary,
    assert_summary_within_text_markers,
    assert_neutral_professional_tone,
    assert_correct_markdown_application,
    assert_no_removed_changes_in_response,
    assert_response_has_correct_headers,
    assert_key_points_are_summarized,
    assert_bullet_points_for_key_points,
    assert_key_points_include_time_stamps_if_applicable,
    assert_adequate_number_of_key_points,
    assert_use_of_specific_keywords,
    assert_informative_tone_for_summary,
    assert_clear_articulation_of_central_thesis,
    assert_summary_of_implications_and_insights,
    assert_exclusion_of_specific_template_keys,
    assert_articulates_central_thesis_1,
    assert_key_points_present_and_summarized,
    assert_key_points_use_bullet_points,
    assert_conclusions_and_takeaways_section,
    assert_brevity_and_essence_preserved,
    assert_non_biased_journalistic_tone,
    assert_includes_date_and_setting_if_applicable,
    assert_response_structure_with_specific_headers,
    assert_key_points_goal_achieved,
]

# Take only first 70 functions for now
ALL_FUNCTIONS = ALL_FUNCTIONS[:70]
