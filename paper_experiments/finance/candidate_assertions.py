from litellm import acompletion


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for a AI-powered expert in finance summarizng earnings calls:\n\n{prompt}\n\nHere is the response:\n{response}",
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


async def assert_synthesize_key_insights(example: dict, prompt: str, response: str):
    """
    Check that the response synthesizes key insights from the transcript into a short form narrative.
    """
    question = "Does the response synthesize key insights from the transcript into a short form narrative?"
    return await ask_llm(prompt, response, question)


async def assert_focus_on_qualitative_quantitative_aspects(
    example: dict, prompt: str, response: str
):
    """
    Check that the response focuses on both qualitative and quantitative aspects of the company's performance.
    """
    includes_qualitative = "qualitative aspects" in response.lower()
    includes_quantitative = "quantitative aspects" in response.lower()
    return includes_qualitative and includes_quantitative


async def assert_rich_comprehensive_concise(example: dict, prompt: str, response: str):
    """
    Check that the response is rich, comprehensive yet concise.
    """
    question = "Is the response rich, comprehensive yet concise?"
    return await ask_llm(prompt, response, question)


async def assert_include_rich_color(example: dict, prompt: str, response: str):
    """
    Check that the response includes 'rich color' from the transcript.
    """
    question = "Does the response include 'rich color' from the transcript?"
    return await ask_llm(prompt, response, question)


async def assert_prioritize_important_info(example: dict, prompt: str, response: str):
    """
    Check that the response prioritizes the most important information first.
    """
    question = "Does the response prioritize the most important information first?"
    return await ask_llm(prompt, response, question)


async def assert_elucidate_company_guidance(example: dict, prompt: str, response: str):
    """
    Check that the response elucidates the company's guidance and rationale as indicated by the transcript.
    """
    question = "Does the response elucidate the company's guidance and the rationale behind it as indicated in the transcript?"
    return await ask_llm(prompt, response, question)


async def assert_avoid_general_statements_and_conciseness(
    example: dict, prompt: str, response: str
):
    """
    Check that the response avoids general statements and is concise.
    """
    question = "Does the response avoid general statements and maintain conciseness?"
    return await ask_llm(prompt, response, question)


async def assert_bucket_similar_findings(example: dict, prompt: str, response: str):
    """
    Check that the response buckets similar findings into grouped themes to consolidate the distillation.
    """
    question = "Does the response bucket similar findings into grouped themes to consolidate the distillation?"
    return await ask_llm(prompt, response, question)


async def assert_use_shorthand_reduce_verbosity(
    example: dict, prompt: str, response: str
):
    """
    Check that the response uses shorthand or quick speech to reduce verbosity.
    """
    question = "Does the response use shorthand or quick speech to reduce verbosity?"
    return await ask_llm(prompt, response, question)


async def assert_clear_picture_financial_health(
    example: dict, prompt: str, response: str
):
    """
    Check that the response provides a clear picture of the company's financial health, market conditions,
    and future prospects.
    """
    question = "Does the response provide a clear picture of the company's financial health, market conditions, and future prospects?"
    return await ask_llm(prompt, response, question)


async def assert_avoid_general_statements(example: dict, prompt: str, response: str):
    """
    Check that the response does not contain general statements but includes rich details from the transcript.
    """
    question = "Does the response avoid general statements and include rich details from the transcript?"
    return await ask_llm(prompt, response, question)


async def assert_coverage_of_specified_topics(
    example: dict, prompt: str, response: str
):
    """
    Check that the response covers a specific set of financial and performance-related topics without
    being limited to them.
    """
    question = "Does the response cover the specified financial and performance-related topics, and potentially more?"
    return await ask_llm(prompt, response, question)


async def assert_prioritization_of_information(
    example: dict, prompt: str, response: str
):
    """
    Check that the response prioritizes the most important information first.
    """
    question = "Does the response prioritize the most important information first?"
    return await ask_llm(prompt, response, question)


async def assert_company_guidance_explanation(
    example: dict, prompt: str, response: str
):
    """
    Check that the response explains the company's guidance and rationale as stated in the transcript.
    """
    question = "Does the response explain the company's guidance and the rationale behind it as indicated in the transcript?"
    return await ask_llm(prompt, response, question)


async def assert_specialized_detail_and_conciseness(
    example: dict, prompt: str, response: str
):
    """
    Check that the response is detailed where necessary but is overall concise.
    """
    question = "Is the response detailed where necessary yet overall concise?"
    return await ask_llm(prompt, response, question)


async def assert_no_overall_assessment(example: dict, prompt: str, response: str):
    """
    Check that the response avoids giving an overall assessment or conclusion.
    """
    question = "Does the response avoid providing an overall assessment or conclusion?"
    return await ask_llm(prompt, response, question)


async def assert_conciseness_without_sacrificing_info(
    example: dict, prompt: str, response: str
):
    """
    Check that the response is as concise as possible, reducing verbosity without sacrificing crucial information.
    """
    question = "Is the response concise without sacrificing any crucial information?"
    return await ask_llm(prompt, response, question)


async def assert_exclusion_of_non_relevant_info(
    example: dict, prompt: str, response: str
):
    """
    Check that the response excludes any non-relevant information that does not fall into the specified buckets of topics.
    """
    question = "Does the response exclude any non-relevant information that does not fall into the specified buckets of topics?"
    return await ask_llm(prompt, response, question)


async def assert_response_is_short_narrative(example: dict, prompt: str, response: str):
    """
    Check that the LLM response synthesizes key insights into a short form narrative.
    """
    question = "Does the response synthesize key insights into a short form narrative?"
    return await ask_llm(prompt, response, question)


async def assert_includes_qualitative_and_quantitative_aspects(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response includes both qualitative and quantitative aspects of the company's performance.
    """
    question = (
        "Does the response include both qualitative and quantitative aspects "
        "of the company's performance?"
    )
    return await ask_llm(prompt, response, question)


async def assert_prioritizes_important_information(
    example: dict, prompt: str, response: str
):
    """
    Check that the response prioritizes the most important information first.
    """
    question = "Does the response prioritize the most important information first?"
    return await ask_llm(prompt, response, question)


async def assert_elucidates_company_guidance(example: dict, prompt: str, response: str):
    """
    Check that the response elucidates the company's guidance and the rationale behind it.
    """
    question = (
        "Does the response elucidate the company's guidance and the rationale behind it, "
        "as indicated in the transcript?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_is_concise_and_non_general(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response is concise and avoids general statements.
    """
    question = "Is the response concise and does it avoid general statements?"
    return await ask_llm(prompt, response, question)


async def assert_response_is_not_overly_verbose(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response is not overly verbose.
    """
    question = (
        "Is the response not overly verbose, and does it use shorthand or quick speech?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_avoids_overall_assessment_or_conclusion(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response does not give an overall assessment or conclusion.
    """
    question = "Does the response avoid giving an overall assessment or conclusion about the company's performance?"
    return await ask_llm(prompt, response, question)


async def assert_quantitative_data_with_context_inclusion(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response includes all available quantitative data with surrounding contextual information.
    """
    question = (
        "Does the response include every single piece of quantitative data along with all surrounding "
        "contextual information?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_avoids_speculation(example: dict, prompt: str, response: str):
    """
    Check that the LLM response refrains from speculating unless the information is explicitly stated in the transcript.
    """
    question = "Does the response avoid speculation and only include information explicitly stated in the transcript?"
    return await ask_llm(prompt, response, question)


async def assert_acceptable_omission_for_irrelevant_information(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response omits aspects without relevant information.
    """
    question = (
        "Does the response omit aspects that lack relevant information or if specific details are missing, "
        "as permissible per the instructions?"
    )
    return await ask_llm(prompt, response, question)


async def assert_similar_findings_are_grouped(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response buckets similar findings into grouped themes.
    """
    question = "Does the response bucket similar findings into grouped themes to consolidate the distillation?"
    return await ask_llm(prompt, response, question)


async def assert_conciseness_without_sacrificing_information(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response is concise without sacrificing crucial information.
    """
    question = "Is the response as concise as possible without sacrificing any crucial information?"
    return await ask_llm(prompt, response, question)


async def assert_detailed_explanations_when_necessary(
    example: dict, prompt: str, response: str
):
    """
    Check that the response provides detailed explanations where necessary.
    """
    question = "Does the response provide detailed explanations where necessary?"
    return await ask_llm(prompt, response, question)


async def assert_inclusion_of_rich_color_from_transcript(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes rich color provided in the transcript.
    """
    question = "Does the response include rich color, as provided in the transcript?"
    return await ask_llm(prompt, response, question)


async def assert_key_insight_synthesis(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response provides synthesized key insights from the financial earnings call transcript.
    """
    return await ask_llm(
        prompt,
        response,
        "Is the key insight from the transcript synthesized adequately?",
    )


async def assert_focus_on_performance_aspects(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response focuses on both qualitative and quantitative aspects of the company's performance.
    """
    return await ask_llm(
        prompt,
        response,
        "Does the response focus on both qualitative and quantitative performance aspects as intended?",
    )


async def assert_inclusion_of_rich_color(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response includes rich color provided from the earnings call transcript.
    """

    return await ask_llm(
        prompt,
        response,
        "Does the response include the rich color provided in the transcript, indicating specific examples or quotes?",
    )


async def assert_no_overall_assessment(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response does not present an overall assessment or conclusion.
    """

    return await ask_llm(
        prompt,
        response,
        "Has the response abstained from giving any overall assessment or conclusion?",
    )


async def assert_conciseness(example: dict, prompt: str, response: str) -> bool:
    """
    Check that the response is in a concise narrative form.
    """

    return await ask_llm(
        prompt,
        response,
        "Is the narrative concise as required, without unnecessary details?",
    )


async def assert_avoidance_of_verbosity(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response avoids verbosity while not sacrificing crucial information.
    """
    return await ask_llm(
        prompt,
        response,
        "Does the response maintain necessary detail while avoiding excess verbosity?",
    )


async def assert_exclusion_of_non_relevant_info(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response excludes non-relevant information.
    """

    return await ask_llm(
        prompt,
        response,
        "Has the response successfully excluded information that isn't relevant to the synthesized insight?",
    )


async def assert_inclusion_of_key_business_aspects(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response includes mention of key business aspects listed in the prompt.
    """
    return await ask_llm(
        prompt,
        response,
        "Are all the listed key business aspects addressed in the response?",
    )


async def assert_theme_grouping(example: dict, prompt: str, response: str) -> bool:
    """
    Check that the response groups similar findings into themes.
    """
    return await ask_llm(
        prompt,
        response,
        "Are similar findings grouped into themes for better consolidation in the response?",
    )


async def assert_includes_quantitative_data_and_context(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes both quantitative data and the surrounding contextual information.
    """
    question = "Does the response include all pieces of quantitative data and their surrounding contextual information?"
    return await ask_llm(prompt, response, question)


async def assert_elucidates_guidance_and_rationale(
    example: dict, prompt: str, response: str
):
    """
    Check that the response elucidates the company's guidance and the rationale behind it.
    """
    question = "Does the response elucidate the company's guidance and the rationale behind it as indicated in the transcript?"
    return await ask_llm(prompt, response, question)


async def assert_avoids_general_statements(example: dict, prompt: str, response: str):
    """
    Check that the response avoids general statements.
    """
    question = "Does the response avoid general statements and instead provide specific, detailed information?"
    return await ask_llm(prompt, response, question)


async def assert_captures_relevant_context(example: dict, prompt: str, response: str):
    """
    Check that the response captures and includes all relevant context.
    """
    question = "Does the response capture and include all relevant context from the provided transcript?"
    return await ask_llm(prompt, response, question)


async def assert_groups_similar_findings(example: dict, prompt: str, response: str):
    """
    Check that the response buckets similar findings into grouped themes to consolidate the distillation.
    """
    question = "Does the response bucket similar findings into grouped themes to consolidate the distillation?"
    return await ask_llm(prompt, response, question)


async def assert_refrains_from_speculation(example: dict, prompt: str, response: str):
    """
    Check that the response refrains from speculating unless the information is explicitly stated in the transcript.
    """
    question = "Does the response refrain from speculating and only include information that is explicitly stated in the transcript?"
    return await ask_llm(prompt, response, question)


async def assert_acceptable_omissions(example: dict, prompt: str, response: str):
    """
    Check that it is acceptable to omit certain aspects if the text does not provide relevant information or if specific details are missing.
    """
    # This is inherently difficult to assert programmatically as it involves judgment of what is relevant.
    # Instead, we confirm whether the response includes all the relevant data.
    question = "Does the response include all the relevant information from the transcript, or are there appropriate omissions where the text does not provide relevant details?"
    return await ask_llm(prompt, response, question)


async def assert_includes_all_context_even_if_irrelevant_seeming(
    example: dict, prompt: str, response: str
):
    """
    Check that the response includes context around anything summarized, even if it seems irrelevant.
    """
    question = "Does the response include context around summaries even if that context seems irrelevant?"
    return await ask_llm(prompt, response, question)


ALL_FUNCTIONS = [
    assert_synthesize_key_insights,
    assert_focus_on_qualitative_quantitative_aspects,
    assert_rich_comprehensive_concise,
    assert_include_rich_color,
    assert_prioritize_important_info,
    assert_elucidate_company_guidance,
    assert_avoid_general_statements_and_conciseness,
    assert_bucket_similar_findings,
    assert_use_shorthand_reduce_verbosity,
    assert_clear_picture_financial_health,
    assert_avoid_general_statements,
    assert_coverage_of_specified_topics,
    assert_prioritization_of_information,
    assert_company_guidance_explanation,
    assert_specialized_detail_and_conciseness,
    assert_no_overall_assessment,
    assert_conciseness_without_sacrificing_info,
    assert_exclusion_of_non_relevant_info,
    assert_response_is_short_narrative,
    assert_includes_qualitative_and_quantitative_aspects,
    assert_prioritizes_important_information,
    assert_elucidates_company_guidance,
    assert_response_is_concise_and_non_general,
    assert_response_is_not_overly_verbose,
    assert_response_avoids_overall_assessment_or_conclusion,
    assert_quantitative_data_with_context_inclusion,
    assert_response_avoids_speculation,
    assert_acceptable_omission_for_irrelevant_information,
    assert_similar_findings_are_grouped,
    assert_conciseness_without_sacrificing_information,
    assert_detailed_explanations_when_necessary,
    assert_inclusion_of_rich_color_from_transcript,
    assert_key_insight_synthesis,
    assert_focus_on_performance_aspects,
    assert_inclusion_of_rich_color,
    assert_no_overall_assessment,
    assert_conciseness,
    assert_avoidance_of_verbosity,
    assert_exclusion_of_non_relevant_info,
    assert_inclusion_of_key_business_aspects,
    assert_theme_grouping,
    assert_includes_quantitative_data_and_context,
    assert_elucidates_guidance_and_rationale,
    assert_avoids_general_statements,
    assert_captures_relevant_context,
    assert_groups_similar_findings,
    assert_refrains_from_speculation,
    assert_acceptable_omissions,
    assert_includes_all_context_even_if_irrelevant_seeming,
]
