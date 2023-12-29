from litellm import acompletion


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for an AI negotiation assistant:\n\n{prompt}\n\nHere is the response:\n{response}",
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


async def verify_markdown_headers(example: dict, prompt: str, response: str) -> bool:
    """
    Check if the response employs markdown notation for headers.
    """
    return all(
        header in response
        for header in (
            "# Executive Summary:",
            "# Negotiation Strategy:",
            "# Supplier Summary:",
            "# Product Summary:",
            "# Market Insights:",
            "# Risks and Opportunities:",
            "# Potential Questions:",
        )
    )


async def ensure_all_headers_included(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check if the response includes specified headers: Executive Summary,
    Negotiation Strategy, Supplier Summary, Product Summary,
    Market Insights, Risks and Opportunities, and Potential Questions.
    """
    required_headers = [
        "Executive Summary",
        "Negotiation Strategy",
        "Supplier Summary",
        "Product Summary",
        "Market Insights",
        "Risks and Opportunities",
        "Potential Questions",
    ]
    return all(header in response for header in required_headers)


async def validate_structured_negotiation_process(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response reflects an outlined structured process
    for creating the negotiation strategy.
    """
    section = "# Negotiation Strategy:"
    start = response.find(section)
    if start == -1:
        return False

    strategy_content = response[start + len(section) :].strip()
    return any(
        keyword in strategy_content
        for keyword in ["Key objectives", "Targets", "Fallback positions"]
    )


async def check_for_placeholder_inclusion(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Verify if all the provided placeholders are included in the response.
    """
    placeholders = [
        "{ideal_outcome}",
        "{suppliers}",
        "{product_service_type}",
        "{current_standing}",
        "{purchasing_volume_dollars}",
        "{target_price_reduction}",
        "{other_factors}",
    ]
    return all(
        placeholder.replace("{", "").replace("}", "") in response
        for placeholder in placeholders
    )


async def ensure_no_extra_sections(example: dict, prompt: str, response: str) -> bool:
    """
    Check if the response does not contain any headers or sections not listed in the prompt template.
    """
    extraneous_sections = [
        "# Background:",
        "# Introduction:",
        "# Conclusion:",
        "# Summary:",
        "# Analysis:",
        "# Discussion:",
        "# Recommendations:",
    ]
    return not any(section in response for section in extraneous_sections)


async def assess_language_clarity_and_adaptability(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Ask the LLM if the response is written in simple, clear language,
    and remains adaptive and flexible to other factors like market volatility and technological advancements.
    """
    question = (
        "Is the response written in simple and clear language that is easy to read, "
        "and does it show an adaptive and flexible approach to factors like market volatility and "
        "technological advancements?"
    )
    return await ask_llm(prompt, response, question)


async def evaluate_understanding_of_supplier_relations(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Ask the LLM if the response reflects an understanding of the relationship with suppliers and how purchasing volume affects bargaining power.
    """
    question = (
        "Does the response reflect an understanding of the company's relationship with the suppliers, "
        "emphasizing prior outcomes and experiences, and detail how purchasing volume affects bargaining power?"
    )
    return await ask_llm(prompt, response, question)


async def validate_target_price_focus(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Ask the LLM if the response aims for the target price reduction specified
    and includes the requested header content format for each specified section.
    """
    question = (
        "Does the response aim for the target price reduction specified, and does it "
        "include the requested header content format for each specified section?"
    )
    return await ask_llm(prompt, response, question)


async def examine_strategy_consistency_and_completeness(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Ask the LLM if the response is consistent with the inputs provided,
    reflecting completeness and coherence in strategy.
    """
    question = "Is the response consistent with the provided inputs, showing a complete and coherent strategy?"
    return await ask_llm(prompt, response, question)


async def assert_response_has_markdown_headers(
    example: dict, prompt: str, response: str
):
    """
    Check that the response is formatted with markdown notation including the specific headers.
    """
    headers = [
        "Executive Summary",
        "Negotiation Strategy",
        "Supplier Summary",
        "Product Summary",
        "Market Insights",
        "Risks and Opportunities",
        "Potential Questions",
    ]
    for header in headers:
        if f"## {header}:" not in response:
            return False
    return True


async def assert_response_is_clear_and_simple_language(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to determine if the response is written in clear and simple language.
    """
    question = "Is the language of the response clear, simple, and easy to read?"
    return await ask_llm(prompt, response, question)


async def assert_response_contains_executive_summary_header(
    example: dict, prompt: str, response: str
):
    """
    Check that the 'Executive Summary' header is present in the response.
    """
    return "## Executive Summary:" in response


async def assert_response_contains_negotiation_strategy_header(
    example: dict, prompt: str, response: str
):
    """
    Check that the 'Negotiation Strategy' header is present in the response.
    """
    return "## Negotiation Strategy:" in response


async def assert_response_excludes_unmentioned_topics(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to check that the response does not mention topics or data points not included in the inputs.
    """
    question = "Does the response mention any topics or data points that were not included in the example inputs?"
    return not await ask_llm(prompt, response, question)


async def assert_placeholders_filled_with_example_data(
    example: dict, prompt: str, response: str
):
    """
    Check that placeholders are correctly filled with the provided example data.
    """
    placeholders = {
        "{ideal_outcome}": str(example.get("ideal_outcome", "")),
        "{suppliers}": str(example.get("suppliers", "")),
        "{product_service_type}": str(example.get("product_service_type", "")),
        "{current_standing}": str(example.get("current_standing", "")),
        "{purchasing_volume_dollars}": str(
            example.get("purchasing_volume_dollars", "")
        ),
        "{target_price_reduction}": str(example.get("target_price_reduction", "")),
        "{other_factors}": str(example.get("other_factors", "")),
    }
    for placeholder, data in placeholders.items():
        if placeholder in prompt and data not in response:
            return False
    return True


async def assert_understanding_of_negotiation_context(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to determine if the response demonstrates understanding of the supplier negotiation context.
    """
    question = "Does the response demonstrate an understanding of the supplier negotiation context?"
    return await ask_llm(prompt, response, question)


async def assert_logical_workflow_followed(example: dict, prompt: str, response: str):
    """
    Use LLM to determine if the response follows a logical workflow according to the prompt instructions.
    """
    question = (
        "Does the response follow a logical workflow as per the prompt instructions?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_includes_supplier_analysis(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to check that the response includes analysis of suppliers' strengths, weaknesses, market standing, and negotiation styles.
    """
    question = "Does the response include analysis of the supplier's strengths, weaknesses, market standing, and negotiation styles?"
    return await ask_llm(prompt, response, question)


async def assert_importance_of_purchasing_volume_discussed(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to verify that the response discusses the importance of the purchasing volume in the negotiation strategy.
    """
    question = "Does the response discuss the importance of the purchasing volume in dollars in the negotiation strategy?"
    return await ask_llm(prompt, response, question)


async def assert_response_includes_product_info(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to confirm that the response covers information about the product or service being sourced.
    """
    question = "Does the response include detailed information about the product or service being sourced?"
    return await ask_llm(prompt, response, question)


async def assert_response_includes_current_standing_with_suppliers(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to confirm that the response includes the current standing with the suppliers.
    """
    question = "Does the response take into account the current standing with the suppliers as described in the example?"
    return await ask_llm(prompt, response, question)


async def assert_response_includes_target_price_reduction(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to verify that the response presents a target price reduction as part of the negotiation strategy.
    """
    question = "Does the response present a target price reduction as part of the negotiation strategy?"
    return await ask_llm(prompt, response, question)


async def assert_response_addresses_strategy_flexibility(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to check that the response addresses the adaptability and flexibility of the strategy for other unpredicted factors.
    """
    question = "Does the response demonstrate adaptability and flexibility to handle unpredicted factors such as market volatility and internal changes?"
    return await ask_llm(prompt, response, question)


async def assert_response_is_comprehensive_and_encompassing(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to evaluate whether the response exhibits completeness by creating an all-encompassing supplier negotiation strategy report.
    """
    question = "Does the response create a comprehensive, all-encompassing supplier negotiation strategy report?"
    return await ask_llm(prompt, response, question)


async def assert_response_provides_complete_executive_summary(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to confirm that the response provides an executive summary which includes outcomes and next steps.
    """
    question = "Does the provided executive summary include the negotiation parties, issues discussed, outcomes, and any next steps?"
    return await ask_llm(prompt, response, question)


async def assert_response_outlines_strategy_details(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to verify that the response outlines the key objectives, targets, and fallback positions under the 'Negotiation Strategy' header.
    """
    question = "Does the 'Negotiation Strategy' section outline key objectives, targets, and fallback positions?"
    return await ask_llm(prompt, response, question)


async def assert_response_consistent_with_provided_info(
    example: dict, prompt: str, response: str
):
    """
    Use LLM to confirm that the response is consistent with the information provided, adding no new unrelated details.
    """
    question = "Is the response consistent with the details provided in the prompt, without adding unrelated new information?"
    return await ask_llm(prompt, response, question)


async def assert_markdown_notation_with_headers(
    example: dict, prompt: str, response: str
):
    """
    Checks if the response follows markdown notation format and includes the required headers.
    """
    required_headers = [
        "Executive Summary",
        "Negotiation Strategy",
        "Supplier Summary",
        "Product Summary",
        "Market Insights",
        "Risks and Opportunities",
        "Potential Questions",
    ]
    response_lines = response.splitlines()
    headers_in_response = [
        line.strip("# ").rstrip(":") for line in response_lines if line.startswith("# ")
    ]
    return all(header in headers_in_response for header in required_headers)


async def assert_executive_summary_inclusion(example: dict, prompt: str, response: str):
    """
    Checks if the response includes an 'Executive Summary' header.
    """
    return "Executive Summary:" in response


async def assert_negotiation_strategy_inclusion_and_detail(
    example: dict, prompt: str, response: str
):
    """
    Checks if the 'Negotiation Strategy' section is included, detailed, and non-generic.
    """
    if "Negotiation Strategy:" not in response:
        return False

    question = "Is the 'Negotiation Strategy' section in the response detailed and tailored specifically to the example situation?"
    return await ask_llm(prompt, response, question)


async def assert_supplier_summary_inclusion(example: dict, prompt: str, response: str):
    """
    Checks if the response includes a 'Supplier Summary' header.
    """
    return "Supplier Summary:" in response


async def assert_product_summary_inclusion(example: dict, prompt: str, response: str):
    """
    Checks if the response includes a 'Product Summary' header.
    """
    return "Product Summary:" in response


async def assert_simple_clear_language(example: dict, prompt: str, response: str):
    """
    Evaluates if the response is written in simple and clear language.
    """
    question = "Is the response written in simple and clear language?"
    return await ask_llm(prompt, response, question)


async def assert_bullets_usage_in_sections(example: dict, prompt: str, response: str):
    """
    Validates the use of bullets in the specified sections of the response.
    """
    strategy_section_start = response.find("Negotiation Strategy:")
    # Ensuring the end index doesn't go out of range in case 'Risks and Opportunities' is the last section
    next_section_start = response.find("# ", strategy_section_start + 1)
    negotiation_strategy_section = response[
        strategy_section_start : next_section_start
        if next_section_start != -1
        else None
    ]

    return all(
        "- " in line
        for line in negotiation_strategy_section.splitlines()
        if line.strip()
    )


async def assert_guiding_principle_achievement(
    example: dict, prompt: str, response: str
):
    """
    Confirms whether the ideal outcome is presented as the guiding principle of the strategy in the response.
    """
    question = (
        "Does the response show that the ideal outcome, which is a 20% cost reduction while maintaining"
        " product quality and delivery timelines, is the guiding principle of the negotiation strategy?"
    )
    return await ask_llm(prompt, response, question)


async def assert_purchasing_volume_as_bargaining_factor(
    example: dict, prompt: str, response: str
):
    """
    Determines if the response considers the purchasing volume in dollars as a bargaining factor.
    """
    question = (
        "Does the response consider the purchasing volume of $2 million annually as an essential factor"
        " that can influence bargaining power and the terms that suppliers are willing to agree on?"
    )
    return await ask_llm(prompt, response, question)


async def assert_target_price_reduction_goal(example: dict, prompt: str, response: str):
    """
    Checks if the target price reduction is clearly articulated as part of the strategy in the response.
    """
    target_reduction = example["target_price_reduction"]

    # Look for a sentence or bullet point in the response that mentions the target price reduction goal.
    response_lines = response.splitlines()
    for line in response_lines:
        if line.strip().startswith("- ") and target_reduction in line:
            return True
    return False


async def assert_adaptive_flexible_strategy_to_unpredictables(
    example: dict, prompt: str, response: str
):
    """
    Confirms the strategy's adaptiveness and flexibility to unpredictable factors as described in the response.
    """
    question = (
        "Does the response address how the negotiation strategy will be adaptive and flexible to handle"
        " unpredictable factors like market volatility and global supply chain disruptions?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_comprehensiveness(example: dict, prompt: str, response: str):
    """
    Ensures the response is comprehensive, optimizing sourcing efforts and appropriately mitigates risks.
    """
    question = "Is the response comprehensive and does it optimize sourcing efforts while mitigating risks?"
    return await ask_llm(prompt, response, question)


async def assert_creativity_in_strategy_development(
    example: dict, prompt: str, response: str
):
    """
    Evaluates the creativity aspect of the negotiation strategy as required by the prompt.
    """
    question = "Is the negotiation strategy report creative as mandated by the prompt?"
    return await ask_llm(prompt, response, question)


async def assert_exclusion_of_irrelevant_information(
    example: dict, prompt: str, response: str
):
    """
    Confirms that the response excludes non-relevant or unnecessary information, and stays on topic.
    """
    question = "Does the response exclude non-relevant or unnecessary information and stay on topic?"
    return await ask_llm(prompt, response, question)


# Note: Each of the above functions assumes the `ask_llm` function returns a boolean value corresponding to
# the answer of the question it is provided with. They also assume that the input `example` is a dictionary
# with the structure matching the expected keys used in the prompt template.
async def assert_markdown_headers_present(example: dict, prompt: str, response: str):
    """
    Checks if the response contains markdown notation with specific headers as required.
    """
    required_headers = [
        "# Executive Summary:",
        "# Negotiation Strategy:",
        "# Supplier Summary:",
        "# Product Summary:",
        "# Market Insights:",
        "# Risks and Opportunities:",
        "# Potential Questions:",
    ]
    for header in required_headers:
        if header not in response:
            return False
    return True


async def assert_executive_summary_inclusion(example: dict, prompt: str, response: str):
    """
    Verifies that the response includes an 'Executive Summary' section.
    """
    return "# Executive Summary:" in response


async def assert_negotiation_strategy_inclusion(
    example: dict, prompt: str, response: str
):
    """
    Validates that the response includes a 'Negotiation Strategy' section.
    """
    return "# Negotiation Strategy:" in response


async def assert_detailed_tactics_and_strategies_for_each_supplier(
    example: dict, prompt: str, response: str
):
    """
    Ensures that the response details tactics and strategies for each supplier.
    """
    section_starts = response.find("# Negotiation Strategy:")
    supplier_starts = response.find("##", section_starts)
    return supplier_starts > section_starts


async def assert_specific_tailored_strategies_not_generic(
    example: dict, prompt: str, response: str
):
    """
    Check if the strategies presented are tailored to the specific situation and not generic.
    """
    question = "Does the 'Negotiation Strategy' section in the response provide specific, non-generic strategies tailored to the supplier mentioned in the example?"
    return await ask_llm(prompt, response, question)


async def assert_negotiation_strategy_in_bullet_points(
    example: dict, prompt: str, response: str
):
    """
    Confirm that the 'Negotiation Strategy' section is presented in bullet points.
    """
    section_starts = response.find("# Negotiation Strategy:")
    section_ends = response.find("#", section_starts + 1)
    section_content = response[section_starts:section_ends]
    return "- " in section_content


async def assert_supplier_summary_inclusion(example: dict, prompt: str, response: str):
    """
    Affirms that the response includes a 'Supplier Summary' section.
    """
    return "# Supplier Summary:" in response


async def assert_product_summary_inclusion(example: dict, prompt: str, response: str):
    """
    Confirms that the response includes a 'Product Summary' section.
    """
    return "# Product Summary:" in response


async def assert_language_clarity_and_simplicity(
    example: dict, prompt: str, response: str
):
    """
    Evaluate whether the language used in the response is clear and simple.
    """
    question = "Is the language in the response clear and simple as specified in the example's requirements?"
    return await ask_llm(prompt, response, question)


async def assert_complete_response_with_all_sections(
    example: dict, prompt: str, response: str
):
    """
    Checks if the response is complete, containing all required sections and information per the prompt specified.
    """
    required_sections = [
        "Executive Summary:",
        "Negotiation Strategy:",
        "Supplier Summary:",
        "Product Summary:",
        "Market Insights:",
        "Risks and Opportunities:",
        "Potential Questions:",
    ]
    for section in required_sections:
        if section not in response:
            return False
    return True


async def assert_strategy_consistency_across_suppliers(
    example: dict, prompt: str, response: str
):
    """
    Verifies that the strategy for each supplier aligns with the overall negotiation framework.
    """
    question = "Is the negotiation strategy consistent across different suppliers, aligning with the overall negotiation framework as per the inputs provided?"
    return await ask_llm(prompt, response, question)


async def assert_response_in_bullets(example: dict, prompt: str, response: str):
    """
    Check if the response is presented in bullet points as required by the strategy section format.
    """
    return all(
        line.strip().startswith("-")
        for line in response.splitlines()
        if "Negotiation Strategy:" in line and line.strip()
    )


async def assert_detailed_strategy(example: dict, prompt: str, response: str):
    """
    Check if the response includes a detailed strategy that considers the ideal outcome,
    suppliers, product or service type, and current standing with the suppliers.
    """
    required_elements = [
        "ideal outcome",
        "suppliers",
        "product_service_type",
        "current_standing",
    ]
    return all(element in response for element in required_elements)


async def assert_workflow_followed(example: dict, prompt: str, response: str):
    """
    Check if the response follows the specified workflow: researching the supplier,
    scraping information from the internet, thinking critically about additional searches,
    and not exceeding three iterations.
    """
    question = "Does the response follow the specified workflow including research, scraping, critical thinking about additional searches, and does it assure not to exceed three iterations?"
    return await ask_llm(prompt, response, question)


async def assert_no_excess_iteration(example: dict, prompt: str, response: str):
    """
    Check if the response does not repeat the research iteration process more than three times.
    """
    iterations = response.count("iteration")
    return iterations <= 3


async def assert_inclusion_of_keywords(example: dict, prompt: str, response: str):
    """
    Check if the response includes certain keywords such as 'ideal outcome', 'suppliers',
    'product_service_type', and 'current_standing'.
    """
    keywords = [
        "ideal outcome",
        "suppliers",
        "product_service_type",
        "current_standing",
    ]
    return all(keyword in response for keyword in keywords)


async def assert_facts_only_no_fabrication(example: dict, prompt: str, response: str):
    """
    Check if the response avoids making things up and only includes gathered facts and data.
    """
    question = "Does the response only state facts and data that have been gathered, with no fabrication?"
    return await ask_llm(prompt, response, question)


async def assert_reference_data_inclusion(example: dict, prompt: str, response: str):
    """
    Check if the response includes all reference data and links to back up the research.
    """
    return "http" in response


async def assert_no_repeated_phrases(example: dict, prompt: str, response: str):
    """
    Check if the response doesn't have unnecessary repeated phrases or sentences.
    """
    without_repetition = set(response.split())
    return len(without_repetition) == len(response.split())


async def assert_inclusion_of_supplier_information(
    example: dict, prompt: str, response: str
):
    """
    Check if the response includes updated and recent information about the suppliers such as current news.
    """
    question = "Does the response include updated and recent information about the suppliers such as current news?"
    return await ask_llm(prompt, response, question)


async def assert_professional_and_comprehensive_overview(
    example: dict, prompt: str, response: str
):
    """
    Check if the response maintains a professional tone, factual style, and provides a comprehensive
    overview of the suppliers, product, market insights, risks, opportunities, and potential questions.
    """
    question = "Does the response maintain a professional tone, factual style, and provide a comprehensive overview?"
    return await ask_llm(prompt, response, question)


async def assert_consistency_and_avoid_unnecessary_repetition(
    example: dict, prompt: str, response: str
):
    """
    Check if the response maintains consistency by avoiding unnecessary repetition.
    """
    question = (
        "Does the response maintain consistency and avoid unnecessary repetition?"
    )
    return await ask_llm(prompt, response, question)


async def assert_new_search_iteration_evaluation(
    example: dict, prompt: str, response: str
):
    """
    Check if the response indicates whether there are new things to search after the iterations of research and scraping.
    """
    question = "Does the response reflect an evaluation of whether new searches are needed after research and scraping iterations?"
    return await ask_llm(prompt, response, question)


async def assert_supplier_summary_inclusion(example: dict, prompt: str, response: str):
    """
    Check if the response includes company profile and historical performance in the Supplier Summary section.
    """
    supplier_summary_index = response.find("Supplier Summary:")
    return (
        "company profile" in response[supplier_summary_index:]
        and "historical performance" in response[supplier_summary_index:]
    )


async def assert_exclusion_of_past_negotiations_in_supplier_summary(
    example: dict, prompt: str, response: str
):
    """
    Check if the response does not include past negotiations in the Supplier Summary section.
    """
    supplier_summary_index = response.find("Supplier Summary:")
    return "past negotiations" not in response[supplier_summary_index:]


async def assert_response_tailored_to_each_supplier(
    example: dict, prompt: str, response: str
):
    """
    Check if the response is tailored specifically to the situation of each supplier and is not generic.
    """
    question = "Is the response tailored specifically to each situation for each supplier and not generic?"
    return await ask_llm(prompt, response, question)


async def assert_complete_strategy_addressing_requirements(
    example: dict, prompt: str, response: str
):
    """
    Check if the response is complete by providing a strategy that explicitly addresses the requirements and follows the outlined steps.
    """
    question = "Is the response complete, providing a strategy that explicitly addresses the requirements and follows the outlined steps?"
    return await ask_llm(prompt, response, question)


async def assert_iterates_research_once(example: dict, prompt: str, response: str):
    """
    Checks that the response includes a research iteration process and does it only once.
    """
    return "1 iteration" in response and not "2 iterations" in response


async def assert_no_fabricated_information(example: dict, prompt: str, response: str):
    """
    Verifies that the response does not include made-up information, only factual data.
    """
    question = "Does the response refrain from including any made-up information and relies solely on factual data?"
    return await ask_llm(prompt, response, question)


async def assert_includes_all_reference_data_and_links(
    example: dict, prompt: str, response: str
):
    """
    Ensures that the response includes all reference data and links to back up the research.
    """
    return all(reference in response for reference in ["http", "www"])


async def assert_understands_company_supplier_relationship(
    example: dict, prompt: str, response: str
):
    """
    Verifies that the response shows an understanding of the company's relationship with the suppliers.
    """
    question = "Does the response demonstrate an understanding of the company's current relationship with the suppliers?"
    return await ask_llm(prompt, response, question)


async def assert_aware_of_sourcing_product_and_challenges(
    example: dict, prompt: str, response: str
):
    """
    Confirms that the response exhibits awareness of the type of product or service being sourced and related procurement challenges.
    """
    question = "Does the response demonstrate an awareness of the product being sourced and the procurement challenges linked to it?"
    return await ask_llm(prompt, response, question)


async def assert_includes_company_standing_with_suppliers(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response contains details regarding the company's current standing with the suppliers.
    """
    return "current standing" in response or "historical performance" in response


async def assert_understands_market_and_supplier_position(
    example: dict, prompt: str, response: str
):
    """
    Assesses whether the response reflects understanding of the overall market and the supplier's positions within it.
    """
    question = "Does the response reflect an understanding of the suppliers' strengths, weaknesses, and market standing?"
    return await ask_llm(prompt, response, question)


async def assert_no_redundant_repetition_in_response(
    example: dict, prompt: str, response: str
):
    """
    Ensures that the response avoids unnecessary repetition of information.
    """
    return (
        response.count(
            "You should include all reference data & links to back up your research;"
        )
        == 1
    )


async def assert_only_verifiable_information_in_response(
    example: dict, prompt: str, response: str
):
    """
    Guarantees that the response includes only verifiable information from the research.
    """
    question = "Does the response contain only verifiable information, without resorting to speculation?"
    return await ask_llm(prompt, response, question)


async def assert_details_internet_search_usage(
    example: dict, prompt: str, response: str
):
    """
    Ensures that the response contains details about how an internet search was used to supplement the information.
    """
    return "internet search" in response and "scrape" in response


async def assert_suggests_new_research_angles_based_on_data(
    example: dict, prompt: str, response: str
):
    """
    Verifies if the response suggests new research angles based on the collected data.
    """
    question = "Does the response suggest additional angles of research based on the data that has been collected to enhance research quality?"
    return await ask_llm(prompt, response, question)


async def assert_response_in_bullet_format(example: dict, prompt: str, response: str):
    """
    Check that the negotiation strategy section of the response is in bullet format.
    """
    return "- " in response and "\n-\n" not in response


async def assert_no_example_in_prompt_template(
    example: dict, prompt: str, response: str
):
    """
    Verify that the prompt template does not provide an example of a good response.
    """
    return "No specific example provided in the prompt template." not in prompt


async def assert_one_scraping_iteration(example: dict, prompt: str, response: str):
    """
    Verify that the workflow for research and scraping in the response does not exceed more than one iteration.
    """
    question = "Does the research and scraping process described in the response go through more than one iteration?"
    is_more_than_one_iteration = await ask_llm(prompt, response, question)
    return not is_more_than_one_iteration


async def assert_comprehensive_search_and_facts(
    example: dict, prompt: str, response: str
):
    """
    Verify each section (Supplier Summary, Product Summary, Market Insights, etc.) includes comprehensive search and facts.
    """
    question = "Does each section of the response include a comprehensive search and presentation of facts?"
    return await ask_llm(prompt, response, question)


async def assert_no_made_up_information(example: dict, prompt: str, response: str):
    """
    Check the LLM response for any made-up information.
    """
    question = (
        "Does the response contain any made-up information or unsupported claims?"
    )
    is_any_made_up_information = await ask_llm(prompt, response, question)
    return not is_any_made_up_information


async def assert_provided_references_and_links(
    example: dict, prompt: str, response: str
):
    """
    Verify that references and links are provided to back up research in the response.
    """
    question = "Are there references and links provided in the response to back up the research?"
    return await ask_llm(prompt, response, question)


async def assert_strategy_tailored_to_each_supplier(
    example: dict, prompt: str, response: str
):
    """
    Check that the strategy is tailored specifically for each supplier and is not generic.
    """
    question = "Is the strategy presented in the response tailored specifically for each supplier and not generic?"
    return await ask_llm(prompt, response, question)


async def assert_correct_placeholders_replacement(
    example: dict, prompt: str, response: str
):
    """
    Verify that placeholders in the template are replaced with appropriate example values in the LLM response.
    """
    for placeholder in [
        "{ideal_outcome}",
        "{suppliers}",
        "{product_service_type}",
        "{current_standing}",
        "{purchasing_volume_dollars}",
        "{target_price_reduction}",
        "{other_factors}",
    ]:
        if placeholder in response:
            return False
    return True


async def assert_ideal_outcome_as_guiding_principle(
    example: dict, prompt: str, response: str
):
    """
    Ensure the ideal outcome is mentioned as the guiding principle for the strategy in the LLM response.
    """
    ideal_outcome = example["ideal_outcome"]
    return ideal_outcome in response


async def assert_relationship_with_suppliers_impact(
    example: dict, prompt: str, response: str
):
    """
    Check that the relationship with suppliers and its impact on the strategy is accurately represented in the response.
    """
    question = "Does the response correctly assess the relationship with suppliers and its impact on the negotiation strategy?"
    return await ask_llm(prompt, response, question)


async def assert_incorporate_purchasing_volume(
    example: dict, prompt: str, response: str
):
    """
    Confirm that the purchasing volume in dollars is accurately incorporated as a factor in the negotiation strategy.
    """
    purchasing_volume = example["purchasing_volume_dollars"]
    return purchasing_volume in response


async def assert_achieve_target_price_reduction(
    example: dict, prompt: str, response: str
):
    """
    Ensure the response acknowledges and works towards achieving the target price reduction.
    """
    target_price_reduction = example["target_price_reduction"]
    return target_price_reduction in response


async def assert_no_repetitions_in_response(example: dict, prompt: str, response: str):
    """
    Verify that the LLM response does not contain any repetitions of phrases.
    """
    return (
        response.count(
            "You should include all reference data & links to back up your research;"
        )
        == 1
    )


async def assert_consistency_with_negotiation_strategy(
    example: dict, prompt: str, response: str
):
    """
    Check that each section of the response maintains consistency with the negotiation strategy elements.
    """
    question = (
        "Does each section of the response maintain consistency with the negotiation strategy elements"
        " presented in the prompt?"
    )
    return await ask_llm(prompt, response, question)


async def assert_response_backs_up_research_with_data_and_links(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response backs up research with data and links.
    """
    # This criterion is subjective and may need contextually relevant data to confirm.
    # It is more straight-forward to use expert LLM judgment in this case.
    question = "Does the response back up the research with actual data and links from credible sources as required?"
    return await ask_llm(prompt, response, question)


async def assert_response_does_not_iterate_scraping_more_than_once(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response does not iterate the scraping and search process more than once.
    """
    # Closely look for any mention of a repeated search or scraping iteration.
    iteration_terms = [
        "second iteration",
        "third iteration",
        "another iteration",
        "repeated search",
        "search again",
        "scrape again",
        "additional search",
    ]
    for term in iteration_terms:
        if term in response:
            return False
    return True


async def assert_response_provides_facts_and_data_only(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response provides only facts and data gathered without any made-up information.
    """
    # This is subjective and best suited for an expert LLM to evaluate.
    question = "Does the response only contain facts and data that were gathered, and does not include any made-up information?"
    return await ask_llm(prompt, response, question)


async def assert_response_includes_product_research(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response includes research on the mentioned product or service type.
    """
    # Looking for specific mentions of product-related research in the response.
    product_service_type = example.get("product_service_type", "")
    return product_service_type in response


async def assert_response_considers_current_standing_with_suppliers(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response considers the current standing with the suppliers.
    """
    # Looking for mentions matching the given current standing with suppliers in the response.
    current_standing = example.get("current_standing", "")
    return current_standing in response


async def assert_response_includes_purchasing_volume(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response includes the purchasing volume in dollars.
    """
    # Looking for the mention of the specific purchasing volume in dollars in the response.
    purchasing_volume_dollars = example.get("purchasing_volume_dollars", "")
    return purchasing_volume_dollars in response


async def assert_response_addresses_target_price_reduction(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response includes the target price reduction goal.
    """
    # Looking for the specific target price reduction goal in the response.
    target_price_reduction = example.get("target_price_reduction", "")
    return target_price_reduction in response


async def assert_response_avoids_removed_research_instructions(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response does not include specific instructions on doing research that have been removed from the prompt template.
    """
    removed_instructions = (
        "do enough research to gather as much information as possible about the market"
    )
    return removed_instructions not in response


async def assert_response_includes_market_insights(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response analyzes current market conditions.
    """
    header = "Market Insights"
    return header in response and "current market conditions" in response


async def assert_response_identifies_risks_opportunities(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response identifies and analyzes potential risks and opportunities associated with the negotiation.
    """
    return "Risks and Opportunities" in response


async def assert_response_lists_potential_questions(
    example: dict, prompt: str, response: str
):
    """
    Checks that the response includes potential questions that may arise during the negotiation process.
    """
    return "Potential Questions" in response and any(
        question in response
        for question in ["Can you", "Will you", "How will", "What if"]
    )


ALL_FUNCTIONS = [
    verify_markdown_headers,
    ensure_all_headers_included,
    validate_structured_negotiation_process,
    check_for_placeholder_inclusion,
    ensure_no_extra_sections,
    assess_language_clarity_and_adaptability,
    evaluate_understanding_of_supplier_relations,
    validate_target_price_focus,
    examine_strategy_consistency_and_completeness,
    assert_detailed_strategy,
    assert_workflow_followed,
    assert_no_excess_iteration,
    assert_inclusion_of_keywords,
    assert_facts_only_no_fabrication,
    assert_reference_data_inclusion,
    assert_no_repeated_phrases,
    assert_inclusion_of_supplier_information,
    assert_professional_and_comprehensive_overview,
    assert_consistency_and_avoid_unnecessary_repetition,
    assert_new_search_iteration_evaluation,
    assert_supplier_summary_inclusion,
    assert_exclusion_of_past_negotiations_in_supplier_summary,
    assert_response_tailored_to_each_supplier,
    assert_complete_strategy_addressing_requirements,
    assert_iterates_research_once,
    assert_no_fabricated_information,
    assert_includes_all_reference_data_and_links,
    assert_understands_company_supplier_relationship,
    assert_aware_of_sourcing_product_and_challenges,
    assert_includes_company_standing_with_suppliers,
    assert_understands_market_and_supplier_position,
    assert_no_redundant_repetition_in_response,
    assert_only_verifiable_information_in_response,
    assert_details_internet_search_usage,
    assert_suggests_new_research_angles_based_on_data,
    assert_response_in_bullet_format,
    assert_no_example_in_prompt_template,
    assert_one_scraping_iteration,
    assert_comprehensive_search_and_facts,
    assert_no_made_up_information,
    assert_provided_references_and_links,
    assert_strategy_tailored_to_each_supplier,
    assert_correct_placeholders_replacement,
    assert_ideal_outcome_as_guiding_principle,
    assert_relationship_with_suppliers_impact,
    assert_incorporate_purchasing_volume,
    assert_achieve_target_price_reduction,
    assert_no_repetitions_in_response,
    assert_consistency_with_negotiation_strategy,
    assert_response_backs_up_research_with_data_and_links,
    assert_response_does_not_iterate_scraping_more_than_once,
    assert_response_provides_facts_and_data_only,
    assert_response_includes_product_research,
    assert_response_considers_current_standing_with_suppliers,
    assert_response_includes_purchasing_volume,
    assert_response_addresses_target_price_reduction,
    assert_response_avoids_removed_research_instructions,
    assert_response_includes_market_insights,
    assert_response_identifies_risks_opportunities,
    assert_response_lists_potential_questions,
]

# Take only the first 25 and last 25 functions
ALL_FUNCTIONS = ALL_FUNCTIONS[:25] + ALL_FUNCTIONS[-25:]
