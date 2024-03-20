from litellm import acompletion


async def ask_llm(response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are a helpful assistant. Here is a response you will be evaluating:\n{response}",
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


import asyncio


async def assert_approx_200_words(response: str) -> bool:
    """
    Checks if the response is approximately 200 words in length.
    """
    word_count = len(response.split())
    return 180 <= word_count <= 220  # Allowing some flexibility around 200 words


async def assert_markdown_format_a(response: str) -> bool:
    """
    Checks if the response is in Markdown format by looking for Markdown features.
    """
    contains_headers = "#" in response
    contains_bold = "**" in response or "__" in response
    contains_italics = "*" in response or "_" in response
    contains_list = "-" in response or "*" in response  # Unordered lists
    return contains_headers and contains_bold and contains_italics and contains_list


async def assert_active_voice(response: str) -> bool:
    """
    Check if the response predominantly uses an active voice. This is a qualitative assessment
    that would likely require the expertise of an LLM or another form of NLP analysis.
    """
    question = "Is the response written predominantly in an active voice?"
    return await ask_llm(response, question)


async def assert_includes_features_and_benefits(response: str) -> bool:
    """
    Check if the response includes both the product's features and benefits.
    """
    question = "Does the response include both the product's features and benefits?"
    return await ask_llm(response, question)


async def assert_includes_unique_selling_points_and_cta(response: str) -> bool:
    """
    Check if the response includes the product's unique selling points and a call to action for the buyer.
    """
    question = "Does the response include the product's unique selling points and a call to action for the buyer?"
    return await ask_llm(response, question)


async def assert_distinguishes_features_benefits(response: str) -> bool:
    """
    Check if the response clearly distinguishes between features and benefits.
    """
    features_mentioned = "Features:" in response or "**Features:**" in response
    benefits_mentioned = "Benefits:" in response or "**Benefits:**" in response
    return features_mentioned and benefits_mentioned


async def assert_not_mention_weaknesses(response: str) -> bool:
    """
    Check if the response does not mention weaknesses of the product.
    """
    question = "Does the response mention any weaknesses of the product?"
    return not await ask_llm(response, question)


async def assert_avoid_generic_repetitive_language(response: str) -> bool:
    """
    Check if the response avoids using generic or repetitive language.
    """
    question = "Does the response avoid using generic or repetitive language?"
    return await ask_llm(response, question)


async def assert_seo_optimized(response: str) -> bool:
    """
    Check if the response is SEO-optimized. This is a qualitative assessment.
    """
    question = "Is the response SEO-optimized?"
    return await ask_llm(response, question)


async def assert_no_overpromise(response: str) -> bool:
    """
    Check if the response does not overpromise on the product's capabilities.
    """
    question = "Does the response overpromise on the product's capabilities?"
    return not await ask_llm(response, question)


async def assert_readable_chunks_with_subheadings_a(response: str) -> bool:
    """
    Check if the response is divided into readable chunks with relevant subheadings.
    """
    return response.count("#") >= 2  # Expecting at least two subheadings


async def assert_no_fabricated_reviews(response: str):
    """
    Check that the response does not contain fabricated review text or quotes.
    """
    return not any(fake_review in response for fake_review in ["5/5", "4/5", "*sigh*"])


async def assert_no_links_in_response(response: str):
    """
    Check that the response does not contain any links.
    """
    return "http://" not in response and "https://" not in response


async def assert_reviews_not_overcited(response: str):
    """
    Check that the response does not rely too heavily on citing reviews.
    """
    review_citations = response.count("*") + response.count('"')
    return (
        review_citations <= 4
    )  # Assuming each review starts and ends with a quote or star.


async def assert_readable_chunks_with_subheadings_b(response: str):
    """
    Check that the response is divided into readable chunks with relevant subheadings.
    """
    return response.count("## ") > 0 and all(
        "\n\n" in section for section in response.split("## ")[1:]
    )


async def assert_proper_length(response: str):
    """
    Check that the response has a length around 200 words, and does not exceed 300 words.
    """
    word_count = len(response.split())
    return 200 <= word_count <= 300


async def assert_markdown_formatting_b(response: str):
    """
    Check that the response is formatted in Markdown.
    """
    return response.startswith("# ") and any(
        subheading in response for subheading in ["## ", "### ", "#### "]
    )


async def assert_features_and_benefits_distinction(response: str):
    """
    Check that the response clearly distinguishes between product features and benefits.
    """
    question = "Does the provided text clearly distinguish between product features and benefits?"
    return await ask_llm(response, question)


async def assert_no_weakness_mention(response: str):
    """
    Check that the response does not mention any weaknesses of the product.
    """
    negative_phrases = [
        "dry out",
        "never order",
        "disappointed to see them sold in my local dollar store",
    ]
    return not any(negative_phrase in response for negative_phrase in negative_phrases)


async def assert_no_generic_repetitive_language_a(response: str):
    """
    Check that the response avoids using generic or repetitive language.
    """
    word_counts = {}
    for word in response.split():
        if word.lower() not in word_counts:
            word_counts[word.lower()] = 1
        else:
            word_counts[word.lower()] += 1
    return all(
        count < 3 for count in word_counts.values()
    )  # Arbitrary threshold for repetitiveness


ALL_FUNCTIONS = [
    assert_approx_200_words,
    assert_markdown_format_a,
    assert_active_voice,
    assert_includes_features_and_benefits,
    assert_includes_unique_selling_points_and_cta,
    assert_distinguishes_features_benefits,
    assert_not_mention_weaknesses,
    assert_avoid_generic_repetitive_language,
    assert_seo_optimized,
    assert_no_overpromise,
    assert_readable_chunks_with_subheadings_a,
    assert_no_fabricated_reviews,
    assert_no_links_in_response,
    assert_reviews_not_overcited,
    assert_readable_chunks_with_subheadings_b,
    assert_proper_length,
    assert_markdown_formatting_b,
    assert_features_and_benefits_distinction,
    assert_no_weakness_mention,
    assert_no_generic_repetitive_language_a,
]
