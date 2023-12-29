from litellm import acompletion
import emoji


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for a large language model pipeline that writes custom Twitter threads. Here is the prompt:\n\n{prompt}\n\nHere is the response:\n{response}",
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


async def assert_concise_within_word_limit(example: dict, prompt: str, response: str):
    """
    Check that each tweet is concise, adhering strictly to the word limit provided.
    """
    word_limit = example["word_count"]
    tweets = response.split("\n")

    for tweet in tweets:
        if tweet.startswith("Tweet") and len(tweet.split()) > word_limit:
            return False

    return True


async def assert_no_hashtags(example: dict, prompt: str, response: str):
    """
    Check that tweets do not contain hashtags.
    """
    return "#" not in response


async def assert_minimal_emojis_and_icons(example: dict, prompt: str, response: str):
    """
    Make a qualitative assessment using `await ask_llm` to determine if emojis and icons usage is minimal.
    """
    question = "Does this Twitter thread use emojis and icons minimally?"
    return await ask_llm(prompt, response, question)


async def assert_thread_format(example: dict, prompt: str, response: str):
    """
    Check that the response is formatted as a Twitter thread, not as a single block of text.
    """
    return (
        response.strip().startswith("TWITTER THREAD:\n\nTweet 1:")
        and "\nTweet " in response
    )


async def assert_engaging_hook_main_tweet(example: dict, prompt: str, response: str):
    """
    Make a qualitative assessment for whether the main tweet contains an engaging hook.
    """
    question = (
        "Does the main tweet contain an engaging hook using subtle and dense wording?"
    )
    return await ask_llm(prompt, response, question)


async def assert_inclusion_of_mentions(example: dict, prompt: str, response: str):
    """
    Check that the response includes mentions (@usernames) to improve thread visibility and engagement.
    """
    tweets = response.split("\n")
    return any("@" in tweet for tweet in tweets if tweet.startswith("Tweet"))


async def assert_coherent_narrative_thread(example: dict, prompt: str, response: str):
    """
    Make a qualitative assessment to ensure each tweet contributes to a coherent narrative in the thread.
    """
    question = "Does each tweet contribute to a coherent narrative, ensuring the Twitter thread tells a complete story or provides valuable insights?"
    return await ask_llm(prompt, response, question)


async def assert_engaging_and_informative_thread(
    example: dict, prompt: str, response: str
):
    """
    Check that the generated Twitter thread aims to be engaging and informative, prompting interaction and sharing.
    """
    question = "Is this Twitter thread engaging and informative, does it prompt interaction and encourage sharing?"
    return await ask_llm(prompt, response, question)


async def assert_proper_thread_start(example: dict, prompt: str, response: str):
    """
    Check that the response for the Twitter thread starts correctly with 'TWITTER THREAD:'.
    """
    return response.strip().startswith("TWITTER THREAD:")


async def assert_individual_tweets_format(example: dict, prompt: str, response: str):
    """
    Check that the individual tweets within the thread are not comma-separated or formatted as a JSON object.
    """
    tweets = response.split("\n")
    return all(not tweet.strip().startswith("{") for tweet in tweets if tweet.strip())


async def assert_no_hashtags_or_emojis_1(example: dict, prompt: str, response: str):
    """
    Check that the response does not include any hashtags or emojis.
    """
    return (
        "#" not in response
        and "" not in response
        and "'" not in response
        and " " not in response
        and "" not in response
        and "'" not in response
        and " " not in response
        and "d" not in response
        and "" not in response
        and "'✈" not in response
        and "'" not in response
        and " " not in response
        and "" not in response
    )


async def assert_response_language(example: dict, prompt: str, response: str):
    """
    Check that the response is in the specified language indicated in the placeholders of the template.
    """
    requested_language = example.get("language", "").lower()
    if requested_language not in [
        "english"
    ]:  # Extend this if there are more languages to support
        return False
    return True  # This is a place holder, actual language detection requires more sophisticated checks or await ask_llm


async def assert_response_length_within_limit(
    example: dict, prompt: str, response: str
):
    """
    Check that the response does not exceed the specified word count limit for each tweet.
    """
    word_count_limit = example.get("word_count", 0)
    tweets = response.split("Tweet")[
        1:
    ]  # Assuming each tweet starts with 'Tweet' followed by a number
    for tweet in tweets:
        tweet_word_count = len(tweet.split())
        if tweet_word_count > word_count_limit:
            return False
    return True


async def assert_maximum_five_tweets(example: dict, prompt: str, response: str):
    """
    Check that the response is presented as a thread with a maximum of 5 tweets.
    """
    return (
        len(response.strip().split("Tweet")) <= 6
    )  # We add 1 because split starts with an empty string


async def assert_work_flow_actions_included(example: dict, prompt: str, response: str):
    """
    Check if the words 'Craft', 'Transform', and 'Generate' are present in the response.
    """
    return all(action in response for action in ["Craft", "Transform", "Generate"])


async def assert_thread_start_indicator(example: dict, prompt: str, response: str):
    """
    Check that the response includes a thread starting indicator such as 'Thread:' or '1/' to represent the beginning of a tweet thread.
    """
    return "Thread:" in response or "1/" in response


async def assert_no_contradictory_remove_hashtags_statement(
    example: dict, prompt: str, response: str
):
    """
    Check that the response does not contain the contradictory statement 'REMOVE hashtags. DO USE THEM'.
    """
    return "REMOVE hashtags. DO USE THEM" not in response


async def assert_responses_tone_and_structure(
    example: dict, prompt: str, response: str
):
    """
    Check that the response follows the tone and structure implied by the instructions.
    """
    # This is a qualitative assessment and would need a more sophisticated natural language processing.
    question = "Does the response follow the tone and structure implied by the template instructions?"
    return await ask_llm(prompt, response, question)


async def assert_consistent_tone(example: dict, prompt: str, response: str):
    """
    Check if the tone of the response is consistent with the instructions given.
    """
    question = "Is the tone of the response consistent with the instructions given for a Twitter thread?"
    return await ask_llm(prompt, response, question)


# The function "assert_correct_hashtag_interpretation" is purposely omitted as there's no ambiguous instruction.
# Instead, there's a clear instruction to "REMOVE hashtags. DO NOT USE THEM.". Any ambiguous assertion should
# deal with genuinely unclear instructions, not clear ones like this.
async def assert_correct_number_of_tweets(example: dict, prompt: str, response: str):
    """
    Ensure the response contains the specified number of tweets.
    """
    number_of_tweets = example.get("number_of_tweets")
    response_tweets = response.strip().split(
        "\n\n"
    )  # Assuming tweets are separated by double newlines
    return len(response_tweets) == number_of_tweets


async def assert_adherence_to_word_limit(example: dict, prompt: str, response: str):
    """
    Check that each tweet in the response stays within the word limit per tweet.
    """
    word_count = example.get("word_count")
    response_tweets = response.strip().split("\n\n")

    for tweet in response_tweets:
        # Remove tweet indicators(e.g., "Tweet 1:") before counting words
        tweet_text = tweet.split(":", 1)[1] if ":" in tweet else tweet
        if len(tweet_text.split()) > word_count:
            return False
    return True


async def assert_no_hashtags_or_emojis_2(example: dict, prompt: str, response: str):
    """
    Confirm the response does not contain hashtags, emojis, or icons.
    """
    if "#" in response or any(char in emoji.EMOJI_DATA for char in response):
        return False
    return True


async def assert_mentions_used(example: dict, prompt: str, response: str):
    """
    Check that the response uses mentions to enhance visibility and engagement.
    """
    return "@" in response


async def assert_coherent_narrative(example: dict, prompt: str, response: str):
    """
    Ensure the response maintains a coherent narrative across the tweets.
    """
    question = "Does the response maintain a coherent narrative across the tweets?"
    return await ask_llm(prompt, response, question)


async def assert_correct_language_use(example: dict, prompt: str, response: str):
    """
    Confirm that the response is in the specified language.
    """
    language = example.get("language")
    question = f"Is the response written in {language}?"
    return await ask_llm(prompt, response, question)


async def assert_engagement_and_interaction_focus(
    example: dict, prompt: str, response: str
):
    """
    Check that the LLM response aims to create an engaging and informative Twitter thread encouraging interaction and sharing.
    """
    question = (
        "Does the Twitter thread encourage interaction and sharing among the target audience, "
        "while providing valuable insights?"
    )
    return await ask_llm(prompt, response, question)


async def assert_inclusion_of_context(example: dict, prompt: str, response: str):
    """
    Check that the response includes a transformation of the input context into tweets.
    """
    context = example.get("text")
    question = f"Has the context '{context}' been transformed into a Twitter thread effectively?"
    return await ask_llm(prompt, response, question)


async def assert_exclusion_of_specified_keywords(
    example: dict, prompt: str, response: str
):
    """
    Ensure that the response excludes any keywords or topics that the prompt template instructs to avoid.
    """
    # Since no specific keywords to exclude are provided in the example, it's assumed to determine from the prompt.
    exclusion_criteria = [
        "REMOVE hashtags. DO NOT USE THEM.",
        "REMOVE emojis nor icons.",
    ]
    return all(criteria not in response for criteria in exclusion_criteria)


async def assert_no_hashtags_emojis_icons(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response does not include hashtags, emojis, or icons.
    """
    return "#" not in response and not any(
        char for char in response if char in emoji.EMOJI_DATA
    )


async def assert_correct_tweet_format(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that each tweet in the response ends with a tweet number over the total (e.g., '1/5').
    This function assumes that each tweet is separated by a newline character.
    """
    tweets = response.strip().split("\n")
    number_of_tweets = example["number_of_tweets"]
    is_correct_format = True
    for index, tweet in enumerate(tweets, start=1):
        if not tweet.endswith(f"{index}/{number_of_tweets}"):
            is_correct_format = False
            break
    return is_correct_format and len(tweets) == number_of_tweets


async def assert_condensed_rephrased_sentences(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Ask an LLM whether the response is formatted as condensed and rephrased sentences.
    """
    question = "Is the response formatted as condensed and rephrased sentences?"
    return await ask_llm(prompt, response, question)


async def assert_audience_tone_maintenance(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Ask an LLM whether the response maintains the same audience tone as the context provides.
    """
    question = (
        "Does the response maintain the same audience tone as the context provides?"
    )
    return await ask_llm(prompt, response, question)


async def assert_proper_tweet_sequence_and_count(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Check that the response contains the correct number of tweets that follow the sequence pattern while keeping under the word limit per tweet.
    """
    number_of_tweets_expected = example["number_of_tweets"]
    word_limit_per_tweet = example["word_count"]
    tweets = response.split("\n")
    correct_tweet_count = True
    under_word_limit = True
    for index, tweet in enumerate(tweets, start=1):
        words = tweet.split()
        tweet_suffix = f"{index}/{number_of_tweets_expected}"
        if not tweet.endswith(tweet_suffix):
            return False
        if len(words) - len(tweet_suffix.split()) > word_limit_per_tweet:
            return False
    return len(tweets) == number_of_tweets_expected


async def assert_workflow_description(
    example: dict, prompt: str, response: str
) -> bool:
    """
    Ask an LLM if the workflow described in the prompt has been followed, focusing on the transformation and simplification of the provided information.
    """
    question = "Does the response follow the workflow described in the prompt, focusing on the transformation and simplification of the provided information?"
    return await ask_llm(prompt, response, question)


ALL_FUNCTIONS = [
    assert_concise_within_word_limit,
    assert_no_hashtags,
    assert_minimal_emojis_and_icons,
    assert_thread_format,
    assert_engaging_hook_main_tweet,
    assert_inclusion_of_mentions,
    assert_coherent_narrative_thread,
    assert_engaging_and_informative_thread,
    assert_proper_thread_start,
    assert_individual_tweets_format,
    assert_no_hashtags_or_emojis_1,
    assert_response_language,
    assert_response_length_within_limit,
    assert_maximum_five_tweets,
    assert_work_flow_actions_included,
    assert_thread_start_indicator,
    assert_no_contradictory_remove_hashtags_statement,
    assert_responses_tone_and_structure,
    assert_consistent_tone,
    assert_correct_number_of_tweets,
    assert_adherence_to_word_limit,
    assert_no_hashtags_or_emojis_2,
    assert_mentions_used,
    assert_coherent_narrative,
    assert_correct_language_use,
    assert_engagement_and_interaction_focus,
    assert_inclusion_of_context,
    assert_exclusion_of_specified_keywords,
    assert_no_hashtags_emojis_icons,
    assert_correct_tweet_format,
    assert_condensed_rephrased_sentences,
    assert_audience_tone_maintenance,
    assert_proper_tweet_sequence_and_count,
    assert_workflow_description,
]
