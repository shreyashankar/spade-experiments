from litellm import acompletion


async def ask_llm(prompt, response, question):
    # Placeholder for asking an expert a true/false question
    # In practice, this would involve a complex implementation potentially requiring human input

    messages = [
        {
            "content": f"You are an evaluator for a large language model chatbot that explains statistics concepts:\n\n{prompt}\n\nHere is the response:\n{response}",
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


async def presentation_format_assert(example: dict, prompt: str, response: str):
    return (
        response.startswith("Great!") and "please provide me with a summary" in response
    )


async def question_based_on_topic_assert(example: dict, prompt: str, response: str):
    return example["question"] in response and "?" in response


async def workflow_description_assert1(example: dict, prompt: str, response: str):
    return "summary of your understanding" in response and "feedback" in response


async def inquiry_and_feedback_presence_assert(
    example: dict, prompt: str, response: str
):
    question = (
        "Does the response contain an inquiry and imply that feedback will be provided?"
    )
    return await ask_llm(prompt, response, question)


async def keyword_inclusion_assert(example: dict, prompt: str, response: str):
    topic = example.get("question")
    return topic in response


async def subject_exclusivity_assert(example: dict, prompt: str, response: str):
    other_subjects = [
        "biology",
        "chemistry",
        "physics",
        "mathematics",
        "history",
        "geography",
    ]
    return not any(subject in response for subject in other_subjects)


async def language_professionalism_assert(example: dict, prompt: str, response: str):
    question = "Is the language used in the response professional and gentle, as a teacher's should be?"
    return await ask_llm(prompt, response, question)


async def correctness_completeness_consistency_assert(
    example: dict, prompt: str, response: str
):
    question = "Is the response correctly following the instructions and does it seem complete and consistent?"
    return await ask_llm(prompt, response, question)


async def workflow_description_assert2(example: dict, prompt: str, response: str):
    should_contain = [
        "please provide me with a summary of your current understanding",
        "Feedback:",
        "Now, let's explore",
        "Your task is to",
    ]
    return all(step in response for step in should_contain)


async def example_filled_with_question_assert(
    example: dict, prompt: str, response: str
):
    topic = prompt.splitlines()[-1]
    return topic in response


async def feedback_inclusion_assert(example: dict, prompt: str, response: str):
    return "Feedback:" in response


async def word_count_assert(example: dict, prompt: str, response: str):
    return 150 <= len(response.split()) <= 250


async def toy_dataset_and_problem_inclusion_assert(
    example: dict, prompt: str, response: str
):
    dataset_keywords = [
        "dataset",
        "survey",
        "employees",
        "company",
        "calculate",
        "compute",
        "assess",
    ]
    return any(keyword in response for keyword in dataset_keywords)


async def willing_to_accept_new_summary_assert(
    example: dict, prompt: str, response: str
):
    expected_phrases = [
        "At any point during our discussion",
        "please provide me with an updated summary",
        "we can continue to delve deeper",
    ]
    return any(phrase in response for phrase in expected_phrases)


async def workflow_completed_correctly_assert(
    example: dict, prompt: str, response: str
):
    question = (
        "Does the LLM's response correctly complete the workflow described in the prompt, "
        "including providing feedback on the user's understanding and being willing to accept a new summary?"
    )
    return await ask_llm(prompt, response, question)


ALL_FUNCTIONS = [
    presentation_format_assert,
    question_based_on_topic_assert,
    workflow_description_assert1,
    inquiry_and_feedback_presence_assert,
    keyword_inclusion_assert,
    subject_exclusivity_assert,
    language_professionalism_assert,
    correctness_completeness_consistency_assert,
    workflow_description_assert2,
    example_filled_with_question_assert,
    feedback_inclusion_assert,
    word_count_assert,
    toy_dataset_and_problem_inclusion_assert,
    willing_to_accept_new_summary_assert,
    workflow_completed_correctly_assert,
]
