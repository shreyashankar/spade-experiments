# SPADE (System for Prompt Analysis and Delta-Based Evaluation)

This is the code for the paper SPADE: Auto-Generated Assertions for Large Language Model
Pipelines.

The SPADE pipeline has the following steps:

1. Decomposing a list of prompts and an example into a set of prompt deltas, assertion concepts, and candidate assertions. This is in `candidate_gen.py`.

2. Generating a set of candidate assertions for each prompt delta. This is also in `candidate_gen.py` (uses a helper function in `assertion_gen.py`).

3. Generates synthetic data and evaluates the assertions (`execute_assertions.py`). The user should label the synthetic data responses as correct or incorrect.

4. Evaluates subsumption of the assertions (`check_subsumes.py`).

5. Runs the assertion selection algorithm (`optimizer.py``).

All the experiments live in paper_experiments/ and have a notebook for each experiment. To run any of the experiments (leveraging cached data so you don't need to run any LLMs), run the `pipeline.ipynb` notebook in the root directory, using the correct experiment name.
