"""
This file optimizes the functions based on the coverage and false failure rate.
"""

import pulp
import pickle

from rich import print

import time

# from spade_v2.check_subsumes import subsumes

# Assuming you have the data for 'm', 'n', 'M', 'y', 'alpha' and 'tau'
# m: number of functions
# n: number of examples
# M: matrix of dimensions m x n (success/failure indication)
# y: array indicating success (1) or failure (0) for each example
# tau: threshold for false failure rate
# alpha: threshold for coverage
# func_order: dictionary mapping function names to indices


def calculate_ffr(selected_functions, M, y):
    """
    Calculates the false failure rate for a given set of selected functions.
    """
    # Calculate the numerator and denominator of the FFR
    numerator = sum(
        1
        for i in range(len(y))
        if y[i] == 1 and any(M[i][j] == 0 for j in selected_functions)
    )
    denominator = sum([1 for i in range(len(y)) if y[i] == 1])

    # Calculate the FFR
    if denominator > 0:
        ffr = numerator / denominator
    else:
        ffr = 0  # Handle division by zero

    return ffr


def calculate_coverage(selected_functions, M, y):
    coverage_count = 0
    denominator = 0

    # Get list of all functions with FFR <= tau
    # all_functions = [i for i in range(len(M[0])) if calculate_ffr([i], M, y) <= tau]

    for i in range(len(y)):
        if y[i] == 0:
            denominator += 1
            for k in selected_functions:
                if M[i][k] == 0:
                    coverage_count += 1
                    break

    coverage_percentage = (coverage_count / denominator) if denominator > 0 else 0
    return coverage_percentage


def optimize_example_coverage_only(
    m, n, M, K, y, tau, alpha, spade_functions, func_order
):
    """This function optimizes functions based on coverage of examples only and FFr.

    Args:
        m (_type_): _description_
        n (_type_): _description_
        M (_type_): _description_
        K (_type_): _description_
        y (_type_): _description_
        tau (_type_): _description_
        alpha (_type_): _description_
        spade_functions (_type_): _description_
        func_order (_type_): _description_
    """
    problem = pulp.LpProblem("SPADE_Function_Selection_Cov", pulp.LpMinimize)

    # Decision Variables
    x = pulp.LpVariable.dicts("x", range(m), cat="Binary")
    w = pulp.LpVariable.dicts(
        "w", [(i, j) for i in range(n) for j in range(m)], cat="Binary"
    )
    z = pulp.LpVariable.dicts("z", range(n), cat="Binary")
    u = pulp.LpVariable.dicts("u", range(n), cat="Binary")

    # Objective Function
    problem += pulp.lpSum([x[j] for j in range(m)])

    # Define w and z
    for i in range(n):
        for j in range(m):
            problem += w[i, j] == (1 - M[i][j]) * x[j]
            problem += z[i] >= y[i] * w[i, j]

    # Constraints
    # Coverage Constraint
    for i in range(n):
        if y[i] == 0:
            problem += u[i] <= pulp.lpSum([w[i, j] for j in range(m)])

    problem += pulp.lpSum([u[i] for i in range(n) if y[i] == 0]) >= alpha * sum(
        y[i] == 0 for i in range(n)
    )

    # FFR Constraint
    problem += pulp.lpSum([z[i] for i in range(n) if y[i] == 1]) <= tau * pulp.lpSum(
        [1 for i in range(n) if y[i] == 1]
    )

    # Solve the problem
    problem.solve()

    # Output the results
    if problem.status == pulp.LpStatusOptimal:
        print("Solution Found:")
        selected_functions = [j for j in range(m) if x[j].varValue == 1]
        print(f"Selected Functions: {selected_functions}")

        # Calculate the coverage
        coverage_count = sum(u[i].varValue for i in range(n) if y[i] == 0)
        total_failures = sum(1 for i in range(n) if y[i] == 0)
        coverage_percentage = (
            (coverage_count / total_failures) * 100 if total_failures > 0 else 0
        )
        # print(f"Coverage: {coverage_percentage:.2f}% of failure examples")
        coverage_percentage = calculate_coverage(selected_functions, M, y)

        # print(f"Coverage: {(coverage_percentage * 100):.2f}% of failure examples")
        ffr = calculate_ffr(selected_functions, M, y)

        # print(f"False Failure Rate (FFR): {ffr:.2f}")

        rev_func_order = {v: k for k, v in func_order.items()}

        not_subsumed_excluded_functions = []

        for j in range(m):
            # Compute any of the excluded functions that are not subsumed

            if j not in selected_functions:
                function_ffr = calculate_ffr([j], M, y)
                function_included_ffr = calculate_ffr(selected_functions + [j], M, y)
                # print(
                #     f"{rev_func_order[j]} is not selected but not subsumed. It has a FFR of {function_ffr:.2f}"
                # )

                if function_included_ffr <= tau:
                    not_subsumed_excluded_functions.append(
                        {
                            "func_name": rev_func_order[j],
                            "ffr": function_ffr,
                            "function_included_ffr": function_included_ffr,
                            "idx": j,
                        }
                    )

        # Compute the fraction of functions selected
        frac_functions_selected = len(selected_functions) / len(spade_functions)

        return (
            selected_functions,
            coverage_percentage,
            ffr,
            frac_functions_selected,
            not_subsumed_excluded_functions,
        )
    else:
        raise ValueError("No optimal solution found.")


def optimize(m, n, M, K, y, tau, alpha, spade_functions, func_order):
    """
    This function optimizes the functions based on the coverage and false failure rate and subsumption.
    """

    candidate_functions = []
    for j in range(m):
        if calculate_ffr([j], M, y) <= tau:
            candidate_functions.append(j)

    # print(
    #     f"Number of candidate functions (don't exceed FFR threshold): {len(candidate_functions)}/{m}"
    # )

    # Initialize the problem
    problem = pulp.LpProblem("SPADE_Function_Selection", pulp.LpMinimize)

    # Decision Variables
    x = pulp.LpVariable.dicts("x", range(m), cat="Binary")
    w = pulp.LpVariable.dicts(
        "w", [(i, j) for i in range(n) for j in range(m)], cat="Binary"
    )
    z = pulp.LpVariable.dicts("z", range(n), cat="Binary")
    u = pulp.LpVariable.dicts("u", range(n), cat="Binary")

    # include_pairs = pulp.LpVariable.dicts(
    #     "include_pairs", [(i, j) for i in range(m) for j in range(m)], cat="Binary"
    # )
    s = pulp.LpVariable.dicts(
        "s", [(i, j) for i in range(m) for j in range(m)], cat="Binary"
    )
    subsumption_penalty = pulp.LpVariable.dicts(
        "subsumption_penalty", range(m), cat="Binary"
    )
    is_subsumed = pulp.LpVariable.dicts("is_subsumed", range(m), cat="Binary")

    # Add Constraints for include_pairs
    # for i in range(m):
    #     for j in range(m):
    #         if i == j:
    #             # Force s[i, i] to be 0
    #             problem += include_pairs[i, j] == 0

    #         else:
    #             problem += include_pairs[i, j] <= x[i]
    #             problem += include_pairs[i, j] <= (1 - x[j])
    #             problem += include_pairs[i, j] >= x[i] + (1 - x[j]) - 1

    # Define s as include_pairs and K
    for i in range(m):
        for j in range(m):
            # problem += s[i, j] == include_pairs[i, j] * K[i, j]
            problem += s[i, j] == x[i] * K[i, j]

    # Define is_subsumed
    for j in range(m):
        problem += is_subsumed[j] <= pulp.lpSum([s[i, j] for i in range(m) if i != j])
        for i in range(m):
            if i != j:
                problem += is_subsumed[j] >= s[i, j]

    # Add Constraints for subsumption_penalty
    for j in range(m):
        problem += subsumption_penalty[j] <= (1 - x[j])

        problem += subsumption_penalty[j] <= (1 - is_subsumed[j])
        problem += subsumption_penalty[j] >= (1 - x[j]) + (1 - is_subsumed[j]) - 1

    # Objective Function
    problem += pulp.lpSum([x[j] for j in range(m)])

    # Minimize the number of functions not selected that are not subsumed by one of the selected functions
    problem += pulp.lpSum([subsumption_penalty[j] for j in range(m)])

    # Define w and z
    for i in range(n):
        for j in range(m):
            problem += w[i, j] == (1 - M[i][j]) * x[j]
            problem += z[i] >= y[i] * w[i, j]

    # Constraints
    # Coverage Constraint
    for i in range(n):
        if y[i] == 0:
            problem += u[i] <= pulp.lpSum([w[i, j] for j in range(m)])

    problem += pulp.lpSum([u[i] for i in range(n) if y[i] == 0]) >= alpha * sum(
        y[i] == 0 for i in range(n)
    )

    # FFR Constraint
    problem += pulp.lpSum([z[i] for i in range(n) if y[i] == 1]) <= tau * pulp.lpSum(
        [1 for i in range(n) if y[i] == 1]
    )

    # Solve the problem
    problem.solve()

    # Output the results
    if problem.status == pulp.LpStatusOptimal:
        print("Solution Found:")
        selected_functions = [j for j in range(m) if x[j].varValue == 1]
        print(f"Selected Functions: {selected_functions}")

        # Calculate the coverage
        coverage_count = sum(u[i].varValue for i in range(n) if y[i] == 0)
        total_failures = sum(1 for i in range(n) if y[i] == 0)
        coverage_percentage = (
            (coverage_count / total_failures) * 100 if total_failures > 0 else 0
        )
        # print(f"Coverage: {coverage_percentage:.2f}% of failure examples")
        coverage_percentage = calculate_coverage(selected_functions, M, y)

        # print(f"Coverage: {(coverage_percentage * 100):.2f}% of failure examples")
        ffr = calculate_ffr(selected_functions, M, y)

        # print(f"False Failure Rate (FFR): {ffr:.2f}")

        # Compute eliminated functions
        eliminated_functions = [
            j for j in range(m) if x[j].varValue == 0 and j in candidate_functions
        ]

        # Print s variables's values
        rev_func_order = {v: k for k, v in func_order.items()}
        distinct_subsumed_functions = set()

        for i in range(m):
            for j in range(m):
                if s[i, j].varValue == 1:
                    # print(
                    #     f"{rev_func_order[i]} (selected) -> {rev_func_order[j]} (not selected)"
                    # )
                    distinct_subsumed_functions.add(j)

        # Assert is_subsumed == distinct_subsumed_functions
        assert sum([is_subsumed[j].varValue for j in range(m)]) == len(
            distinct_subsumed_functions
        )

        not_subsumed_excluded_functions = []

        for j in range(m):
            if subsumption_penalty[j].varValue == 1:
                function_ffr = calculate_ffr([j], M, y)
                function_included_ffr = calculate_ffr(selected_functions + [j], M, y)
                # print(
                #     f"{rev_func_order[j]} is not selected but not subsumed. It has a FFR of {function_ffr:.2f}"
                # )

                if function_included_ffr <= tau:
                    not_subsumed_excluded_functions.append(
                        {
                            "func_name": rev_func_order[j],
                            "ffr": function_ffr,
                            "function_included_ffr": function_included_ffr,
                            "idx": j,
                        }
                    )

        # Compute the fraction of functions selected
        frac_functions_selected = len(selected_functions) / len(spade_functions)

        return (
            selected_functions,
            coverage_percentage,
            ffr,
            frac_functions_selected,
            not_subsumed_excluded_functions,
        )
    else:
        raise ValueError("No optimal solution found.")


# Function to run the pipeline
def select_functions(filename: str, tau=0.25, alpha=0.9, track_time: bool = False):
    with open(filename, "rb") as f:
        optimizer_input = pickle.load(f)
        y = optimizer_input["y"]
        cost_vector = optimizer_input["cost"]
        func_order = optimizer_input["func_order"]
        spade_functions = optimizer_input["spade_functions"]
        M = optimizer_input["M"]
        K = optimizer_input["K"]

    n = len(y)
    m = len(cost_vector)

    # Compute FFR and coverage for total set of functions
    eligible_functions = [j for j in range(m) if calculate_ffr([j], M, y) <= tau]
    ffr_eligible_functions = calculate_ffr(eligible_functions, M, y)
    coverage_eligible_functions = calculate_coverage(eligible_functions, M, y)

    # Print the FFR and coverage for the total set of functions
    # print(f"FFR for eligible functions: {ffr_eligible_functions:.2f}")
    # print(f"Coverage for eligible functions: {coverage_eligible_functions:.2f}")
    # print(f"Frac eligible functions selected: {(len(eligible_functions) / m):.2f}")
    eligible_function_names = [
        name for name, idx in func_order.items() if idx in eligible_functions
    ]
    # print(f"Eligible functions: {eligible_function_names}")

    runtimes = {}
    if track_time:
        start = time.time()

    # Run cov optimizer
    (
        cov_selected_functions,
        cov_coverage_percentage,
        cov_ffr,
        cov_frac_functions_selected,
        cov_not_subsumed_excluded_functions,
    ) = optimize_example_coverage_only(
        m, n, M, K, y, tau, alpha, spade_functions, func_order
    )

    if track_time:
        runtimes["cov"] = time.time() - start
        start = time.time()

    (
        selected_functions,
        coverage_percentage,
        ffr,
        frac_functions_selected,
        not_subsumed_excluded_functions,
    ) = optimize(m, n, M, K, y, tau, alpha, spade_functions, func_order)

    if track_time:
        runtimes["sub"] = time.time() - start

    # Print name of selected functions
    selected_function_names = [
        name for name, idx in func_order.items() if idx in selected_functions
    ]
    # print(
    #     f"Selected {len(selected_functions) / len(cost_vector) * 100}% functions: {selected_function_names}"
    # )

    # Return various solutions
    spade_base = {
        "selected_functions": eligible_functions,
        "selected_function_names": eligible_function_names,
        "ffr": ffr_eligible_functions,
        "coverage": coverage_eligible_functions,
        "frac_functions_selected": float(len(eligible_functions) / m),
        "not_subsumed_excluded_functions": [],
        "frac_non_subsumed_excluded_functions": 0.0,
    }

    spade_cov = {
        "selected_functions": cov_selected_functions,
        "selected_function_names": [
            name for name, idx in func_order.items() if idx in cov_selected_functions
        ],
        "ffr": cov_ffr,
        "coverage": cov_coverage_percentage,
        "frac_functions_selected": cov_frac_functions_selected,
        "not_subsumed_excluded_functions": cov_not_subsumed_excluded_functions,
        "frac_non_subsumed_excluded_functions": len(cov_not_subsumed_excluded_functions)
        / m,
    }

    spade_sub = {
        "selected_functions": selected_functions,
        "selected_function_names": selected_function_names,
        "ffr": ffr,
        "coverage": coverage_percentage,
        "frac_functions_selected": frac_functions_selected,
        "not_subsumed_excluded_functions": not_subsumed_excluded_functions,
        "frac_non_subsumed_excluded_functions": len(not_subsumed_excluded_functions)
        / m,
    }

    return {
        "spade_base": spade_base,
        "spade_cov": spade_cov,
        "spade_sub": spade_sub,
        "tau": tau,
        "alpha": alpha,
        "runtimes": runtimes,
    }
