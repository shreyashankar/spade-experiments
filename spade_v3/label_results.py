import pandas as pd
import numpy as np


def label_responses(result_df: pd.DataFrame, function_success_threshold: float = 0.4):
    # 1 means good response, 0 means bad response

    function_means = (
        result_df.groupby("function_name")
        .agg({"result": "mean"})
        .reset_index()
        .sort_values("result", ascending=False)
    )
    good_functions = function_means[
        function_means["result"] > function_success_threshold
    ]["function_name"].tolist()
    print(f"Found {len(good_functions)} good functions: {good_functions}")

    responses_and_labels = (
        result_df.groupby("response")
        .apply(
            lambda x: len(
                set(x[x["result"] == False]["function_name"].tolist()).intersection(
                    set(good_functions)
                )
            )
            == 0
        )
        .astype(int)
        .reset_index()
        .rename(columns={0: "label"})
    )
    return responses_and_labels


def prepare_for_optimizer(result_df, label_df):
    # Compute M, y, K, etc
    # Compute cost vector
    cost = result_df.groupby("function_name")[
        ["prompt_tokens", "completion_tokens"]
    ].mean()

    # do prompt_tokens * 0.03 + completion_tokens * 0.01
    cost["cost"] = cost["prompt_tokens"] * 0.03 + cost["completion_tokens"] * 0.01
    cost.fillna(0, inplace=True)
    cost = cost["cost"].to_dict()

    func_order = {func: idx for idx, func in enumerate(cost.keys())}
    reverse_func_order = {idx: func for idx, func in enumerate(cost.keys())}
    cost_vector = np.zeros(len(cost))
    for func, idx in func_order.items():
        cost_vector[idx] = cost[func]

    # Create label dict
    y = label_df.set_index(["response"])["label"].to_dict()
    y_order = {prp: idx for idx, prp in enumerate(y.keys())}
    y_vector = np.zeros(len(y))
    for prp, idx in y_order.items():
        y_vector[idx] = y[prp]

    # Create function result dict
    M = np.zeros((len(y_vector), len(cost_vector)))
    for group, group_df in result_df.groupby("response"):
        y_idx = y_order[group]
        for func in cost.keys():
            func_idx = func_order[func]
            try:
                M[y_idx, func_idx] = (
                    group_df[group_df["function_name"] == func]["result"].values[0]
                ).astype(int)
            except:
                print(f"Error for {func}")
                pass

    return {
        "y": y_vector,
        "M": M,
        "cost": cost_vector,
        "func_order": func_order,
        "reverse_func_order": reverse_func_order,
        "y_order": y_order,
    }
