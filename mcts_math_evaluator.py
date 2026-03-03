"""Math evaluator for Marco-o1 MCTS tree search.

Drop-in replacement for the stub math_evaluator in
Marco-o1/src/v2/src/tree_search/evaluator/evaluator.py.

Reuses extract_model_answer + check_correctness from reward_func_em.py
to evaluate MCTS answer nodes.

Interface: math_evaluator(answer_node, ground_truth) -> int (1 or 0)
"""

from reward_func_em import extract_model_answer, check_correctness


def math_evaluator(answer_node, ground_truth) -> int:
    """Evaluate an MCTS answer node against ground truth.

    Args:
        answer_node: MCTS AnswerNode with .node_value containing the
                     <answer>...</answer> text.
        ground_truth: Expected answer (string or number).

    Returns:
        1 if correct, 0 if incorrect.
    """
    text = answer_node.node_value

    # Guard against tag spam (same as count_latter_evaluator)
    if text.count("<") > 4:
        return 0

    model_answer = extract_model_answer(text)
    if not model_answer:
        return 0

    return 1 if check_correctness(model_answer, str(ground_truth)) == 1.0 else 0
