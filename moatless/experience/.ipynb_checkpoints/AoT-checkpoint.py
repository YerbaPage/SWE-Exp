def multistep(question: str):
    instruction = """
You are a precise bug fixer. Fix the given repository-level bug step by step:

QUESTION:\n {question}

Please extend your chain of thought as much as possible; the longer the chain of thought, the better.

You can freely reason in your response.

Output:
    """
    prompt = instruction.format(question=question)
    return prompt

def label(question: str, chain: str):
    instruction = """
You are tasked with breaking down a repository-level bug solution reasoning process into sub-questions.

Original Question: {question}
Complete Reasoning Process: {chain}

Instructions:
1. Break down the reasoning process into a series of sub-questions
2. Each sub-question should:
   - Be written in interrogative form
   - List its other sub-questions' indexes it depends (0-based, can be an empty list)
3. Dependencies are defined as information needed to answer the current sub-question that:
   - Does NOT come directly from the original question
   - MUST come from the answers of previous sub-questions
    """
    formatter = """
Format your response as the following JSON object:
{
    "sub-questions": [
        {
            "description": "<clear interrogative question>",
            "depend": [<indices of prerequisite sub-questions>]
        },
        ...
    ]
}

Output:
    """
    return (instruction + formatter).format(question=question, chain=chain)