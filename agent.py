"""
agent.py
------------

This module defines functions and classes to build a simple
Language Model (LLM) agent that can answer questions about
economics and interpret insights derived from an exploratory
data analysis (EDA).  The agent uses the HuggingFace ``transformers``
pipeline to load a lightweight causal language model that can run
without GPU resources (``gpt2`` by default).  While ``gpt2`` is a small
model compared to state‑of‑the‑art chat systems, it is fully open
source and can be executed locally without an API key.  The agent
exposes a single function, :func:`generate_response`, which accepts
a prompt and optional context and returns a generated answer.

The design intentionally keeps the agent simple so that it can be
embedded within a Streamlit application and run within the
constrained hardware provided in this environment.  Should
additional compute resources become available, the model name
can be changed to a more capable community model (e.g.
``distilbert-base-uncased``, ``gpt2-medium``, or other instruction‑tuned
models) by modifying the ``MODEL_NAME`` constant.

Example usage::

    from agent import generate_response

    context = "Descriptive statistics: mean=1.5, std=0.3."
    answer = generate_response(
        question="¿Cuál es la tendencia de crecimiento económico?",
        context=context
    )
    print(answer)

``generate_response`` concatenates the context and the user's question
into a prompt, feeds it into the text‑generation pipeline and
returns the generated text.  A small maximum length (200 tokens) is
used to keep latency reasonable.

"""

from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Name of the HuggingFace model to load.  ``gpt2`` is light enough
# to run on CPU but still capable of producing coherent answers.
MODEL_NAME = "gpt2"

# Module‑level variables are initialised lazily so that the model
# weights are downloaded the first time ``generate_response`` is called.
_pipeline = None  # type: Optional[pipeline]


def _init_pipeline() -> pipeline:
    """Create and cache the HuggingFace text generation pipeline.

    Returns
    -------
    pipeline
        A HuggingFace text‑generation pipeline bound to the specified
        model and tokenizer.
    """
    global _pipeline
    if _pipeline is None:
        # Load tokenizer and model; use ``AutoModelForCausalLM`` so that
        # ``gpt2`` can be swapped for another causal language model if
        # desired.  Setting ``trust_remote_code=True`` allows the use of
        # community models that implement custom generation logic.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        _pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if model.device.type == "cuda" else -1,
        )
    return _pipeline


def generate_response(question: str, context: Optional[str] = None, max_length: int = 200) -> str:
    """Generate a response to a user's question using the language model.

    Parameters
    ----------
    question : str
        The user's question or instruction.
    context : str, optional
        Additional context (e.g. EDA summary) that will be prepended
        to the question.  Including context helps ground the model's
        responses in specific information about the data.
    max_length : int, default 200
        Maximum number of tokens to generate.  Smaller values reduce
        latency and prevent excessively long answers.

    Returns
    -------
    str
        The generated response truncated to remove the original prompt.
    """
    pipe = _init_pipeline()
    # Compose the final prompt.  The agent uses a simple instruction
    # format to differentiate between context and question.  Adding
    # separator tokens (\n\n) encourages the model to respond to the
    # question rather than continue the context.
    if context:
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    else:
        prompt = f"Question:\n{question}\n\nAnswer:"
    # Generate the output; ``do_sample=True`` enables sampling which
    # produces varied answers on repeated runs.  ``max_length`` is set
    # relative to the length of the input prompt.
    outputs = pipe(
        prompt,
        max_length=len(_init_pipeline().tokenizer.encode(prompt)) + max_length,
        do_sample=True,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
    )
    # Extract the text of the first sequence.  Remove the prompt
    # from the generated string to return only the answer.
    generated = outputs[0]["generated_text"]
    answer = generated[len(prompt) :].strip()
    return answer
