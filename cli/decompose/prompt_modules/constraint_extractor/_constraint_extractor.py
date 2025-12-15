import re
from collections.abc import Callable
from typing import Any, TypeVar, final

from mellea import MelleaSession
from mellea.backends.types import ModelOption
from mellea.stdlib.chat import Message

from .._prompt_modules import PromptModule, PromptModuleString
from ._exceptions import BackendGenerationError, TagExtractionError
from ._prompt import get_system_prompt, get_user_prompt

T = TypeVar("T")

RE_VERIFIED_CONS_COND = re.compile(
    r"<constraints_and_requirements>(.+?)</constraints_and_requirements>",
    flags=re.IGNORECASE | re.DOTALL,
)


@final
class _ConstraintExtractor(PromptModule):
    @staticmethod
    def _default_parser(generated_str: str) -> list[str]:
        r"""Default parser of the `constraint_extractor` module.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and plan for unreliable results._

        Args:
            generated_str (`str`): The LLM's answer to be parsed.

        Returns:
            list[str]: A list of identified constraints and requirements in natural language. The list
            will be empty if no constraints were identified by the LLM.

        Raises:
            TagExtractionError: An error occurred trying to extract content from the
                generated output. The LLM probably failed to open and close
                the \<constraints_and_requirements\> tags.
        """
        constraint_extractor_match = re.search(RE_VERIFIED_CONS_COND, generated_str)

        constraint_extractor_str: str | None = (
            constraint_extractor_match.group(1).strip()
            if constraint_extractor_match
            else None
        )

        if constraint_extractor_str is None:
            raise TagExtractionError(
                'LLM failed to generate correct tags for extraction: "<constraints_and_requirements>"'
            )

        # TODO: Maybe replace this logic with a RegEx?
        constraint_extractor_str_upper = constraint_extractor_str.upper()
        if (
            "N/A" in constraint_extractor_str_upper
            or "N / A" in constraint_extractor_str_upper
            or "N/ A" in constraint_extractor_str_upper
            or "N /A" in constraint_extractor_str_upper
        ):
            return []

        return [
            line.strip()[2:] if line.strip()[:2] == "- " else line.strip()
            for line in constraint_extractor_str.splitlines()
        ]

    def generate(  # type: ignore[override]
        # About the mypy ignore above:
        # Since the extra argument has a default value, it should be safe to override.
        # It doesn't violate the Liskov Substitution Principle, but mypy doesn't like it.
        self,
        mellea_session: MelleaSession,
        input_str: str | None,
        max_new_tokens: int = 4096,
        parser: Callable[[str], T] = _default_parser,  # type: ignore[assignment]
        # About the mypy ignore above: https://github.com/python/mypy/issues/3737
        enforce_same_words: bool = False,
        **kwargs: dict[str, Any],
    ) -> PromptModuleString[T]:
        """Generates an unordered list of identified constraints and requirements based on a provided task prompt.

        _**Disclaimer**: This is a LLM-prompting module, so the results will vary depending
        on the size and capabilities of the LLM used. The results are also not guaranteed, so
        take a look at this module's Exceptions and use them accordingly._

        Args:
            mellea_session (`MelleaSession`): A mellea session with a backend.
            input_str (`str`): Natural language (non-templated) prompt describing a task to be executed.
            max_new_tokens (`int`, optional): Maximum tokens to generate.
                Try increasing the value if you are getting `TagExtractionError`.
                Defaults to `8192`.
            parser (`Callable[[str], Any]`, optional): A string parsing function.
                Defaults to `_ConstraintExtractor._default_parser`.
            enforce_same_words (`bool`, optional): Whether to enforce using same words from original prompt.
                Defaults to `False`.
            **kwargs: Additional keyword arguments:
                - custom_template (`str`, optional): Name of custom template to use instead of default.
                - custom_icl_examples (`list[dict]`, optional): Custom ICL examples to use instead of default.
                - custom_user_template (`str`, optional): Name of custom user template (for hybrid per-subtask approach).
                - original_task_prompt (`str`, optional): Original task prompt (for hybrid per-subtask approach).
                - subtask_description (`str`, optional): Subtask title/description (for hybrid per-subtask approach).
                - subtask_prompt (`str`, optional): Subtask prompt template (for hybrid per-subtask approach).

        Returns:
            PromptModuleString: A `PromptModuleString` class containing the generated output.

            The `PromptModuleString` class behaves like a `str`, but with an additional `parse()` method
            to execute the parsing function passed in the `parser` argument of
            this method (the `parser` argument defaults to `_ConstraintExtractor._default_parser`).

        Raises:
            BackendGenerationError: Some error occurred during the LLM generation call.
        """
        # Extract custom parameters from kwargs if provided
        custom_template = kwargs.get('custom_template', None)
        custom_icl_examples = kwargs.get('custom_icl_examples', None)
        custom_user_template = kwargs.get('custom_user_template', None)
        
        # Extract hybrid approach parameters (for per-subtask constraint extraction)
        original_task_prompt = kwargs.get('original_task_prompt', None)
        subtask_description = kwargs.get('subtask_description', None)
        subtask_prompt = kwargs.get('subtask_prompt', None)
        
        # Validate input
        if custom_user_template:
            # Hybrid approach: requires original_task_prompt, subtask_description, subtask_prompt
            if not all([original_task_prompt, subtask_description, subtask_prompt]):
                raise ValueError(
                    "When using custom_user_template, must provide: "
                    "original_task_prompt, subtask_description, subtask_prompt"
                )
        else:
            # Standard approach: requires input_str
            assert input_str is not None, 'This module requires the "input_str" argument'
        
        system_prompt = get_system_prompt(
            enforce_same_words=enforce_same_words,
            custom_template=custom_template,
            custom_icl_examples=custom_icl_examples
        )
        
        user_prompt = get_user_prompt(
            task_prompt=input_str,
            custom_user_template=custom_user_template,
            original_task_prompt=original_task_prompt,
            subtask_description=subtask_description,
            subtask_prompt=subtask_prompt,
        )

        action = Message("user", user_prompt)

        try:
            gen_result = mellea_session.act(
                action=action,
                model_options={
                    ModelOption.SYSTEM_PROMPT: system_prompt,
                    ModelOption.TEMPERATURE: 0,
                    ModelOption.MAX_NEW_TOKENS: max_new_tokens,
                },
            ).value
        except Exception as e:
            raise BackendGenerationError(f"LLM generation failed: {e}")

        if gen_result is None:
            raise BackendGenerationError(
                "LLM generation failed: value attribute is None"
            )

        return PromptModuleString(gen_result, parser)


constraint_extractor = _ConstraintExtractor()
