import re
from enum import Enum
from typing import Literal, TypedDict

from typing_extensions import NotRequired

from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend
from mellea.backends.types import ModelOption

from .prompt_modules import (
    constraint_extractor,
    # general_instructions,
    subtask_constraint_assign,
    subtask_list,
    subtask_prompt_generator,
    validation_decision,
)
from .prompt_modules.subtask_constraint_assign import SubtaskPromptConstraintsItem
from .prompt_modules.subtask_list import SubtaskItem
from .prompt_modules.subtask_prompt_generator import SubtaskPromptItem


class ConstraintResult(TypedDict):
    constraint: str
    validation_strategy: str


class DecompSubtasksResult(TypedDict):
    subtask: str
    tag: str
    constraints: list[ConstraintResult]
    prompt_template: str
    # general_instructions: str
    input_vars_required: list[str]
    depends_on: list[str]
    generated_response: NotRequired[str]


class DecompPipelineResult(TypedDict):
    original_task_prompt: str
    subtask_list: list[str]
    identified_constraints: list[ConstraintResult]
    subtasks: list[DecompSubtasksResult]
    final_response: NotRequired[str]


class DecompBackend(str, Enum):
    ollama = "ollama"
    openai = "openai"
    rits = "rits"
    claude = "claude"


RE_JINJA_VAR = re.compile(r"\{\{\s*(.*?)\s*\}\}")


def decompose(
    task_prompt: str,
    user_input_variable: list[str] | None = None,
    model_id: str = "mistral-small3.2:latest",
    backend: DecompBackend = DecompBackend.ollama,
    backend_req_timeout: int = 300,
    backend_endpoint: str | None = None,
    backend_api_key: str | None = None,
    subtask_list_template: str | None = None,
    custom_icl_examples: list[dict] | None = None,
    constraint_extractor_template: str | None = None,
    constraint_extractor_icl_examples: list[dict] | None = None,
    last_subtask_system_template: str | None = None,
    last_subtask_user_template: str | None = None,
) -> DecompPipelineResult:
    if user_input_variable is None:
        user_input_variable = []

    # region Backend Assignment
    match backend:
        case DecompBackend.ollama:
            m_session = MelleaSession(
                OllamaModelBackend(
                    model_id=model_id, model_options={ModelOption.CONTEXT_WINDOW: 16384}
                )
            )
        case DecompBackend.openai:
            assert backend_endpoint is not None, (
                'Required to provide "backend_endpoint" for this configuration'
            )
            assert backend_api_key is not None, (
                'Required to provide "backend_api_key" for this configuration'
            )
            m_session = MelleaSession(
                OpenAIBackend(
                    model_id=model_id,
                    base_url=backend_endpoint,
                    api_key=backend_api_key,
                    model_options={"timeout": backend_req_timeout},
                )
            )
        case DecompBackend.rits:
            assert backend_endpoint is not None, (
                'Required to provide "backend_endpoint" for this configuration'
            )
            assert backend_api_key is not None, (
                'Required to provide "backend_api_key" for this configuration'
            )

            from mellea_ibm.rits import RITSBackend, RITSModelIdentifier  # type: ignore

            m_session = MelleaSession(
                RITSBackend(
                    RITSModelIdentifier(endpoint=backend_endpoint, model_name=model_id),
                    api_key=backend_api_key,
                    model_options={"timeout": backend_req_timeout},
                )
            )
        case DecompBackend.claude:
            assert backend_api_key is not None, (
                'Required to provide "backend_api_key" for this configuration'
            )
            # LiteLLM uses the ANTHROPIC_API_KEY environment variable by default,
            # but we'll set it explicitly here if provided
            import os

            os.environ["ANTHROPIC_API_KEY"] = backend_api_key

            m_session = MelleaSession(
                LiteLLMBackend(
                    model_id=model_id,
                    base_url=backend_endpoint,  # Optional, only needed for custom endpoints
                    model_options={"timeout": backend_req_timeout},
                )
            )
    # endregion

    subtasks: list[SubtaskItem] = subtask_list.generate(
        m_session, 
        task_prompt, 
        custom_template=subtask_list_template,
        custom_icl_examples=custom_icl_examples
    ).parse()

    # Generate subtask prompts FIRST (before constraint extraction)
    subtask_prompts: list[SubtaskPromptItem] = subtask_prompt_generator.generate(
        m_session,
        task_prompt,
        user_input_var_names=user_input_variable,
        subtasks_and_tags=subtasks,
        last_subtask_system_template=last_subtask_system_template,
        last_subtask_user_template=last_subtask_user_template,
    ).parse()

    # NEW APPROACH: Generate constraints PER SUBTASK using hybrid method
    # This considers both the original task prompt and the specific subtask context
    subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem] = []
    all_constraints_dict: dict[str, Literal["code", "llm"]] = {}  # Track all unique constraints
    
    for subtask_prompt in subtask_prompts:
        # Extract constraints for THIS specific subtask using hybrid approach
        subtask_constraints: list[str] = constraint_extractor.generate(
            m_session,
            input_str=None,  # Not used in hybrid mode
            enforce_same_words=False,
            custom_template=constraint_extractor_template or "per_subtask_hybrid",
            custom_icl_examples=constraint_extractor_icl_examples,
            custom_user_template=constraint_extractor_template or "per_subtask_hybrid",
            original_task_prompt=task_prompt,
            subtask_description=subtask_prompt.subtask,
            subtask_prompt=subtask_prompt.prompt_template,
        ).parse()
        
        # Determine validation strategy for each constraint
        for constraint in subtask_constraints:
            if constraint not in all_constraints_dict:
                # Only call validation_decision once per unique constraint
                all_constraints_dict[constraint] = validation_decision.generate(
                    m_session, constraint
                ).parse()
        
        # Build SubtaskPromptConstraintsItem
        subtask_prompts_with_constraints.append(
            SubtaskPromptConstraintsItem(
                subtask=subtask_prompt.subtask,
                tag=subtask_prompt.tag,
                prompt_template=subtask_prompt.prompt_template,
                constraints=subtask_constraints,
            )
        )

    decomp_subtask_result: list[DecompSubtasksResult] = [
        DecompSubtasksResult(
            subtask=subtask_data.subtask,
            tag=subtask_data.tag,
            constraints=[
                {
                    "constraint": cons_str,
                    "validation_strategy": all_constraints_dict[cons_str],
                }
                for cons_str in subtask_data.constraints
            ],
            prompt_template=subtask_data.prompt_template,
            # general_instructions=general_instructions.generate(
            #     m_session, input_str=subtask_data.prompt_template
            # ).parse(),
            input_vars_required=list(
                dict.fromkeys(  # Remove duplicates while preserving the original order.
                    [
                        item
                        for item in re.findall(
                            RE_JINJA_VAR, subtask_data.prompt_template
                        )
                        if item in user_input_variable
                    ]
                )
            ),
            depends_on=list(
                dict.fromkeys(  # Remove duplicates while preserving the original order.
                    [
                        item
                        for item in re.findall(
                            RE_JINJA_VAR, subtask_data.prompt_template
                        )
                        if item not in user_input_variable
                    ]
                )
            ),
        )
        for subtask_data in subtask_prompts_with_constraints
    ]

    return DecompPipelineResult(
        original_task_prompt=task_prompt,
        subtask_list=[item.subtask for item in subtasks],
        identified_constraints=[
            {
                "constraint": cons_str,
                "validation_strategy": validation_strategy,
            }
            for cons_str, validation_strategy in all_constraints_dict.items()
        ],
        subtasks=decomp_subtask_result,
    )
