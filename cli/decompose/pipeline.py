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
    watsonx = "watsonx"
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
    backend_project_id: str | None = None,
    max_new_tokens: int = 4096,
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
        case DecompBackend.watsonx:
            assert backend_api_key is not None, (
                'Required to provide "backend_api_key" for watsonx configuration'
            )
            assert backend_endpoint is not None, (
                'Required to provide "backend_endpoint" (for watsonx configuration)'
            )
            assert backend_project_id is not None, (
                'Required to provide "backend_project_id" for watsonx configuration'
            )
         
            
            from mellea.backends.watsonx import WatsonxAIBackend
            
            m_session = MelleaSession(
                WatsonxAIBackend(
                    model_id=model_id,
                    api_key=backend_api_key,
                    base_url=backend_endpoint,  
                    model_options={"timeout": backend_req_timeout},
                    project_id=backend_project_id,
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

    print("🔄 Step 1/4: Generating subtasks...")
    subtasks: list[SubtaskItem] = subtask_list.generate(
        m_session, 
        task_prompt, 
        max_new_tokens=max_new_tokens,
        custom_template=subtask_list_template,
        custom_icl_examples=custom_icl_examples
    ).parse()
    print(f"   ✓ Generated {len(subtasks)} subtasks")

    # Generate subtask prompts FIRST (before constraint extraction)
    print("🔄 Step 2/4: Generating prompts for each subtask...")
    subtask_prompts: list[SubtaskPromptItem] = subtask_prompt_generator.generate(
        m_session,
        task_prompt,
        user_input_var_names=user_input_variable,
        subtasks_and_tags=subtasks,
        max_new_tokens=max_new_tokens,
        last_subtask_system_template=last_subtask_system_template,
        last_subtask_user_template=last_subtask_user_template,
    ).parse()
    print(f"   ✓ Generated {len(subtask_prompts)} subtask prompts")

    # NEW APPROACH: Generate constraints PER SUBTASK using hybrid method
    # This considers both the original task prompt and the specific subtask context
    print(f"🔄 Step 3/4: Extracting constraints for each subtask (hybrid approach)...")
    subtask_prompts_with_constraints: list[SubtaskPromptConstraintsItem] = []
    all_constraints_dict: dict[str, Literal["code", "llm"]] = {}  # Track all unique constraints
    
    for idx, subtask_prompt in enumerate(subtask_prompts, 1):
        print(f"   Processing subtask {idx}/{len(subtask_prompts)}: {subtask_prompt.subtask[:60]}...")
        # Extract constraints for THIS specific subtask using hybrid approach
        subtask_constraints: list[str] = constraint_extractor.generate(
            m_session,
            input_str=None,  # Not used in hybrid mode
            enforce_same_words=False,
            max_new_tokens=max_new_tokens,
            custom_template=constraint_extractor_template or "per_subtask_hybrid",
            custom_icl_examples=constraint_extractor_icl_examples,
            custom_user_template=constraint_extractor_template or "per_subtask_hybrid",
            original_task_prompt=task_prompt,
            subtask_description=subtask_prompt.subtask,
            subtask_prompt=subtask_prompt.prompt_template,
        ).parse()
        
        # Determine validation strategy for each constraint
        new_constraints_count = 0
        for constraint in subtask_constraints:
            if constraint not in all_constraints_dict:
                # Only call validation_decision once per unique constraint
                new_constraints_count += 1
                all_constraints_dict[constraint] = validation_decision.generate(
                    m_session, constraint, max_new_tokens=max_new_tokens
                ).parse()
        print(f"      → Found {len(subtask_constraints)} constraints ({new_constraints_count} new, {len(subtask_constraints)-new_constraints_count} duplicates)")
        
        # Build SubtaskPromptConstraintsItem
        subtask_prompts_with_constraints.append(
            SubtaskPromptConstraintsItem(
                subtask=subtask_prompt.subtask,
                tag=subtask_prompt.tag,
                prompt_template=subtask_prompt.prompt_template,
                constraints=subtask_constraints,
            )
        )
    
    print(f"   ✓ Extracted total of {len(all_constraints_dict)} unique constraints across all subtasks")
    print("🔄 Step 4/4: Assembling final decomposition result...")

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

    result = DecompPipelineResult(
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
    
    print("✅ Decomposition complete!\n")
    return result
