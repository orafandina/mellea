from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ._icl_example_groups import (
    ICLExampleGroup,
    icl_example_groups as default_icl_example_groups,
)

this_file_dir = Path(__file__).resolve().parent

environment = Environment(loader=FileSystemLoader(this_file_dir), autoescape=False)
system_template = environment.get_template("system_template.jinja2")
user_template = environment.get_template("user_template.jinja2")
system_template_last_subtask = environment.get_template("system_template_last_subtask.jinja2")
user_template_last_subtask = environment.get_template("user_template_last_subtask.jinja2")


def get_system_prompt(
    icl_example_groups: list[ICLExampleGroup] = default_icl_example_groups,
    user_input_variables_exists: bool = False,
    is_last_subtask: bool = False,
    custom_template_name: str | None = None,
) -> str:
    # Only use special last subtask template if explicitly requested via custom_template_name
    if is_last_subtask and custom_template_name:
        # Load custom template from custom_templates directory
        custom_templates_dir = this_file_dir.parent.parent.parent.parent.parent / "custom_templates" / "last_subtask_system"
        if (custom_templates_dir / f"{custom_template_name}.jinja2").exists():
            custom_env = Environment(loader=FileSystemLoader(str(custom_templates_dir)), autoescape=False)
            template = custom_env.get_template(f"{custom_template_name}.jinja2")
        else:
            # Fallback to built-in last subtask template if custom not found
            template = system_template_last_subtask
    elif is_last_subtask:
        # Use built-in last subtask template
        template = system_template_last_subtask
    else:
        # Default: use regular template for all subtasks
        template = system_template
    
    return template.render(
        icl_example_groups=icl_example_groups,
        user_input_variables_exists=user_input_variables_exists,
    ).strip()


def get_user_prompt(
    task_prompt: str,
    execution_plan: list[str],
    available_content_variables: list[str],
    target_subtask: str,
    is_last_subtask: bool = False,
    custom_template_name: str | None = None,
) -> str:
    # Only use special last subtask template if explicitly requested via custom_template_name
    if is_last_subtask and custom_template_name:
        # Load custom template from custom_templates directory
        custom_templates_dir = this_file_dir.parent.parent.parent.parent.parent / "custom_templates" / "last_subtask_user"
        if (custom_templates_dir / f"{custom_template_name}.jinja2").exists():
            custom_env = Environment(loader=FileSystemLoader(str(custom_templates_dir)), autoescape=False)
            template = custom_env.get_template(f"{custom_template_name}.jinja2")
        else:
            # Fallback to built-in last subtask template if custom not found
            template = user_template_last_subtask
    elif is_last_subtask:
        # Use built-in last subtask template
        template = user_template_last_subtask
    else:
        # Default: use regular template for all subtasks
        template = user_template
    
    return template.render(
        task_prompt=task_prompt,
        execution_plan=execution_plan,
        available_content_variables=available_content_variables,
        target_subtask=target_subtask,
    ).strip()
