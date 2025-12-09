from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ._icl_examples import ICLExample, icl_examples as default_icl_examples

this_file_dir = Path(__file__).resolve().parent
mellea_root = this_file_dir.parent.parent.parent.parent.parent  # libs/mellea/
custom_templates_dir = mellea_root / "custom_templates" / "subtask_list"

# Default environment for built-in templates
environment = Environment(loader=FileSystemLoader(this_file_dir), autoescape=False)
system_template = environment.get_template("system_template.jinja2")
user_template = environment.get_template("user_template.jinja2")


def get_system_prompt(
    icl_examples: list[ICLExample] = default_icl_examples,
    custom_template: str | None = None,
    custom_icl_examples: list[dict] | None = None
) -> str:
    """Get system prompt for subtask list generation.
    
    Args:
        icl_examples: In-context learning examples to include (default: mellea's built-in)
        custom_template: Name of custom template (e.g., "focused" looks for
                        custom_templates/subtask_list/focused.jinja2)
        custom_icl_examples: Custom ICL examples to use instead of default.
                           If provided, overrides icl_examples parameter.
                           Pass [] to disable ICL examples entirely.
    
    Returns:
        Rendered system prompt
    """
    # Determine which ICL examples to use
    if custom_icl_examples is not None:
        # Use custom examples (could be empty list to disable ICL)
        examples_to_use = custom_icl_examples
    else:
        # Use default mellea examples
        examples_to_use = icl_examples
    
    if custom_template:
        # Try to load custom template
        custom_path = custom_templates_dir / f"{custom_template}.jinja2"
        if custom_path.exists():
            custom_env = Environment(
                loader=FileSystemLoader(custom_templates_dir), 
                autoescape=False
            )
            template = custom_env.get_template(f"{custom_template}.jinja2")
            return template.render(icl_examples=examples_to_use).strip()
        else:
            raise FileNotFoundError(
                f"Custom template '{custom_template}' not found at {custom_path}. "
                f"Available templates: {list(custom_templates_dir.glob('*.jinja2'))}"
            )
    
    # Use default template
    return system_template.render(icl_examples=examples_to_use).strip()


def get_user_prompt(task_prompt: str) -> str:
    return user_template.render(task_prompt=task_prompt).strip()
