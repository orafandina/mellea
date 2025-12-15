# Custom Constraint Extractor Templates

This directory contains custom templates for the constraint extraction step in the Mellea decompose() pipeline.

## What Are Constraint Extractors?

During decomposition, Mellea extracts constraints from your task prompt in this step:

```python
task_prompt_constraints: list[str] = constraint_extractor.generate(
    m_session, task_prompt, enforce_same_words=False
).parse()
```

These constraints are then assigned to relevant subtasks to ensure they comply with the original task requirements.

## Why Custom Templates?

The default constraint extractor is designed to extract ALL explicitly mentioned constraints from a task prompt. However, for domain-specific tasks (like COBOL code analysis), many extracted constraints may be:
- Too generic (e.g., "provide clear explanations")
- Not relevant to the technical output
- Focused on communication style rather than technical requirements

Custom templates allow you to:
1. **Filter constraints** to only include domain-relevant ones
2. **Focus on technical constraints** that affect code structure, syntax, or business logic
3. **Ignore generic constraints** about explanation quality or communication style

## How to Use Custom Templates

### 1. Create Your Custom Template

Create a new `.jinja2` file in this directory. For example: `my_custom_extractor.jinja2`

Your template should:
- Follow the same output format as the default template (use `<constraints_and_requirements>` tags)
- Include the `icl_examples` loop if you want to support in-context learning examples
- Support the `enforce_same_words` variable if needed

### 2. Use It in Your Decompose Call

```python
from cli.decompose.pipeline import decompose, DecompBackend

result = decompose(
    task_prompt=your_task_prompt,
    user_input_variable=your_input_vars,
    backend=DecompBackend.claude,
    model_id="claude-haiku-4-5-20251001",
    
    # Add this parameter to use your custom template:
    constraint_extractor_template="my_custom_extractor",  # without .jinja2 extension
    
    # Optionally provide custom ICL examples:
    constraint_extractor_icl_examples=[
        {
            "task_prompt": "Example task...",
            "constraints_and_requirements": ["Constraint 1", "Constraint 2"]
        }
    ],
    
    # Or disable ICL examples entirely:
    # constraint_extractor_icl_examples=[],
)
```

## Example: COBOL-Focused Template

This directory includes `cobol_focused.jinja2` as an example. It:
- Filters out generic constraints about "explaining clearly" or "being thorough"
- Focuses on COBOL-specific technical constraints (syntax, data structures, business logic)
- Provides clear guidance on what is/isn't relevant for COBOL tasks

### Using the COBOL Template:

```python
result = decompose(
    task_prompt=cobol_task_prompt,
    user_input_variable=["cobol_code"],
    backend=DecompBackend.claude,
    model_id="claude-haiku-4-5-20251001",
    constraint_extractor_template="cobol_focused",  # Use the COBOL-focused template
)
```

## Template Variables Available

Your custom template has access to:

- `icl_examples`: List of in-context learning examples (if provided)
- `enforce_same_words`: Boolean flag for whether to enforce using exact words from the original prompt
- `task_prompt`: The actual task prompt (available in user_template.jinja2, not system_template.jinja2)

## Output Format Requirements

Your template MUST generate output with:

1. `<constraints_and_requirements>` opening tag
2. A markdown unordered list (using `-` for bullets) of constraints, OR "N/A" if none found
3. `</constraints_and_requirements>` closing tag
4. The phrase: "All tags are closed and my assignment is finished."

Example output:
```
<constraints_and_requirements>
- Preserve the original COBOL program structure
- Maintain compatibility with COBOL-85 standard
- Handle COMP-3 (packed decimal) fields correctly
</constraints_and_requirements>

All tags are closed and my assignment is finished.
```

Or if no constraints:
```
<constraints_and_requirements>
N/A
</constraints_and_requirements>

All tags are closed and my assignment is finished.
```

## Troubleshooting

If you get a `FileNotFoundError`:
- Make sure your template file is in `libs/mellea/custom_templates/constraint_extractor/`
- Make sure the filename matches what you pass to `constraint_extractor_template` (without `.jinja2`)
- Check that the file has the `.jinja2` extension

If constraints are still not relevant:
- Review your template's instructions to make them more specific
- Consider providing custom ICL examples that demonstrate the types of constraints you want
- Test with `constraint_extractor_icl_examples=[]` to disable built-in examples that might be misleading

