import random
import torch

from jinja2 import Template
import torch.distributed as dist

def render_prompt_mc(item,continuation_delimiter,fewshot_examples=None):

    """Returns a prompt for each choice in the MCQ."""

    template_str="""
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    
    template=Template(template_str)
    fewshot_examples=fewshot_examples or []

    context={
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter':continuation_delimiter,
        'item':item
    }

    prompts=[template.render(choice=choice,**context) for choice in item['choices']]

    return prompts

def render_prompts_schema(item,continuation_delimiter,fewshot_examples=None):

    """In schema tasks, the model evaluates different contexts with the same answer."""

    template_str="""
    {%- for example in fewshot_examples -%}
    {{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

    {% endfor -%}
    {{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()

    template=Template(template_str)
    fewshot_examples=fewshot_examples or []
    context={
        'fewshot_examples':fewshot_examples,
        'continuation_delimiter':continuation_delimiter,
        'item':item
    }

    prompts=[template.render(context=context_option,**context) for context_option in item['context_options']]

    return prompts
    

def render_prompt_lm(item,continuation_delimiter,fewshot_examples=None):
    template_str="""
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template=Template(template_str)
    fewshot_examples=fewshot_examples or []
    context={
        'fewshot_examples':fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item':item
    }

    prompt_without=template.render(include_continuation=False,**context)
    prompt_with=template.render(include_continuation=True,**context)

    prompt_without=prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequence,direction='left'):
    """Length of the common prefix or suffix across token sequences."""
    
    min_len=min(len(seq) for seq in token_sequence)
    indices={
        'left':range(min_len),
        'right':range(-1,-min_len-1,-1)
    }[direction]

    for i, idx in enumerate(indices):
        token=token_sequence[0][idx]
        if not all(seq[idx]==token for seq in token_sequence):
            return idx

    return min_len