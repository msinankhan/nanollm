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
    """ There are no choices

        There is one continuation

        The question is:
        👉 Given a prefix, does the model correctly predict the continuation?


        For LM evaluation, we need two versions of the prompt:

        Prompt WITHOUT continuation
        → Used to identify where the continuation begins in token space

        Prompt WITH continuation
        → Used to compute loss / predictions on the continuation tokens

        That’s why this function returns two prompts.

    """


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
    """ Length of the common prefix or suffix across token sequences.
        Given multiple token sequences, we want to know:

            How many tokens are identical at the start (prefix), or

            How many tokens are identical at the end (suffix)

        This is essential because:

            Multiple choice → same prefix, different endings

            Schema → different prefixes, same suffix

    """
    
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

def stack_sequences(tokens,pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz,seq_len=len(tokens),max(len(x) for x in tokens)
    input_ids=torch.full((bsz,seq_len),pad_token_id,dtype=torch.long)
    for i,x in enumerate(tokens):
        input_ids[i,:len(x)] =torch.tensor(x,dtype=torch.long)

    return input_ids


def batch_sequences_mc(tokenizer,prompts):
    tokens=tokenizer(prompts,prepend=tokenizer.get_bos_token())
    answer_start_idx=find_common_length(tokens,direction='left')
    start_idx=[answer_start_idx] * len(prompts)    # Because the prompts are the same, but we have different answers
    end_idx=[len(x) for x in tokens]
    
    return tokens, start_idx, end_idx

def batch_sequences_schema(tokenizer,prompts):

    """In Schema, the context varies but the continuation remains the same.
       In schema tasks, we measure how well each context predicts that continuation"""

    tokens=tokenizer(prompts, prepend=tokenizer.get_bos_token())

    suffix_length=find_common_length(tokens,direction='right')
    end_idx=[len(x) for x in tokens]
    start_idx=[ei - suffix_length for ei in end_idx]
    return tokens, start_idx, end_idx



def batch_sequences_lm(tokenizer,prompts):
    tokens=tokenizer(prompts, prepend=tokenizer.get_bos_token())
    tokens_without,tokens_with=tokens
    start_idx,end_idx=len(tokens_without),len(tokens_with)
    assert start_idx<end_idx , "Prompt_without is supposed to be a prefix of prompt with."
    assert tokens_without==tokens_with[:start_idx], "Prompt_without is supposed to be a prefix of prompt_with"

    return [tokens_with], [start_idx], [end_idx]



@torch.no_grad()
def forward(model,input_ids):
    batch_size,seq_len=input_ids.size()
    outputs=model(input_ids)

    target_ids=torch.roll(input_ids,shifts=-1, dims=1)

    losses=torch.nn.functional.cross_entropy(
        outputs.view(batch_size*seq_len,-1),
        target_ids.view(batch_size*seq_len),
        reduction='none'
    ).view(batch_size,seq_len)

    losses[:,-1]=float('nan')
    predictions=outputs.argmax(dim=-1)
    return losses,predictions


@torch.no_grad()
def evaluate_example(idx,model,tokenizer,data,device,task_meta):
    item=data[idx]
    task_type=task_meta["task_type"]
    num_fewshot=task_meta["num_fewshot"]
    continuation_delimiter=task_meta["continuation_delimiter"]

    fewshot_examples=[]

    if num_fewshot>0:
        rng=random.Random(1234+idx)
        available_indices=[i for i in range(len(data)) if i!=idx]
        fewshot_indices=rng.sample(available_indices,num_fewshot)
        fewshot_examples=[data[i] for i in fewshot_indices]


    if task_type=="multiple_choice":
        prompts=render_prompt_mc(item,continuation_delimiter,fewshot_examples)
        tokens,start_idx,end_idx=batch_sequences_mc(tokenizer,prompts)
    elif task_type=="schema":
        prompts=render_prompts_schema(item,continuation_delimiter,fewshot_examples)
        tokens,start_idx,end_idx=batch_sequences_schema(tokenizer,prompts)
    elif task_meta=="language_modeling":
        prompts=render_prompt_lm(item,continuation_delimiter,fewshot_examples)
        tokens,start_idx,end_idx=batch_sequences_lm(tokenizer,prompts)

    else:
        raise ValueError(f"Unsupported task type:{task_type}")
    
    if hasattr(model,'max_seq_len') and model.max_seq_len is not None:
        max_tokens=model.max_seq_len
        new_tokens,new_start_idx,new_end_idx=[],[],[]
        for t,s,e in enumerate(zip(start_idx,end_idx)):
            if len(t)>max_tokens:
                num_to_crop=len(t)-max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idx.append(s-num_to_crop)
                new_end_idx.append(e-num_to_crop)
                assert s- num_to_crop>=0, f"This shouldn't happen, righttt?"
                assert e - num_to_crop>=0, f"This should happen either, right?"

            else:
                new_tokens.append(t)
                new_start_idx.append(s)
                new_end_idx.append(e)

        tokens,start_idx,end_idx=new_tokens,new_start_idx,new_end_idx



    pad_token_id=tokenizer.get_bos_token()
    input_ids=stack_sequences(tokens,pad_token_id)
    input_ids=input_ids.to(device)


    losses,predictions=forward(model,input_ids)


    if task_type == 'language_modeling':
        si=start_idx[0]
        ei=end_idx[0]

        predicted_tokens=predictions[:,si-1:ei-1]
        actual_token=input_ids[0,si:ei]
        is_correct=torch.all(predicted_tokens==actual_token).item()


    elif task_type in ['multiple_choice','schema']:
        mean_losses=[losses[i,si-1:ei-1].mean().item()
                     for i ,(si,ei) in enumerate(zip(start_idx,end_idx))]
        
        pred_idx=mean_losses.index(min(mean_losses))
        is_correct = pred_idx==item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    return is_correct


def evaluate_task( model,tokenizer,data,device,task_meta):
    rank=dist.get_rank() if dist.is_initialized() else 0
    world_size=dist.get_world_size() if dist.is_initialized else 1
    correct =torch.zeros(len(data), dtype=torch.float32, device=device)

    for idx in range(rank,len(data), world_size):
        is_correct = evaluate_example(idx,model,tokenizer, data,device,task_meta)
        correct[idx] =is_correct

    if world_size>1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOP.SUM) #gradient synchronization in DDP

    mean_correct=correct.mean().item()

    return mean_correct