import argparse
import os
import itertools
import wandb
import torch
import torch.distributed as dist
from nanollm.commons import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanollm.checkpoint_manager import save_checkpoint, load_model
from nanollm.engine import Engine
from tasks.gsm8k import GSM8K


parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()
user_config = vars(args).copy()


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank ==0


use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanollm-rl", name=args.run, config=user_config)

model,tokenizer,meta = load_model("sft", device, phase="eval", model_tag= args.model_tag, step=args.model_step)
engine= Engine(model,tokenizer)


train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
assert args.num_epochs > 0
assert args.device_batch_size > 0
assert args.examples_per_step > 0
assert args.num_samples > 0
assert args.max_new_tokens > 0
assert args.eval_every > 0
assert args.eval_examples > 0
assert args.save_every > 0
assert args.temperature >= 0
assert args.top_k >= 0
num_steps = (len(train_task)// args.examples_per_step) * args.num_epochs


assert num_steps > 0
assert args.examples_per_step % ddp_world_size == 0

print0(f"Calculate number of steps: {num_steps}")

@torch.no_grad()
def get_batch():

    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)

    assert len(rank_indices) > 0

    for example_idx in itertools.cycle(rank_indices):

        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length= len(tokens)

        model.eval()
        generated_token_sequences = []
        masks= []

        # num_sampling_steps = args.num_samples //args.device_batch_size

        for sample_start in range(0, args.num_samples, args.device_batch_size):
            current_batch_size = min( args.device_batch_size, args.num_samples - sample_start,)
            sampling_step = sample_start // args.device_batch_size

            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=current_batch_size,
                max_tokens = args.max_new_tokens, 
                temperature = args.temperature, 
                top_k= args.top_k,
                seed=seed,
            )

            

            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)


        assert len(generated_token_sequences) == args.num_samples
        assert len(masks) == args.num_samples
        assert all( len(sequence) == len(mask) for sequence, mask in zip(generated_token_sequences, masks))

        rewards=[]


        for sample_tokens in generated_token_sequences:

            generated_tokens= sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)

            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)


        max_length= max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks= [mask + [0] * (max_length- len(mask)) for mask in masks]

        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        inputs = ids[:, :-1]
        targets = ids[:,1:].clone()
        targets[mask_ids[:,1:]==0] = -1

        rewards = torch.tensor(rewards, dtype = torch.float, device=device)

        mu = rewards.mean()

        advantages = rewards - mu
        yield generated_token_sequences, inputs, targets, rewards, advantages



def run_gsm8k_eval(task,tokenizer,engine,
                   max_examples=None, num_samples=1,
                   max_completion_tokens=256, temperature =0.0,
                   top_k=50):

    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank,max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        assert num_samples <= args.device_batch_size

        generated_token_sequences, masks = engine.generate_batch(
            tokens,num_samples=num_samples,
            max_tokens = max_completion_tokens,
            temperature= temperature,
            top_k=top_k
        )

        outcomes=[]

        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct" : is_correct
            })
        record ={
                "idx" : idx,
                "outcomes" : outcomes,
            }

        yield record


optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group['lr']

def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

print0(f"Total Sequences per step: {args.examples_per_step * args.num_samples}")
assert args.examples_per_step % ddp_world_size ==0, "Desired examples per step must be divisible by the number of ranks."
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Calculated examples per rank: {examples_per_rank}")

batch_iterator = get_batch()

for step in range(num_steps):

    if step % args.eval_every ==0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter)

        for k in range(1, args.device_batch_size + 1):
            passk[k-1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)

        num_records = torch.tensor(len(records) , dtype = torch.long, device=device)

        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)

        assert num_records.item() > 0
        passk = passk/ num_records.item()
        print_passk = [f"Pass@{k}: {passk[k-1].item():.4f}" for k in range(1, args.device_batch_size +1)]
        print0(f"Step {step} | { ', '.join(print_passk)}")
        log_passk = {f"pass@{k}" : passk[k-1].item() for k in range(1,args.device_batch_size +1)}

        wandb_run.log({
            "step" : step,
            **log_passk
        })

        rewards_list =[]
        sequence_lengths =[]

        for example_step in range(examples_per_rank):
            sequences_all , inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

            model.train()

            assert inputs_all.size(0) % args.device_batch_size == 0

            num_passes = inputs_all.size(0) // args.device_batch_size
            for pass_idx in range(num_passes):
                b0,b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
                inputs=inputs_all[b0:b1]
                targets= targets_all[b0:b1]
                rewards = rewards_all[b0:b1]
                advantages = advantages_all[b0:b1]

                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)

                pg_obj = (logp * advantages.unsqueeze(-1)).sum()

                num_valid = (targets >=0).sum().clamp(min=1)
                pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

                loss = -pg_obj
                loss.backward()
                print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")

            rewards_list.append(rewards_all.mean().item())
            sequence_lengths.extend(len(seq) for seq in sequences_all)

        mean_reward = sum(rewards_list) / len(rewards_list)
        mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)

        if ddp:
            mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
            mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype = torch.float, device=device)
            dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(mean_sequence_length, op=dist.ReduceOp.AVG)
            mean_reward = mean_reward_tensor.item()
            mean_sequence_length = mean_sequence_length_tensor.item()

        print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")

        wandb_run.log({
            "step": step,
            "reward" : mean_reward,
            "sequence_length" : mean_sequence_length,
        })

        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        optimizer.step()
        model.zero_grad(set_to_none=True)
        wandb_run.log({
            "step":step,
            "lrm" : lrm,
        })

        if master_process and ((step>0 and step % args.save_every==0) or step==num_steps -1):
            base_dir = get_base_dir()
            depth = model.config.n_layer
            output_dirname = args. model_tag if args.model_tag else f"d{depth}"
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
            model_config_kwargs = model.config.__dict__
            save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish() # wandb run finish
compute_cleanup()









