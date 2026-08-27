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

        for sample_start in range(0, args.num_samples, args.device_batch_size, ):
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


        





