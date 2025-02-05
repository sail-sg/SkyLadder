import glob
import math
import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
import datetime

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, \
    lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random

import os

print("Hello!")
# model_name = "tiny_LLaMA_120M"
# name = "tinyllama_120M"
# TODO: What is a better way to pass the model name?
model_name = os.environ['MODEL_NAME']
dataset_name = os.environ['DATASET_NAME']
save_name = os.environ['WANDB_NAME']
gpu_memory = os.environ['GPU_MEMORY']
# if there is a checkpoint path specified
out_dir = Path(os.environ.get('CHECKPOINT_PATH', 'out')) / save_name
# Hyperparameters
num_of_devices = int(os.environ['NUMBER_OF_GPU'])
num_nodes = int(os.getenv('NUM_NODES', 1))
print("num_nodes", num_nodes, 'out_dir', out_dir)
if num_nodes > 1:
    num_of_devices = num_of_devices * num_nodes
global_batch_size = 512
learning_rate = 4e-4
if "120M" in model_name:
    micro_batch_size = 32
elif '1b' in model_name:
    micro_batch_size = 16
elif '360M' in model_name:
    micro_batch_size = 32
elif '7b' in model_name:
    micro_batch_size = 8
elif '3b' in model_name:
    micro_batch_size = 8
else:
    raise ValueError("Invalid model name")
if '512' in model_name:
    micro_batch_size = micro_batch_size * 4  # 1k tokens
    global_batch_size = global_batch_size * 4
elif '1k' in model_name:
    micro_batch_size = micro_batch_size * 2  # 1k tokens
    global_batch_size = global_batch_size * 2
elif '4k' in model_name:
    micro_batch_size = micro_batch_size // 2  # 4k tokens
    global_batch_size = global_batch_size // 2
elif '8k' in model_name:
    micro_batch_size = micro_batch_size // 4  # 8k tokens
    global_batch_size = global_batch_size // 4
elif '16k' in model_name:
    micro_batch_size = micro_batch_size // 8  # 8k tokens
    global_batch_size = global_batch_size // 8
elif '32k' in model_name:
    global_batch_size = global_batch_size // 16
    micro_batch_size = micro_batch_size // 8

if True or gpu_memory == '40960':  # do not change the micro_batch_size
    micro_batch_size = micro_batch_size // 2

if "2b_tokens" in dataset_name or "2b" in dataset_name:
    max_step = 10000
elif "20b_tokens" in dataset_name or "20b" in dataset_name:
    max_step = 40000
elif "c4_news" in dataset_name or "wiki" in dataset_name:
    # around 9b tokens?
    max_step = 25000
elif "mathpro" in dataset_name:
    max_step = 20000  # 4 epochs on mathpro dataset
elif "cc" in dataset_name or 'proweb' in dataset_name or 'fineweb' in dataset_name or 'code' in dataset_name:
    max_step = 100000  # 100B tokens
else:
    raise ValueError("Invalid dataset name")
warmup_steps = 2000
log_step_interval = 10
eval_iters = 100
save_step_interval = min(max_step // 1, 2500)
eval_step_interval = 500

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0, f"gradient_accumulation_steps must be greater than 0, got {gradient_accumulation_steps}, because batch_size {batch_size} micro_batch_size {micro_batch_size} global batch size  {global_batch_size} num_of_devices {num_of_devices}"
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
train_data_config = [
    ("train", 1.0),
    # ("train_slim", 0.693584),
    # ("train_star", 0.306416),
]

val_data_config = [
    ("valid", 1.0),
]

RESET_ROPE=False
if 'rope' in dataset_name:
    rope_update_steps = {0:5000, # 1k
         1000*2: 20000, # for 2k
        2000*2:30000, # for 4k
        4000*2: 100000, # for 8k
        8000*2: 400000, # for 16k
        16000*2: 1000000, # for 32k
                         }
    RESET_ROPE=True


hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", save_name, flush_logs_every_n_steps=log_iter_interval)
wandb_logger = WandbLogger()


def setup(
        num_devices: int = 1,
        train_data_dir: Path = Path("data/redpajama_sample"),
        val_data_dir: Optional[Path] = None,
        precision: Optional[str] = None,
        tpu: bool = False,
        resume: Union[bool, Path] = False,
        eval_only: bool = False,
        load_from: Optional[Path] = None
) -> None:
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)
    print("devices", num_devices, "precision", precision, "resume", resume, "eval_only", eval_only)
    print("train_data_dir", train_data_dir, "val_data_dir", val_data_dir)
    if num_devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            num_devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
                timeout=datetime.timedelta(seconds=14400),
            )
    else:
        strategy = "auto"
    fabric = L.Fabric(devices=num_devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    fabric.print("micro batch size", micro_batch_size)
    # fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume, eval_only, load_from)


def main(fabric, train_data_dir, val_data_dir, resume, eval_only, load_from):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    print("model_name", model_name)
    print("config.intradoc_mask", config.intradoc_mask)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

        # load pretrained model
        if load_from is not None:
            # use torch.load to load the model
            print("loading model from {}".format(load_from))
            state_dict = torch.load(load_from, map_location=fabric.device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=True, assign=True)


    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    elif not resume and len(list(out_dir.glob("*.pth"))) > 0:
        fabric.print(f"Found existing checkpoints in {out_dir}. Resuming from the latest one.")
        resume = sorted(out_dir.glob("*.pth"))[-1]
        print("Resuming from iter_num", state["iter_num"], "on step", state["step_count"])
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state, strict=False)

    train_time = time.perf_counter()
    fabric.print("Running evaluation only mode?", eval_only)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
        mask_attn=config.intradoc_mask,
        merge_method=config.merge_method,
        initial_iter=adjust_iter_num(model_name, state["iter_num"])
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    train(fabric, state, train_dataloader, val_dataloader, monitor, resume, eval_only)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def adjust_iter_num(model_name, initial_iter):
    return 0


def step_to_window_size(step, min_window_size, max_window_size):
    curr_window_size = min_window_size + step
    return min(curr_window_size, max_window_size)


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume, eval_only=False):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        loss = validate(fabric, model, val_dataloader)  # sanity check
        fabric.print(f"Validation loss: {loss:.4f}, PPL: {math.exp(loss):.4f}")
        if eval_only:
            if os.getenv("LOG_FILE"):
                with open(os.getenv("LOG_FILE"), "w") as f:
                    json.dump({"val_loss": loss.item(), "val_ppl": math.exp(loss.item())}, f)
                print("Saved val_loss and val_ppl to {}".format(os.getenv("LOG_FILE")))
            return

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        # measured_flops = measure_flops(meta_model, x)
        # fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0
    initial_time = time.perf_counter()
    fabric.print("initial_iter", initial_iter, "state", state)
    wandb_name = os.environ['WANDB_NAME']
    go_through_dataloader = True
    if "cont" in wandb_name:
        go_through_dataloader = False
        fabric.print("go_through_dataloader is False, because the wandb_name contains 'cont'.")

    if 'proweb' not in dataset_name:
        # only adjust iter_num for non-proweb dataset
        curr_iter = adjust_iter_num(model_name, initial_iter)
        print("Current iter adjusted to", curr_iter, "from initial iter", initial_iter, 'because of dataset',
              dataset_name)
    # fabric.print("curr_iter origins from lower bound of 240000 of initial iter ", initial_iter, "curr_iter", curr_iter,
    #              "max_iters", max_iters)
    fabric.print("curr_iter", curr_iter, "initial_iter", initial_iter, "max_iters", max_iters)
    loss_func = FusedCrossEntropyLoss()

    if "dwd1" in model_name:
        # find out whether it is dwd
        factor = 1
    elif 'dwd2' in model_name:
        factor = 2
    elif 'dwd4' in model_name:
        factor = 4
    else:
        factor = -1

    if RESET_ROPE:
        model.reset_rope_cache(new_base=rope_update_steps[0])


    for train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume and go_through_dataloader:
            if curr_iter % 1000 == 0:
                print("curr_iter", curr_iter, "took", time.perf_counter() - initial_time)
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()
        train_input_ids = train_data["idx"]
        # print("train input ids shape", train_input_ids.shape)

        input_ids = train_input_ids[:, 0: model.config.block_size].contiguous()
        targets = train_input_ids[:, 1: model.config.block_size + 1].contiguous()


        # print("input_ids", input_ids.shape, "targets", targets.shape)

        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            if "fragment_lens" in train_data:
                # print("using fragment_lens and fragment_nums for training.")
                fragment_lens = train_data["fragment_lens"]
                fragment_nums = train_data["fragment_nums"]
                window_size = step_to_window_size(state["step_count"] // factor, 32,
                                                  model.config.block_size) if factor > 0 else None
                # print("iter_num", state["iter_num"], "step_count", state["step_count"], "window_size", window_size)
                logits = model(input_ids, fragment_lens=fragment_lens, fragment_nums=fragment_nums,
                               window_size=window_size)
            elif factor > 0:
                logits = model(input_ids, window_size=step_to_window_size(state["step_count"] // factor, 32,
                                                                          model.config.block_size))
            else:
                logits = model(input_ids)
            # print('logits', logits.shape, 'targets', targets.shape)
            # print('logits', logits[0])
            loss = loss_func(logits, targets)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            grad_norm = fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            fabric.log_dict({
                "gradient_norm": grad_norm.item()
            })
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        if RESET_ROPE and state["step_count"] in rope_update_steps:
            fabric.print("resetting rope with base ", rope_update_steps[state["step_count"]])
            model.reset_rope_cache(new_base=rope_update_steps[state["step_count"]])

        state["iter_num"] += 1
        # input_id: B L
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
            f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. "
            # print days as well
            f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
        )

        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item()
        )

        assert val_dataloader is not None, "val_dataloader is None"

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (
                        state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (
                        state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.barrier()
        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt-step-{state['step_count']}.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        val_data = val_data["idx"]
        input_ids = val_data[:, 0: model.config.block_size].contiguous()
        targets = val_data[:, 1: model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()

    out = losses.mean()

    model.train()
    return out


def create_dataloader(
        batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345,
        split="train", mask_attn="", merge_method="no",
        initial_iter=0
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        print("Found {} files for {}".format(len(filenames), prefix))
        random.seed(seed)
        random.shuffle(filenames)
        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size.
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=4 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
            mask_attn=mask_attn if split == "train" and mask_attn else "",
            merge_method=merge_method,
            initial_iter=initial_iter,
            samples_per_step = micro_batch_size * gradient_accumulation_steps # how many pieces of data is needed for one step
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    def collate_fn_with_intradoc_mask(examples: dict, max_num_fragments_in_chunk=512):
        # print(examples[0].keys())
        # a = ([example["idx"] for example in examples])
        # print(len(a), a[0].shape, type(a[0]))
        input_ids = torch.LongTensor(torch.stack([example["idx"] for example in examples]))
        # print("input_ids", input_ids.shape, input_ids.dtype)
        # if "labels" not in examples[0]:
        #     labels = input_ids
        # else:
        #     labels = torch.LongTensor([example["labels"] for example in examples])
        batch_inputs = {"idx": input_ids}
        if "fragment_lens" in examples[0]:
            fragment_lens = [
                torch.tensor(item["fragment_lens"] + (max_num_fragments_in_chunk - len(item["fragment_lens"])) * [-1])
                for item in examples
            ]
            batch_inputs["fragment_lens"] = torch.stack(fragment_lens)
            fragment_nums = torch.tensor([item["fragment_nums"] for item in examples], dtype=torch.int32)
            batch_inputs["fragment_nums"] = fragment_nums
        return batch_inputs

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                      collate_fn=collate_fn_with_intradoc_mask)


def create_dataloaders(
        batch_size: int,
        block_size: int,
        fabric,
        train_data_dir: Path = Path("data/redpajama_sample"),
        val_data_dir: Optional[Path] = None,
        seed: int = 12345,
        mask_attn="",
        merge_method="none",
        initial_iter=0
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
        mask_attn=mask_attn,
        merge_method=merge_method,
        initial_iter=initial_iter
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
