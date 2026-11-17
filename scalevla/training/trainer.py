"""Cache-first training loop: the feature mapper and an action head trained from scratch with flow matching on cached
DeepSeek-V4.1-Flash features; the backbone itself never runs during training. Single-GPU and multi-GPU (torchrun) runs
share the loop; entry point scripts.train.

<output_dir>/run.json            configuration, data record (normalization stats, profile), sampling record, parameter counts
<output_dir>/init/               initial mapper weights and the head / mapper initialization hashes
<output_dir>/train_log.jsonl     loss, gradient norm, learning rate every 10 steps
<output_dir>/checkpoint-<step>/  policy checkpoint (scalevla.model.checkpoint) and, at run.save_state steps, the resumable
                                 training state (train_state_main.pt + one train_state_rank<r>.pt per rank)
<output_dir>/timing.json         wall-clock record of the run
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from ..data import cache as C
from ..data import embodiments as E
from ..data.dataset import CachedSteps, collate
from ..model.checkpoint import Policy, save_policy
from ..model.head import build_head, ensure_head_init, init_head_state, sha256_state
from ..model.mapper import FeatureMapper
from .config import TrainConfig
from .optim import make_optimizer, make_scheduler
from .sampling import build_stream

# fields that a resumed run must share with the state it resumes from
RESUME_FREE = {("data", "workers"), ("data", "cache_reader"), ("run", "output_dir"), ("run", "save"), ("run", "save_state"),
               ("run", "resume")}


def _steps(spec):
    """"5000,10000" and / or "every:N" -> predicate on the step."""
    fixed = {int(x) for x in spec.split(",") if x and not x.startswith("every:")}
    every = [int(x[6:]) for x in spec.split(",") if x.startswith("every:")]
    return lambda st: st in fixed or any(st % e == 0 for e in every)


def _resume_diff(saved, cfg):
    return {f"{s}.{k}": (saved[s].get(k), v) for s, sec in cfg.items() for k, v in sec.items()
            if (s, k) not in RESUME_FREE and saved.get(s, {}).get(k) != v}


def train(cfg: TrainConfig) -> None:
    D, M, O, R = cfg.data, cfg.model, cfg.optim, cfg.run
    out, t_proc = R.output_dir, time.time()
    profile = E.for_store(D.samples, D.horizon)  # registers the DexJoCo layout before anything else
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    os.makedirs(out, exist_ok=True)

    # the common fresh head initialization, then the mapper from its own seed
    head, gcfg = build_head(M.head_config, D.horizon)
    if M.head_init:
        if rank == 0:
            ensure_head_init(M.head_init, M.head_config, M.init_seed, D.horizon)
        dist.barrier()
        head.load_state_dict(torch.load(M.head_init, map_location="cpu"))
    else:
        head.load_state_dict(init_head_state(M.head_config, M.init_seed, D.horizon))
    ds = CachedSteps(D.samples, D.cache, profile, state_dropout=D.state_dropout, pad_end=D.pad_end, reader=D.cache_reader,
                     demos_per_task=D.demos_per_task, demo_seed=D.demo_seed)
    assert ds.h == D.horizon, f"data.horizon {D.horizon} != the {profile.name} profile's {ds.h}"
    m = C.load_manifest(D.cache)
    d_in = m["spec"]["d"]
    torch.manual_seed(M.mapper_seed)
    mapper = FeatureMapper(d_in, gcfg.backbone_embedding_dim, rank=M.rank)
    model = nn.parallel.DistributedDataParallel(Policy(mapper, head).to(dev), device_ids=[dev.index])
    n_par = {"mapper": sum(p.numel() for p in mapper.parameters()), "head": sum(p.numel() for p in head.parameters())}
    opt = make_optimizer(O, model.module)
    sched, sched_rec = make_scheduler(O, R.steps, opt)
    conf = cfg.to_dict()

    start = 0
    if R.resume:
        rs = torch.load(f"{R.resume}/train_state_rank{rank}.pt", map_location="cpu", weights_only=False)
        main = torch.load(f"{R.resume}/train_state_main.pt", map_location="cpu", weights_only=False, mmap=True)
        diff = _resume_diff(main["config"], conf)
        if diff:
            raise SystemExit(f"run.resume: the state was trained with different settings: {diff}")
        model.module.load_state_dict(main["model"])
        opt.load_state_dict(main["opt"])
        sched.load_state_dict(main["sched"])
        start = rs["step"]
        del main
        torch.set_rng_state(rs["rng_cpu"]); torch.cuda.set_rng_state(rs["rng_cuda"])
        random.setstate(rs["rng_py"]); np.random.set_state(rs["rng_np"])
    elif rank == 0:
        os.makedirs(f"{out}/init", exist_ok=True)
        torch.save(model.module.mapper.state_dict(), f"{out}/init/mapper_init.pt")
        json.dump({"head_init_sha256": sha256_state(model.module.head.state_dict()),
                   "mapper_init_sha256": sha256_state(model.module.mapper.state_dict()),
                   "head_init": M.head_init or f"built in-process from init_seed {M.init_seed}"},
                  open(f"{out}/init/init_record.json", "w"), indent=1)
    stream, sampling_rec = build_stream(ds, cfg, rank, world, start)
    if not R.resume:
        random.seed(R.seed * 1000 + rank); np.random.seed(R.seed * 1000 + rank); torch.manual_seed(R.seed * 1000 + rank)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=stream, num_workers=D.workers, persistent_workers=D.workers > 0,
                                     prefetch_factor=4 if D.workers else None, collate_fn=collate, pin_memory=True,
                                     worker_init_fn=lambda w: random.seed(R.seed * 1000 + rank * 100 + w + start))
    save, save_state = _steps(R.save), _steps(R.save_state)

    def write_state(step):  # model / optimizer / scheduler once (DDP replicas are identical), RNG states per rank
        ck = f"{out}/checkpoint-{step}"
        os.makedirs(ck, exist_ok=True)
        if rank == 0:
            torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "config": conf}, f"{ck}/train_state_main.pt.tmp")
            os.replace(f"{ck}/train_state_main.pt.tmp", f"{ck}/train_state_main.pt")
        torch.save({"step": step, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                    "rng_py": random.getstate(), "rng_np": np.random.get_state()}, f"{ck}/train_state_rank{rank}.pt")
        dist.barrier()
        if rank == 0:
            print(f"saved resumable state {ck}", flush=True)

    log = wb = None
    if rank == 0:
        json.dump({"config": conf, "world": world, "n_params": n_par, "n_chunk_starts": len(ds), "cache_backbone": m["backbone"],
                   "cache_spec": m["spec"], **ds.meta(), "sampling": sampling_rec, "scheduler": sched_rec, "start_step": start,
                   "time": time.strftime("%F %T")}, open(f"{out}/run.json", "w"), indent=1)
        log = open(f"{out}/train_log.jsonl", "a")
        if os.environ.get("WANDB_PROJECT"):  # optional online log; train_log.jsonl stays the record
            import wandb
            wb = wandb.init(project=os.environ["WANDB_PROJECT"], name=os.path.basename(out.rstrip("/")), dir=out, resume="allow",
                            config={"config": conf, "world": world, "backbone": m["backbone"]})
        print(f"params {n_par}, chunk starts {len(ds)}, steps {R.steps}, global batch {R.gbs}, world {world}", flush=True)

    model.train()
    t_loop = t_last = time.time()
    t_data = t_data_all = 0.0
    it = iter(dl)
    step = start
    for step in range(start + 1, R.steps + 1):
        td = time.time()
        b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
        t_data += time.time() - td
        t_data_all += time.time() - td
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == 1:
            l = loss.detach().clone()
            dist.all_reduce(l)
            if rank == 0:
                dt = time.time() - t_last
                rec = {"step": step, "loss": float(l) / world, "grad_norm": float(gn), "lr": sched.get_last_lr()[0],
                       "s_per_step": dt / (10 if step > 1 else 1), "data_wait_frac": t_data / max(dt, 1e-9)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if wb:
                    wb.log(rec, step=step)
                if step % 100 == 0 or step == 1:
                    print(rec, flush=True)
                t_last, t_data = time.time(), 0.0
        if save(step) and rank == 0:
            ck = f"{out}/checkpoint-{step}"
            save_policy(ck, model.module, {"step": step, "rank": M.rank, "d_in": d_in, **ds.meta(), "cache_spec": m["spec"]},
                        M.head_config)
            print(f"saved {ck}", flush=True)
        if save_state(step):
            write_state(step)
    dist.barrier()
    if rank == 0:
        n, loop_s = step - start, time.time() - t_loop
        json.dump({"start": time.strftime("%F %T", time.localtime(t_proc)), "end": time.strftime("%F %T"), "world": world,
                   "gpu": torch.cuda.get_device_name(dev), "steps": n, "startup_s": round(t_loop - t_proc, 1), "loop_s": round(loop_s, 1),
                   "s_per_step": round(loop_s / max(n, 1), 4), "data_wait_frac_rank0": round(t_data_all / max(loop_s, 1e-9), 3),
                   "gpu_hours": round(world * (time.time() - t_proc) / 3600, 2),
                   "peak_mem_gb_rank0": round(torch.cuda.max_memory_allocated(dev) / 2**30, 1)}, open(f"{out}/timing.json", "w"), indent=1)
    dist.destroy_process_group()
