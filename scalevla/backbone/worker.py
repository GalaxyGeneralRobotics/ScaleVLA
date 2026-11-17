"""Worker-side functions of the DeepSeek backbone, called inside the vLLM workers through LLM.collective_rpc.

Which rows belong to which request comes from vLLM's own batch bookkeeping, never from token ids alone (different frames
of one task have identical prompt ids: image placeholders and the same instruction). For every forward step, rank 0
records
  - the step layout as the model runner built it: V2 runner (default) = the BatchReqState returned by
    gather_batch_req_state (request ids sorted by the runner, not in submission order; scheduled token counts; start =
    computed prefill tokens); V1 runner = input_batch.req_ids / num_scheduled_tokens / num_computed_tokens_cpu
  - the step's positions and input_ids (pre-hook on layers[0]) and the feature rows (hook on the feature tap)
_assemble() rebuilds every request from its own chunks and checks that the positions equal the chunk range, the ids equal
the request's prompt_token_ids slice and the rows cover 0..L-1 exactly once. Tensor-parallel ranks hold identical
replicated activations, so only rank 0 records and writes. Shards are written here, inside the worker, so GBs of
features never cross the RPC boundary.
"""
_RUNNER_V1 = ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner")
_RUNNER_V2 = ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner")


def _rank0():
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0


def _state():
    import builtins
    return builtins.__dict__.setdefault("_scalevla_cache", {"hooks": [], "steps": [], "cur": None, "restore": None})


def _layers(model):
    import re
    found = {}
    for name, mod in model.named_modules():
        m = re.search(r"(^|\.)layers\.(\d+)$", name)
        if m:
            found[int(m.group(2))] = (name, mod)
    return [found[i] for i in sorted(found)]


def _hook_feature(model, fn, n_enc=20):
    """Register fn(feature) at the feature point: the output of `feature_tap` when the early-exit patch is active
    (layers[n_enc].attn then never runs), else the input of layers[n_enc].attn (attn(positions, x, None))."""
    for name, mod in model.named_modules():
        if name.endswith("feature_tap"):
            parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
            if getattr(parent, "scalevla_feature_layer", 0) == n_enc:
                return mod.register_forward_hook(lambda m, a, o: (fn(o), None)[1])
    return _layers(model)[n_enc][1].attn.register_forward_pre_hook(lambda m, args: (fn(args[1]), None)[1])


def _unhook(st):
    for h in st["hooks"]:
        h.remove()
    st["hooks"] = []
    if st["restore"] is not None:
        runner, name, orig = st["restore"]
        setattr(runner, name, orig)
        st["restore"] = None


def arm_layout(worker, n_enc=20):
    st = _state()
    _unhook(st)
    st["steps"], st["cur"] = [], None
    runner = worker.model_runner
    kind = (type(runner).__module__, type(runner).__name__)
    if kind not in (_RUNNER_V1, _RUNNER_V2):
        raise RuntimeError(f"layout capture supports {_RUNNER_V1} / {_RUNNER_V2}, got {kind}")
    if not _rank0():
        return {"runner": kind, "recording": False}
    st["lay"] = None
    if kind == _RUNNER_V2:
        orig = runner.gather_batch_req_state

        def gather_batch_req_state(scheduler_output, dummy_run, *a, **k):
            out = orig(scheduler_output, dummy_run, *a, **k)
            bs = out[0]
            st["lay"] = None if bs is None else [
                (r, int(s), int(n)) for r, s, n in zip(bs.req_ids, bs.num_computed_prefill_tokens_np, bs.num_scheduled_tokens)]
            return out
        runner.gather_batch_req_state = gather_batch_req_state
        st["restore"] = (runner, "gather_batch_req_state", orig)
    else:
        orig = runner.execute_model

        def execute_model(scheduler_output, *a, **k):
            st["so"] = scheduler_output
            return orig(scheduler_output, *a, **k)
        runner.execute_model = execute_model
        st["restore"] = (runner, "execute_model", orig)

    def pre(m, args, kwargs):
        pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
        ids = kwargs.get("input_ids", args[2] if len(args) > 2 else None)
        if kind == _RUNNER_V2:
            lay, st["lay"] = st["lay"], None
            if lay is None:
                raise RuntimeError("forward step without a captured batch layout")
        else:
            ib, so = runner.input_batch, st["so"]
            lay = [(r, int(ib.num_computed_tokens_cpu[ib.req_id_to_index[r]]), int(so.num_scheduled_tokens[r]))
                   for r in ib.req_ids[:ib.num_reqs]]
        st["cur"] = (lay, pos.detach().to("cpu", copy=True), ids.detach().to("cpu", copy=True))

    def feat(f):
        lay, pos, ids = st["cur"]
        st["cur"] = None
        n = sum(c[2] for c in lay)
        if not (f.shape[0] == pos.numel() == ids.numel() == n):
            raise RuntimeError(f"step rows {f.shape[0]} positions {pos.numel()} ids {ids.numel()} layout {n}")
        st["steps"].append((lay, pos, ids, f.detach().to("cpu", copy=True)))

    st["hooks"].append(_layers(worker.get_model())[0][1].register_forward_pre_hook(pre, with_kwargs=True))
    st["hooks"].append(_hook_feature(worker.get_model(), feat, n_enc))
    return {"runner": kind, "recording": True}


def disarm_layout(worker):
    st = _state()
    _unhook(st)
    st["steps"] = []
    return True


def _assemble(steps, reqs):
    """reqs: [(external request id, prompt ids)] -> [L, d] feature tensors in reqs order."""
    import torch
    chunks = {}
    for lay, pos, ids, rows in steps:
        o = 0
        for rid, start, n in lay:
            chunks.setdefault(rid.rsplit("-", 1)[0], []).append((start, pos[o:o + n], ids[o:o + n], rows[o:o + n]))
            o += n
    out = []
    for ext, pid in reqs:
        cs = sorted(chunks.pop(ext, []), key=lambda c: c[0])
        L, p, parts = len(pid), 0, []
        pid_t = torch.as_tensor(pid)
        for start, pos, ids, rows in cs:
            n = rows.shape[0]
            if start != p:
                raise RuntimeError(f"request {ext}: chunk starts at {start}, expected {p} (gap / overlap)")
            if not torch.equal(pos.to(torch.int64), torch.arange(start, start + n)):
                raise RuntimeError(f"request {ext}: positions do not match chunk [{start}, {start + n})")
            if not torch.equal(ids.to(torch.int64), pid_t[start:start + n].to(torch.int64)):
                raise RuntimeError(f"request {ext}: input_ids differ from prompt_token_ids[{start}:{start + n}]")
            parts.append(rows)
            p += n
        if p != L:
            raise RuntimeError(f"request {ext}: rows cover {p} of {L} prompt tokens")
        out.append(torch.cat(parts))
    if chunks:
        raise RuntimeError(f"rows recorded for unexpected requests: {sorted(chunks)[:5]}")
    return out


def write_shard(worker, reqs, sample_idx, token_types, path, metadata):
    """reqs: [(external request id, prompt ids)]; sample_idx and token_types aligned with reqs. Atomic write on rank 0."""
    from scalevla.data.cache import save_shard
    st = _state()
    if not _rank0():
        st["steps"] = []
        return None
    feats = _assemble(st["steps"], reqs)
    st["steps"] = []
    res = save_shard(path, feats, token_types, sample_idx, metadata)
    return {"n": len(reqs), "rows": res["rows"], "dtype": str(feats[0].dtype)}


def take(worker, reqs):
    """Features of the given requests, run since the last arm / take (live server)."""
    st = _state()
    if not _rank0():
        st["steps"] = []
        return None
    feats = _assemble(st["steps"], reqs)
    st["steps"] = []
    return feats
