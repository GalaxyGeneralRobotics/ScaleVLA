"""Live feature server for policy rollouts. Local socket (127.0.0.1 only), pickled messages:
  {"cmd": "features", "frames": [[uint8 HxWx3 per camera] per item], "instructions": [str]}
    -> {"features": [int16 [L, d] (bf16 bits)], "token_type": [int8 [L]], "prompt_token_ids": [[int]]}
  {"cmd": "spec"} -> the backbone spec (+ live_overrides);  {"cmd": "ping"} -> "pong"
Replies are {"ok": True, "out": ..., "t_server_s": compute seconds} or {"ok": False, "error", "tb"}; the server keeps
serving after an error. Client: scalevla.features.client.FeatureClient.

batch_window_s > 0 (several policy servers sharing one backbone): the "features" requests that arrive within the window
after the first one (same camera count) run as one backbone call, and each client gets its own rows back. The
batch-invariant kernels of the DeepSeek engine make the rows independent of the batch.
"""
import pickle
import time
import traceback

import torch


def _pack(res):
    return {"features": [f.view(torch.int16).numpy() for f, _, _ in res], "token_type": [t for _, t, _ in res],
            "prompt_token_ids": [p for _, _, p in res]}


def _error(e):
    return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}


def serve_batched(backbone, port, spec, batch_window_s):
    import zmq
    sock = zmq.Context().socket(zmq.ROUTER)
    sock.bind(f"tcp://127.0.0.1:{port}")
    print(f"serving {backbone.name} on 127.0.0.1:{port}, batch window {batch_window_s * 1e3:.0f} ms", flush=True)

    def reply(ident, rep):
        sock.send_multipart([ident, b"", pickle.dumps(rep)])

    while True:
        f = sock.recv_multipart()
        pending = [(f[0], pickle.loads(f[-1]))]
        end = time.perf_counter() + batch_window_s
        while sock.poll(max(int((end - time.perf_counter()) * 1e3), 0)):
            f = sock.recv_multipart()
            pending.append((f[0], pickle.loads(f[-1])))
        groups = {}
        for ident, msg in pending:
            if msg.get("cmd") == "features" and msg.get("frames"):
                groups.setdefault(len(msg["frames"][0]), []).append((ident, msg))
            elif msg.get("cmd") in ("ping", "spec"):
                reply(ident, {"ok": True, "out": "pong" if msg["cmd"] == "ping" else spec, "t_server_s": 0.0})
            else:
                reply(ident, {"ok": False, "error": f"unknown cmd {msg.get('cmd')}", "tb": ""})
        for grp in groups.values():
            t_req = time.perf_counter()
            try:
                res = backbone.features([it for _, m in grp for it in zip(m["frames"], m["instructions"])])
                n, i = len(res), 0
                for ident, m in grp:
                    k = len(m["frames"])
                    reply(ident, {"ok": True, "out": _pack(res[i:i + k]), "t_server_s": time.perf_counter() - t_req, "batch_items": n})
                    i += k
            except Exception as e:  # report to every client of the batch, keep serving
                for ident, _ in grp:
                    reply(ident, _error(e))


def serve(backbone, port, live_overrides=None, batch_window_s=0.0):
    spec = {**backbone.spec, "live_overrides": live_overrides or {}, "batch_window_s": batch_window_s}
    if batch_window_s > 0:
        return serve_batched(backbone, port, spec, batch_window_s)
    import zmq
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{port}")
    print(f"serving {backbone.name} on 127.0.0.1:{port}", flush=True)
    while True:
        msg = pickle.loads(sock.recv())
        t_req = time.perf_counter()
        try:
            if msg["cmd"] == "ping":
                out = "pong"
            elif msg["cmd"] == "spec":
                out = spec
            elif msg["cmd"] == "features":
                out = _pack(backbone.features(list(zip(msg["frames"], msg["instructions"]))))
            else:
                raise ValueError(f"unknown cmd {msg.get('cmd')}")
            sock.send(pickle.dumps({"ok": True, "out": out, "t_server_s": time.perf_counter() - t_req}))
        except Exception as e:  # report to the client, keep serving
            sock.send(pickle.dumps(_error(e)))
