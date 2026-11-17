"""Policy server: the ScaleVLA policy behind the GR00T PolicyServer and Gr00tSimPolicyWrapper, so that the rollout clients
of each benchmark connect directly. Several policy servers (on different GPUs or ports) can share one feature server."""
import json
import time


def run_server(ckpt, port=5555, feature_port=5601, device="cuda:0", cuda_graph=False, flow_steps=0, host="127.0.0.1"):
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer
    from .policy import CachedHeadPolicy
    t0 = time.time()
    pol = CachedHeadPolicy(ckpt, feature_port, device, cuda_graph=cuda_graph, flow_steps=flow_steps)
    print("feature server:", pol.ds.call(cmd="ping"), "| policy", ckpt, "|",
          json.dumps({**pol.describe(), "device": device, "startup_s": round(time.time() - t0, 1)}), flush=True)
    PolicyServer(Gr00tSimPolicyWrapper(pol), host=host, port=port).run()
