"""Serve a trained ScaleVLA checkpoint through the GR00T PolicyServer (GR00T environment). Features come from a running
feature server (scripts.serve_features) on --feature-port.

    python -m scripts.serve_policy --ckpt <run>/checkpoint-5000 --feature-port 5601 --port 5555 [--cuda-graph]
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--feature-port", type=int, default=5601)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cuda-graph", action="store_true", help="replay the action head as a CUDA graph (same actions, lower latency)")
    ap.add_argument("--flow-steps", type=int, default=0, help="flow-matching integration steps (0 = the configured 4)")
    a = ap.parse_args()
    from scalevla.serving.server import run_server
    run_server(a.ckpt, a.port, a.feature_port, a.device, a.cuda_graph, a.flow_steps)


if __name__ == "__main__":
    main()
