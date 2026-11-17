"""Client of the ScaleVLA policy server (GR00T PolicyServer): zmq REQ with msgpack / msgpack-numpy, the wire format of
gr00t.policy.server_client without importing gr00t, so it runs in any simulator environment (pyzmq, msgpack,
msgpack-numpy). call("get_action", {"observation": flat GR00T sim observation, "options": ...}) -> (action dict, info);
call("reset", {"options": None})."""
import io

import numpy as np


class PolicyClient:
    def __init__(self, port, host="127.0.0.1", timeout_ms=600000):
        import zmq
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def call(self, endpoint, data=None):
        import msgpack
        import msgpack_numpy as mnp

        def dec(o):
            if isinstance(o, dict) and (o.get("__ndarray_class__") or o.get(b"__ndarray_class__")):
                return np.load(io.BytesIO(o.get("as_npy", o.get(b"as_npy"))), allow_pickle=False)
            return mnp.decode(o)
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(msgpack.packb(req, default=mnp.encode))
        r = msgpack.unpackb(self.sock.recv(), object_hook=dec, raw=False)
        if isinstance(r, dict) and "error" in r:
            raise RuntimeError(f"policy server: {r['error']}")
        return r
