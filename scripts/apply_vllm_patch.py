"""Patch vLLM 0.31 for DeepSeek-V4.1-Flash feature extraction (DeepSeek vLLM environment). Every hunk is an exact string
match: the script stops without writing anything if the source differs, compiles the result before writing, and is
idempotent. storage/patches/vllm_early_exit.diff shows the resulting change.

    python -m scripts.apply_vllm_patch "$(python -c 'import site; print(site.getsitepackages()[0])')" early_exit

  early_exit  with SCALEVLA_DS_FEATURE_LAYER=K (read at model init), DeepseekV4Model.forward runs layers 0..K-1, then only
              the pre-attention step of layer K (_attn_pre: the post step of the previous sublayer, engram injection, the
              hyper-connection pre-collapse and attn_norm = the input of layers[K].attn, the feature for K = 20), passes
              it through `feature_tap` (an nn.Identity hook point) and returns it as the final hidden states. The
              attention of layer K, the decoder layers after it and the final norm never run. DecoderLayer.forward is
              split into _attn_pre + attn + _attn_post_ffn without changing its computation. With the variable unset or
              0, the model runs unchanged.
"""
import argparse
import difflib
import os
import sys

MODEL41 = "vllm/models/deepseek_v41/nvidia/model.py"

PATCHES = {
    # DeepSeek-V4.1-Flash resolves to DeepseekV41ForCausalLM (vllm/models/deepseek_v41/nvidia/{vl_model,model}.py); its
    # vision path already runs data-parallel with --mm-encoder-tp-mode data, so only the early exit is patched.
    "early_exit": [
        (MODEL41,
         "import vllm.envs as envs\n",
         "import os\n\nimport vllm.envs as envs\n"),
        # DecoderLayer.forward -> _attn_pre (everything up to the attention input) + attn + _attn_post_ffn
        (MODEL41,
         "        previous_aux: torch.Tensor | None = None\n"
         "        mhc_stream = self.mhc_stream\n",
         "        residual, post_mix, res_mix, x, attn_pre, previous_aux, mhc_stream = self._attn_pre(\n"
         "            x, positions, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask,\n"
         "            capture_previous_aux,\n"
         "        )\n"
         "        x = self.attn(positions, x, None)\n"
         "        return self._attn_post_ffn(\n"
         "            x, positions, input_ids, residual, post_mix, res_mix, attn_pre, previous_aux, mhc_stream,\n"
         "            mega_gate_metadata,\n"
         "        )\n"
         "\n"
         "    def _attn_pre(\n"
         "        self, x, positions, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask,\n"
         "        capture_previous_aux,\n"
         "    ):\n"
         "        \"\"\"ScaleVLA early_exit: everything before this layer's attention (previous sublayer's post,\n"
         "        engram injection, mHC pre-collapse + attn_norm). Returns (residual, post_mix, res_mix, attention\n"
         "        input, attn_pre, previous_aux, mhc_stream).\"\"\"\n"
         "        previous_aux: torch.Tensor | None = None\n"
         "        mhc_stream = self.mhc_stream\n"),
        (MODEL41,
         "        if self.use_sequence_parallel:\n"
         "            x = sp_all_gather(x)[: positions.shape[0]]\n"
         "\n"
         "        x = self.attn(positions, x, None)\n",
         "        if self.use_sequence_parallel:\n"
         "            x = sp_all_gather(x)[: positions.shape[0]]\n"
         "        return residual, post_mix, res_mix, x, attn_pre, previous_aux, mhc_stream\n"
         "\n"
         "    def _attn_post_ffn(\n"
         "        self, x, positions, input_ids, residual, post_mix, res_mix, attn_pre, previous_aux, mhc_stream,\n"
         "        mega_gate_metadata,\n"
         "    ):\n"),
        # Model: hook point + config
        (MODEL41,
         "        self.start_layer, self.end_layer, self.layers = make_layers(\n",
         "        # ScaleVLA early_exit: stop after producing the attention input of this layer (0 = off)\n"
         "        self.scalevla_feature_layer = int(os.environ.get(\"SCALEVLA_DS_FEATURE_LAYER\", \"0\"))\n"
         "        self.feature_tap = nn.Identity()\n"
         "        self.start_layer, self.end_layer, self.layers = make_layers(\n"),
        (MODEL41,
         "            start=self.start_layer,\n"
         "        ):\n"
         "            hidden_states, residual, post_mix, res_mix, pre_mix, previous_aux = layer(\n",
         "            start=self.start_layer,\n"
         "        ):\n"
         "            if self.scalevla_feature_layer and idx == self.scalevla_feature_layer:\n"
         "                assert not self.use_sequence_parallel and get_pp_group().world_size == 1\n"
         "                *_, feature, _, _, mhc_stream = layer._attn_pre(\n"
         "                    hidden_states, positions, pre_mix, post_mix, res_mix, residual, engram_hashes,\n"
         "                    engram_mask, False,\n"
         "                )\n"
         "                if mhc_stream is not None:\n"
         "                    torch.cuda.current_stream().wait_stream(mhc_stream)\n"
         "                return self.feature_tap(feature)\n"
         "            hidden_states, residual, post_mix, res_mix, pre_mix, previous_aux = layer(\n"),
    ],
}


def main():
    ap = argparse.ArgumentParser(description="Patch vLLM 0.31 for DeepSeek-V4.1-Flash feature extraction")
    ap.add_argument("site", help="site-packages dir of the DeepSeek vLLM environment")
    ap.add_argument("names", nargs="+", help="patches to apply: early_exit")
    ap.add_argument("--diff-out", help="dir to write the resulting diff to")
    a = ap.parse_args()
    # transactional: every hunk is matched and the result compiled in memory before any file is written
    orig, cur = {}, {}
    for name in a.names:
        for rel, old, new in PATCHES[name]:
            if rel not in cur:
                orig[rel] = cur[rel] = open(os.path.join(a.site, rel)).read()
            s = cur[rel]
            if new in s:
                print(f"[{name}] {rel}: already applied"); continue
            n = s.count(old)
            if n != 1:
                sys.exit(f"[{name}] {rel}: expected exactly 1 match, found {n} -- nothing written:\n{old}")
            cur[rel] = s.replace(old, new)
            print(f"[{name}] {rel}: hunk ok")
    for rel, s in cur.items():
        compile(s, rel, "exec")
    for rel, s in cur.items():
        if s == orig[rel]:
            continue
        open(os.path.join(a.site, rel), "w").write(s)
        if a.diff_out:
            os.makedirs(a.diff_out, exist_ok=True)
            d = difflib.unified_diff(orig[rel].splitlines(True), s.splitlines(True), f"a/{rel}", f"b/{rel}")
            open(os.path.join(a.diff_out, "+".join(a.names) + "__" + rel.replace("/", "_") + ".diff"), "w").writelines(d)
        print(f"wrote {rel}")


if __name__ == "__main__":
    main()
