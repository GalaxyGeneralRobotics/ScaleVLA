"""Train the feature mapper and the action head on a feature cache, from one YAML recipe.

    torchrun --nproc_per_node 4 -m scripts.train storage/configs/libero.yaml \
        --data.samples <samples> --data.cache <cache> --run.output_dir <run>

Arguments after the YAML override its fields (--section.field value); --help lists every field.
"""
import os
import sys

from scalevla.training.config import describe, load_config


def main():
    usage = "usage: torchrun ... -m scripts.train <config.yaml> [--section.field value ...]"
    if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
        print(usage + "\n\n" + describe())
        return
    if len(sys.argv) < 2:
        raise SystemExit(usage)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(sys.argv[1], sys.argv[2:])
    from scalevla.training.trainer import train
    train(cfg)


if __name__ == "__main__":
    main()
