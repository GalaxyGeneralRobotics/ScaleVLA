"""Join feature-cache parts written with scripts.extract_features --part k/n into one cache.

    python -m scripts.merge_features <out_dir> <part_dir> [<part_dir> ...] [--allow-partial]
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--allow-partial", action="store_true", help="the parts may leave samples uncovered (complete = false)")
    a = ap.parse_args()
    from scalevla.features.merge import merge
    merge(a.out, a.parts, a.allow_partial)


if __name__ == "__main__":
    main()
