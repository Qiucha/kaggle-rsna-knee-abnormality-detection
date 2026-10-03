"""Build the navigable research-digest site under digest/site/.

    .venv/bin/python scripts/build_digest_site.py           # build the site (idempotent)
    .venv/bin/python scripts/build_digest_site.py export    # refresh digest/site/data/ snapshots
                                                            # from gitignored checkpoints/, then build

The build reads only digest/* and digest/site/data/*. It fails when a digest/
file has no row in scripts/digest_site/catalog.py. Generated output:
digest/site/index.html, pages/<slug>.html, docs/<name>.html. Hand-owned:
digest/site/assets/*. Snapshots: digest/site/data/*.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from digest_site import build, data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="build", choices=["build", "export"])
    args = parser.parse_args()
    if args.command == "export":
        data.export()
    build.build()


if __name__ == "__main__":
    main()
