"""Pre-compile the Numba kernels so the first request does not pay JIT cost.

Run this during the deploy build:

    pip install -r requirements.txt && python -m dsp.warmup

Numba writes the compiled machine code next to the module; the deploy artifact
ships it, so the running instance loads the cache instead of compiling.  On a
0.1-CPU free instance the compile alone takes over a minute, which is enough to
time out the first request.
"""

import numpy as np

from dsp.lms import NUMBA_AVAILABLE, lms, nlms, rls


def main() -> None:
    order = 8
    n = 64
    x = np.zeros(n)
    d = np.zeros(n)
    lms(x, d, 0.01, order)
    nlms(x, d, 0.5, order)
    rls(x, d, 0.99, order)
    print(f"numba available: {NUMBA_AVAILABLE}")


if __name__ == "__main__":
    main()
