# Makes tests/ an importable package so that `python -m unittest discover`
# works from the repository root and with an explicit top-level directory
# (`-t .`), not only when tests/ is itself the top-level directory.
# The sys.path.append() at the top of every test module stays necessary: it is
# what lets the modules import common/solver_classic when they are run directly.
