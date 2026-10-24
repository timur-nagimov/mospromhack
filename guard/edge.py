"""Compatibility script; the guard.edge package owns the durable CLI."""
from guard.edge import main

if __name__ == "__main__":
    raise SystemExit(main())
