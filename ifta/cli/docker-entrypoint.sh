#!/usr/bin/env bash

# Try to start a local version memcached, but fail gracefully if we can't.
memcached -u root -d -m 1024 || true

exec /usr/bin/python3 -m slicer_cli_web.cli_list_entrypoint "$@"