#!/bin/bash
# shellcheck disable=SC2034  # everything here is consumed by the steps that source it
# lib/bash/references.sh — the shared reference paths, for bash.
#
# This is a thin wrapper. The definitions live in lib/python/utils/references.py,
# which also fetches them (`cli.py download-references`), so the paths the
# pipeline reads and the paths the downloader writes cannot drift -- there is
# only one set. Same arrangement config.sh has with config.yaml.
#
# Override the root:  export REFERENCE_ROOT=/path/to/Data
#
# Rendered before any conda env exists, so references.py imports only the
# standard library. It still needs python >= 3.9 to parse (walrus, PEP 604),
# which the system python3 on some clusters is not -- bootstrap_python (defined
# in common.sh, which sources this file) probes for one.

_refs_python="$(bootstrap_python)" || {
    echo "ERROR: need python >= 3.9 on PATH to resolve the reference paths." >&2
    echo "  Set BOOTSTRAP_PYTHON to one, or activate an environment first." >&2
    return 1
}
_refs_shell="$("${_refs_python}" "${REPO_ROOT}/lib/python/utils/references.py" export)" || {
    echo "ERROR: could not resolve the reference paths" >&2
    return 1
}
eval "${_refs_shell}"
unset _refs_shell _refs_python
