# Local validation

Install the test dependencies and run the CPU contract checks:

```bash
pip install -e '.[test]'
pytest -q
```

Tests cover Context preparation, Query-pixel isolation, rejected-prefill cleanup,
endpoint selection, geometry calculations, distributed output ownership and mocked
VQA requests. The README quickstart checks installation, toy generation and pixel
evaluation without loading a native model.

For native validation, prepare a real mini case and follow
[native setup](native_backends.md). Check the local prefill report, continuation
state/frame checks and decoded video before evaluating all six metrics. A
single-case smoke check does not establish coverage for every trajectory or
history length.

`locks/core-tested.txt` and `locks/native-tested.json` record dependency versions.
Native component licenses are listed in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
