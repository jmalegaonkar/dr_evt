# dr_evt_market

`dr_evt_market` provides platform adapters for a federated market over DR_EVT.
It supports in-process simulations and simulations served over gRPC.

From the repository root, install the package and gRPC dependencies with:

```bash
python3 -m pip install -e "python[grpc]"
```

Run the package tests with:

```bash
./tests/run_market_tests.sh
```

See the [Market Python API](../../docs/api/MARKET_PYTHON_API.md) for the full
contract, adapter lifecycle, and testing details.
