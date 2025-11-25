# rc-grpc-client

This package is a **stub** for a client that would send `ResourceDecision`s
to an RC xApp over gRPC (or another IPC mechanism).

- In this skeleton, it only logs the decisions.
- In a real deployment, you would:
  - Import the gRPC stubs generated from the RC xApp's `.proto` files.
  - Map fields from `ResourceDecision` into the appropriate request messages.
