from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceDecision:
    """Minimal copy of the decision model used by the policy engine.

    In a real system you might import this from `uav_policy.policy_engine`
    instead of duplicating it, but for now we keep this module independent.
    """

    uav_id: str
    target_cell_id: str
    slice_id: Optional[str]
    prb_quota: Optional[int]
    reason: str


class RcGrpcClient:
    """Stub for a client that would talk to an RC xApp over gRPC.

    For now, `apply_decision` just prints to stdout; you can replace this with
    real gRPC calls as needed.
    """

    def apply_decision(self, decision: ResourceDecision) -> None:
        # TODO: wire this to real RC gRPC stubs.
        print(f"[RC] Apply decision for {decision.uav_id}: "
              f"cell={decision.target_cell_id}, slice={decision.slice_id}, "
              f"prb_quota={decision.prb_quota}, reason={decision.reason}")
