from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceDecision:
    uav_id: str
    target_cell_id: Optional[str] = None
    slice_id: Optional[str] = None
    prb_quota: Optional[int] = None
    notes: str = ""


class RcGrpcClient:
    def __init__(self, host: str = "rc-xapp.ricxapp", port: int = 50051) -> None:
        self.host = host
        self.port = port

    def apply_decision(self, decision: ResourceDecision) -> None:
        print(f"[RC-CLIENT] Would send decision to RC xApp @ {self.host}:{self.port}: {decision}")
