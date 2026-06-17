from typing import TypedDict, Optional


class RaceReportState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    # Input
    season: int           # e.g. 2024
    round_number: int     # e.g. 5
    circuit_name: str     # e.g. "Monaco"

    # Node outputs
    raw_data: dict                    # Data Agent output
    historical_context: str           # RAG Agent output
    analysis: str                     # Analysis Agent output
    report: str                       # Report Agent final output

    # Metadata
    error: Optional[str]              # Set by any node on failure
    s3_report_url: Optional[str]      # Set after S3 upload
