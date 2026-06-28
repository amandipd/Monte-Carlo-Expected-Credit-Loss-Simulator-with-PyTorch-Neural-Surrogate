"""Tests for executive report synthesis."""
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.report_synthesizer import mock_synthesize_report, synthesize_report


def test_mock_synthesize_report_mentions_ecl():
    report = mock_synthesize_report(
        "Severe recession with rising unemployment",
        10.0,
        7.0,
        80.0,
        5_000_000_000.0,
    )
    assert "5,000,000,000" in report
    assert "unemployment" in report.lower()


def test_synthesize_report_mock_mode():
    report = synthesize_report(
        "Rate hike shock",
        4.0,
        8.0,
        95.0,
        1_000_000.0,
        mock=True,
    )
    assert "Rate hike shock" in report
