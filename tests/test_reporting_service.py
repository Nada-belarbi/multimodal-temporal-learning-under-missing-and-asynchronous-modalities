import json
import tempfile
import unittest
from pathlib import Path

from mmt.services.reporting import ReportServiceError, render_saved_report


def write_evaluation(directory: Path, *, completed=True):
    directory.mkdir()
    (directory / "status.json").write_text(
        json.dumps(
            {
                "status": "completed" if completed else "incomplete",
                "rows": 2,
            }
        )
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "counts": {"eligible": 2, "computable": 2},
                "quality_S": {"macro_f1": 0.5, "balanced_accuracy": 0.5},
                "probability": {"nll": 0.4, "brier": 0.2, "ece": 0.1},
                "coverage": {
                    "availability_coverage": 1.0,
                    "accepted_coverage": 1.0,
                    "selective_risk": 0.5,
                },
            }
        )
    )


class ReportingServiceTests(unittest.TestCase):
    def test_report_reads_saved_metrics_and_defaults_to_not_qualified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "eval"
            write_evaluation(evaluation)

            report = render_saved_report(
                {"clean": evaluation},
                root / "report.html",
                "Saved fixture",
                limitations=["No real dataset"],
            )
            html = report.read_text()

        self.assertIn("implementation_verified = True", html)
        self.assertIn("predictive_qualification = not_qualified", html)
        self.assertIn("No real dataset", html)

    def test_report_rejects_incomplete_evaluation_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "incomplete"
            write_evaluation(incomplete, completed=False)
            with self.assertRaises(ReportServiceError):
                render_saved_report({"clean": incomplete}, root / "bad.html", "bad")

            complete = root / "complete"
            write_evaluation(complete)
            output = root / "report.html"
            render_saved_report({"clean": complete}, output, "first")
            with self.assertRaises(FileExistsError):
                render_saved_report({"clean": complete}, output, "second")


if __name__ == "__main__":
    unittest.main()
