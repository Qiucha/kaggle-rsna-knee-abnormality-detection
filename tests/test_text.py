from __future__ import annotations

from contrastive_learning.text import select_report_view


def test_report_views_extract_findings_and_impression_without_technique() -> None:
    report = "TECHNIQUE: MRI without contrast\nFINDINGS: ACL tear.\nIMPRESSION: Complete ACL tear."

    assert select_report_view(report, "impression") == "Complete ACL tear."
    assert select_report_view(report, "findings_impression") == (
        "FINDINGS: ACL tear.\nIMPRESSION: Complete ACL tear."
    )


def test_report_view_recognizes_spanish_headings_and_falls_back_to_full_text() -> None:
    spanish = "TÉCNICA: RM de rodilla\nHALLAZGOS: Derrame.\nIMPRESIÓN: Derrame articular."

    assert select_report_view(spanish, "impression") == "Derrame articular."
    assert select_report_view("unstructured report", "impression") == "unstructured report"
