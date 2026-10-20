from __future__ import annotations

from contrastive_learning.text import select_report_view


def test_impression_first_keeps_findings_and_handles_inline_and_plural_headings() -> None:
    for separator in ("\n\n", " "):
        report = f"Findings: Medial tear. Normal ACL.{separator}Impressions: Meniscal tear."
        ordered = select_report_view(report, "impression_first")
        assert ordered == "Impressions: Meniscal tear.\n\nFindings: Medial tear. Normal ACL."
        assert sorted(ordered.split()) == sorted(report.split())


def test_impression_first_preserves_early_summary_and_ambiguous_reports() -> None:
    reports = [
        "Impression: ACL tear.\nTechnique: PD.\nFindings: Torn ACL.",
        "No heading, but the impression is of a tear.",
        "Findings: Tear.\nImpression: Tear.\nConclusion: Tear.",
        "Findings: The clinical impression: injury is unconfirmed.",
    ]
    for report in reports:
        assert select_report_view(report, "impression_first") == report


def test_impression_first_keeps_negation_and_contradictory_findings() -> None:
    report = "Findings: Medial tear.\nConclusion: No significant abnormality."
    assert select_report_view(report, "impression_first") == (
        "Conclusion: No significant abnormality.\n\nFindings: Medial tear."
    )


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
