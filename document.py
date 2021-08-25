"""
Script for documenting the code of the ReportingObserver.
"""
import os
import base.documentation
import ReportingObserver

root_folder = os.path.abspath(os.path.join(os.path.dirname(base.__file__), ".."))
base.documentation.write_changelog(
    "ReportingObserver",
    ReportingObserver.ReportingObserver.VERSION,
    os.path.join(root_folder, "..", "variant", "ReportingObserver", "CHANGELOG.md")
)
base.documentation.write_contribution_notes(
    os.path.join(root_folder, "..", "variant", "ReportingObserver", "CONTRIBUTING.md"))
