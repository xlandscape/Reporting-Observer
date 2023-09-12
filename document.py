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
base.documentation.write_repository_info(
    os.path.join(root_folder, "..", "variant", "ReportingObserver"),
    os.path.join(root_folder, "..", "variant", "ReportingObserver", "repository.json"),
    os.path.join(root_folder, "..", "..", "..", "versions.json"),
    "component"
)
