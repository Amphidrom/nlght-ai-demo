# 0001. Publish tagged installable demo releases

- Status: accepted
- Date: 2026-07-20

## Context

The demo gallery needs reproducible versions that can follow nlght-ai platform releases. Repository
snapshots alone do not prove that Python packaging is complete, and the portal depends on static
assets that setuptools does not include automatically.

## Decision

Tags beginning with `v` trigger a GitHub Actions release workflow. The workflow derives the Python
package version from the tag, builds a wheel and source distribution, installs the wheel in a clean
environment, verifies the CLI and packaged portal assets, and publishes both files on a GitHub
Release. Releases remain GitHub-only; PyPI publishing is not part of this decision.

## Consequences

Maintainers get immutable, versioned demo artifacts and an auditable installation check. Tags must
contain a PEP 440-compatible version after the `v` prefix. The build workflow temporarily injects
that version into package metadata, while normal development retains the repository's development
version. GitHub Actions requires `contents: write` for release creation.
