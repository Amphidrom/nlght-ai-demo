# Versioned releases

The demo uses annotated Git tags such as `v0.2.0` as immutable version boundaries. A matching tag
push triggers `.github/workflows/release.yml` and creates a GitHub Release containing:

- a platform-independent Python wheel for installation;
- a source distribution for source-level reproduction;
- generated release notes from GitHub.

The workflow derives Python version `0.2.0` from tag `v0.2.0`, validates its format, builds both
artifacts, installs the wheel in a clean virtual environment, and checks the `nlght-ai-demo` console
entry point plus all portal static assets. A failed build or installation check prevents publishing.

GitHub Releases are the distribution boundary for the demo; this workflow does not publish to
PyPI. See `README.md` for tag creation, artifact download, installation, and runtime prerequisites.
