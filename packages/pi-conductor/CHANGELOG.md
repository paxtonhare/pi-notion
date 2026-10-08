# Changelog

## [Unreleased]

### Fixed

- Support Pi 1.1.0's ModelRuntime and resource-loader contracts for headless worker preflight and persistent sessions. Preserve the configured Pi auth location and exact worker tool allowlist.
- Keep cancellation effective during asynchronous runtime/session setup and report pending or deferred model responses as incomplete runs.

### Changed

- Use current Earendil Pi imports and test against Pi 1.1.0. Conductor remains opt-in; the repository's normal install still loads only Notion.
