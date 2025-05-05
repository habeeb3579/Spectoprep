Changelog
=========

All notable changes to this project will be documented here.

1.0.1-alpha - 2025-05-05
----------------------

Features
~~~~~~~~
* feat(ci): support manual and automatic GitHub releases with full artifact upload and changelog generation (1ebb93f)

Bug Fixes
~~~~~~~~~
* fix: update workflow to ensure conda-compatible version by editing Jinja set statement in meta.yaml (4cac660)
* fix(ci): normalize pre-release versions for conda compatibility (c857b20)
* fix(ci): normalize pre-release versions for conda compatibility (5a5a6a0)
* fix(release): handle missing README.md by checking README.rst and improve release file preparation robustness (ab4ab8d)

Other Changes
~~~~~~~~~~~~~
* docs: update CHANGELOG for version 1.0.0 (75bf31f)
* chore: update Jinja version set line in meta.yaml via sed and enable workflow_dispatch for manual GitHub release runs (c546e7f)
* chore(ci): increase sleep to 10 minutes, add 🚀 and 📝 icons (b404c5d)
* chore: add env controls for build wait job with workflow_dispatch inputs (c79162e)
* chore: add env controls for build wait job with workflow_dispatch inputs (4c2a7c5)


1.0.1 - 2025-05-05
----------------

Chore
~~~~~
* chore: bump version to 1.0.1-alpha (6e8eb73)

