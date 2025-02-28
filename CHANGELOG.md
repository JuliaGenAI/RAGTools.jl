# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Progress bar for `get_embeddings` given it can take a while for large documents (new dependency `ProgressMeter.jl`, but extremely lightweight)

### Fixed

## [0.1.1]

### Fixed
- Fixed `preprocess_tokens`, `get_tags`, and `get_embeddings` to not trigger package extension checks (leftover from carve out from PromptingTools)
- Clean up docs references to `PromptingTools.Experimental.RAGTools`

## [0.1.0]

### Added
- Initial release of RAGTools.jl, simple carve-out of module RAGTools.jl from PromptingTools.jl.
