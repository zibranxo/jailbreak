# Contributing Guidelines

Thank you for your interest in contributing to the Hybrid LLM Safety System!

## Getting Started

1. Ensure you have Python 3.9+ installed.
2. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up pre-commit hooks to format and lint your code automatically:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

1. Create a branch for your feature or bugfix.
2. Write tests for your changes.
3. Run tests locally: `pytest tests/ -v`
4. Ensure linting passes: `black .`, `flake8 .`, `mypy .`
5. Submit a Pull Request.

## Pull Request Process

- Ensure the CI pipeline passes.
- Every PR must maintain or improve test coverage.
- Update `CHANGELOG.md` with your changes.
- Add/update relevant documentation.

## Code Style

- Use `black` for formatting.
- Follow `flake8` for style checking.
- Use type hints (`mypy` compliant).
