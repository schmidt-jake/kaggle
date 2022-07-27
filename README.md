# kaggle

Kaggle competition monorepo

[![.github/workflows/cicd.yaml](https://github.com/schmidt-jake/kaggle/actions/workflows/cicd.yaml/badge.svg)](https://github.com/schmidt-jake/kaggle/actions/workflows/cicd.yaml)

## Getting Started

First, create a Python virtual environment at `./venv`. Here's one way to do this:

```bash
python -m venv venv --prompt=kaggle
```

Then, activate your environment (`source venv/bin/activate`) and install requirements:

```bash
pip install -r requirements/main.txt -r requirements/eda.txt -r requirements/dev.txt
```

### VSCode Powerups

If using VSCode, consider adding the following configuration to your workspace settings to enable some powerups:

```json
{
    "git.branchProtection": [
        "main"
    ],
    "python.formatting.provider": "black",
    "python.formatting.blackPath": "${workspaceFolder}/venv/bin/black",
    "python.formatting.blackArgs": [
        "--config=${workspaceFolder}/pyproject.toml"
    ],
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Path": "${workspaceFolder}/venv/bin/pflake8",
    "python.linting.flake8Args": [
        "--config=${workspaceFolder}/pyproject.toml"
    ],
    "python.sortImports.args": [
        "--settings-path=${workspaceFolder}/pyproject.toml"
    ],
    "editor.formatOnSave": true,
    "python.linting.mypyEnabled": true,
    "python.linting.mypyPath": "${workspaceFolder}/venv/bin/mypy",
    "python.linting.mypyArgs": [
        "--config-file=${workspaceFolder}/pyproject.toml"
    ],
    "python.linting.banditEnabled": true,
    "python.linting.banditPath": "${workspaceFolder}/venv/bin/bandit",
    "json.schemas": [
        {
            "fileMatch": [
                "*/dataset-metadata.json"
            ],
            "url": "https://specs.frictionlessdata.io/schemas/data-package.json"
        },
    ],
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
}
```

### Pre-commit hooks (optional)

You can enable pre-commit hooks with:

```bash
pre-commit install --install-hooks
```

## To-Do

- further filter ROIs that are empty-ish. Some crops are coming up empty in the dataset and we can't find a valid random crop.
- Add tensorboard to training loop
- Create a validation set (or cross-validation)
- Create an inference loop
- Create an automated submission script
- [EDA to-do](mayo_clinic_strip_ai/eda/README.md)
