# MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos

Repositorio del curso MDS7202 (Otoño 2026), Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile.

## Integrantes

| Nombre | GitHub |
|--------|--------|
| Agustín Solís | [@TinSolis](https://github.com/TinSolis) |
| Lorenzo Gao | [@LoverIsLife](https://github.com/LoverIsLife) |

## Estructura del repositorio

.
├── .github/
│   ├── workflows/
│   │   └── lint.yml
│   └── pull_request_template.md
├── labs\lab_1
│   ├── lab_1/
│   └── ...
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
└── uv.lock

## Configuración del entorno

uv sync --locked --all-groups
uv run pre-commit install
