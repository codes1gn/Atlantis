# 检测Python版本并设置环境变量
PYTHON ?= python
ATLANTIS_PYTHON_VERSION := $(shell python -c 'import sys; print("{}.{}".format(sys.version_info.major,sys.version_info.minor));')


VENV_NAME_BASE = atlantis
VENV_NAME := $(VENV_NAME_BASE)-$(ATLANTIS_PYTHON_VERSION)-base
PYTHON_VERSIONS := py38 py310
ACTIVATE_SCRIPT := bin/activate
SHELL_PATH := $(shell echo $$SHELL)

.DEFAULT_GOAL := help

help:
	@echo "Usage:"
	@echo "Basic Commands:"
	@echo "  USE: make install-atlantis | install    ; install atlantis to the OS env"
	@echo "  USE: poetry shell                       ; to enter the atlantis dev shell"
	@echo "  USE: deactivate                         ; to exit the atlantis dev shell"
	@echo "Advanced commands:"
	@echo "  make bootstrap | boot                   ; bootstrap atlantis dependencies"

.PHONY: bootstrap 
bootstrap: check-version create-venv activate-venv

check-version:
ifeq ($(ATLANTIS_PYTHON_VERSION),3.8)
  PYTHON_CMD := python3.8
else ifeq ($(ATLANTIS_PYTHON_VERSION),3.10)
  PYTHON_CMD := python3.10
else
    $(error Unsupported Python version. Use VERSION=py38 or py310)
endif

create-venv:
	@echo ">> Creating environment for Python $(ATLANTIS_PYTHON_VERSION)..."
	@echo ">> Creating venv = "$(VENV_NAME)
	$(PYTHON_CMD) -m venv $(VENV_NAME)

# 激活虚拟环境
activate-venv:
ifeq ($(SHELL_PATH), /bin/fish)
	@echo "Activating environment by invoking command at below:"
	@. $(VENV_NAME)/$(ACTIVATE_SCRIPT).fish
else ifeq ($(SHELL_PATH), /bin/bash)
	echo "Activating environment by invoking command at below:"
	. $(VENV_NAME)/$(ACTIVATE_SCRIPT)
else
	@echo "Shell is not bash or fish. Using default shell."
	exit
endif



.PHONY: install-atlantis
install-atlantis: bootstrap
	# 检测Python版本并存储为环境变量
	# . $(VENV_NAME)/$(ACTIVATE_SCRIPT)
	# @echo "Detected Python version: $(ATLANTIS_PYTHON_VERSION)"
	# 根据Python版本复制pyproject.toml文件并安装依赖
ifeq ($(ATLANTIS_PYTHON_VERSION), 3.8)
	@echo "Python 3.8 detected, using pyproject.toml.py38"
	@cp tools/version-control/pyproject.toml.py38 pyproject.toml
else ifeq ($(ATLANTIS_PYTHON_VERSION), 3.10)
	@echo "Python 3.10 detected, using pyproject.toml.py310"
	@cp tools/version-control/pyproject.toml.py310 pyproject.toml
else
	@echo "Error: Unsupported Python version $(ATLANTIS_PYTHON_VERSION)"
	@exit 1
endif
	# 调用poetry install
	pip install poetry
	@poetry lock --no-update
	@poetry install


# Targets Aliases
boot: bootstrap
install: install-atlantis
