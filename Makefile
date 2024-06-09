#!/bin/bash

# 检测Python版本并设置环境变量
CUR_PYVERSION := $(shell python -c 'import sys; print("py{}{}".format(sys.version_info.major,sys.version_info.minor));')
PYVERSION ?= $(CUR_PYVERSION)
VENV_NAME_BASE = atlantis
VENV_NAME := $(VENV_NAME_BASE)-$(PYVERSION)-base
ACTIVATE_SCRIPT := bin/activate
# 检测 Fish shell 是否可用
ifeq (, $(shell which fish))
  RUN_SHELL = bash
else
  RUN_SHELL = fish
endif

.DEFAULT_GOAL := help

help:
	@echo "Usage:"
	@echo "Basic Commands:"
	@echo "  USE: make install-atlantis | install    ; install atlantis to the OS env"
	@echo "  USE: poetry shell                       ; to enter the atlantis dev shell"
	@echo "  USE: deactivate                         ; to exit the atlantis dev shell"
	@echo "Advanced commands:"
	@echo "  make bootstrap | boot                   ; bootstrap atlantis dependencies"
	@echo "You can specify python version with cmd in form of: PYVERSION=py38 | py310"

.PHONY: bootstrap 
bootstrap: check-version create-venv activate-venv
ifeq ($(PYVERSION), py38)
	@echo "Python 3.8 detected, using pyproject.toml.py38"
	@cp tools/version-control/pyproject.toml.py38 pyproject.toml
else ifeq ($(PYVERSION), py310)
	@echo "Python 3.10 detected, using pyproject.toml.py310"
	@cp tools/version-control/pyproject.toml.py310 pyproject.toml
else
	@echo "Error: Unsupported Python version $(PYVERSION)"
	@exit 1
endif

check-version:
ifeq ($(PYVERSION), py38)
  PYTHON_CMD := python3.8
else ifeq ($(PYVERSION), py310)
  PYTHON_CMD := python3.10
else
    $(error Unsupported Python version. Use VERSION=py38 or py310)
endif

create-venv:
	@echo ">> Creating environment for Python $(PYVERSION)..."
	@echo ">> Creating venv = "$(VENV_NAME)
	$(PYTHON_CMD) -m venv $(VENV_NAME)

# 激活虚拟环境
activate-venv:
ifeq ($(SHELL_PATH), /bin/fish)
	@echo "Activating environment by invoking command at below:"
  @. $(VENV_NAME)/$(ACTIVATE_SCRIPT).fish
else ifeq ($(SHELL_PATH), /bin/bash)
	@echo "Activating environment by invoking command at below:"
	@. $(VENV_NAME)/$(ACTIVATE_SCRIPT)
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
	# 调用poetry install
	@pip install poetry
	@poetry env use $(shell which $(PYTHON_CMD))
	@poetry lock --no-update
	@poetry install

.PHONY: enter-dev-env exit-dev-env
enter-dev-env:
	@echo "Activating environment"
ifeq ($(RUN_SHELL), fish)
	@fish tools/python-env/enter-dev-env.fish && cd $(word 2, $(MAKECMDGOALS))
else
	@. tools/python-env/enter-dev-env.sh && cd $(word 2, $(MAKECMDGOALS))
endif

exit-dev-env:
	deactivate

.PHONY: boost install enter
# Targets Aliases
boot: bootstrap
install: install-atlantis
enter: enter-dev-env
exit: exit-dev-env
