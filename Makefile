# 检测Python版本并设置环境变量
ATLANTIS_PYTHON_VERSION := $(shell python -c 'import sys; print("{}.{}".format(sys.version_info.major,sys.version_info.minor));')
PYTHON ?= python
VERSION ?= py38
VENV_NAME_BASE = atlantis
VENV_NAME := $(VENV_NAME_BASE)-$(VERSION)-base
PYTHON_VERSIONS := py38 py310
ACTIVATE_SCRIPT := bin/activate
SHELL_PATH := $(shell echo $$SHELL)

# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
help:
	@echo "Usage:"
	@echo "  make setup-env VERSION=py38|py310       ; create specific atlantis venv atop of specified python version"
	@echo "  make bootstrap | boot                   ; bootstrap atlantis dependencies"
	@echo "  make install-atlantis | install | build ; install atlantis to the OS env"
	@echo "  make enter-shell | enter | shell        ; enter the atlantis dev shell"

# 设置环境
.PHONY: setup-env
setup-env: check-version create-venv activate-venv

check-version:
ifeq ($(VERSION),py38)
  PYTHON_CMD := python3.8
else ifeq ($(VERSION),py310)
  PYTHON_CMD := python3.10
else
    $(error Unsupported Python version. Use VERSION=py38 or py310)
endif

# 创建虚拟环境
create-venv:
	@echo ">> Creating environment for Python $(VERSION)..."
	@echo ">> Creating venv = "$(VENV_NAME)
	$(PYTHON_CMD) -m venv $(VENV_NAME)

# 激活虚拟环境
activate-venv:
ifeq ($(SHELL_PATH), /bin/fish)
	@echo "Activating environment by invoking command at below:"
	@echo ">> source $(VENV_NAME)/$(ACTIVATE_SCRIPT).fish"
else ifeq ($(SHELL_PATH), /bin/bash)
	@echo "Activating environment by invoking command at below:"
	@echo ">> source $(VENV_NAME)/$(ACTIVATE_SCRIPT)"
else
	@echo "Shell is not bash or fish. Using default shell."
	exit
endif

# 退出环境
.PHONY: exit-env
exit-env:
	@if [ "$$VIRTUAL_ENV" = "$(VENV_NAME)" ]; then \
		echo "Exiting $(VENV_NAME) environment..."; \
		deactivate; \
	else \
		echo "Not in a $(VENV_NAME) environment."; \
	fi



.PHONY: bootstrap 
bootstrap:
	# 检测Python版本并存储为环境变量
	@echo "Detected Python version: $(ATLANTIS_PYTHON_VERSION)"
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
	@poetry lock --no-update

.PHONY: install-atlantis
install-atlantis:
	# 检测Python版本并存储为环境变量
	@echo "Detected Python version: $(ATLANTIS_PYTHON_VERSION)"
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
	@poetry install

.PHONY: enter-shell
enter-shell:
	poetry shell


boot: bootstrap
build: install-atlantis
install: install-atlantis
shell: enter-shell
