SRC = src
EXE = $(SRC)

VENV = .venv
PYTHON = $(VENV)/bin/python3

UV_CACHE = .uv_cache
HF_CACHE = .hf_cache

UV_CACHE_DIR = $(CURDIR)/$(UV_CACHE)
HF_CACHE_DIR = $(CURDIR)/$(HF_CACHE)

UV_SETUP = UV_CACHE_DIR="$(UV_CACHE_DIR)" UV_LINK_MODE=copy
HF_SETUP = HF_HOME="$(HF_CACHE_DIR)" HUGGINGFACE_HUB_CACHE="$(HF_CACHE_DIR)/hub" TRANSFORMERS_CACHE="$(HF_CACHE_DIR)/transformers"

TRASH = $(SRC)/__pycache__ llm_sdk/__pycache__ __pycache__ $(VENV) .mypy_cache data/output .uv_cache data/output

MYPY_FLAGS = --strict --no-error-summary

# ANSI codes
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[0;33m
BOLD = \033[1m

SUCCESS = \033[1;92m
RESET = \033[0m

install:
	@echo "$(BOLD)Initializing project environment:$(RESET)"

	@echo "$(YELLOW)Creating virtual environment...$(RESET)"
	@$(UV_SETUP) $(HF_SETUP) \
		uv venv $(VENV) --clear > /dev/null 2>&1
	@echo "$(GREEN)Virtual environment setup complete$(RESET)"

	@bash -c '\
		$(UV_SETUP) $(HF_SETUP) UV_PROJECT_ENVIRONMENT="$(VENV)" \
		uv sync > /dev/null 2>&1 & pid=$$!; \
		spin="-\|/"; i=0; \
		while kill -0 $$pid 2>/dev/null; do \
			printf "\r$(YELLOW)[%s] Installing dependencies...$(RESET)" "$${spin:$$((i % 4)):1}"; \
			i=$$((i + 1)); \
			sleep 0.1; \
		done; \
		wait $$pid'

	@echo "\n$(GREEN)All dependencies successfully installed$(RESET)"
	@echo "\n$(SUCCESS)Project can now be run safely!$(RESET)"

run:
	@$(HF_SETUP) uv run python -m $(EXE) $(FLAGS)

lint:
	@echo "$(BOLD)Verifying project source code:$(RESET)"

	@echo "$(YELLOW)Running flake8 and mypy...$(RESET)"
	@$(PYTHON) -m flake8 $(SRC)
	@$(PYTHON) -m mypy $(SRC) $(MYPY_FLAGS)
	@echo "$(GREEN)LINT OK$(RESET)"

debug:
	@$(HF_SETUP) $(PYTHON) -m pdb -m $(EXE) $(FLAGS)

clean:
	@echo "$(YELLOW)Initiating cleanup operation...$(RESET)"
	@found=0; \
	for i in $(TRASH); do \
		if [ -d "$$i" ]; then \
			echo " - Removed directory $(RED)'$$i'$(RESET)"; \
			found=1; \
		elif [ "$$i" ]; then \
			echo " - Removed file $(RED)'$$i'$(RESET)"; \
			found=1; \
		fi; \
	done; \
	if [ $$foundq 0 ]; then \
		echo "$(GREEN)Nothing to clean :)$(RESET)"; \
	fi
	@rm -rf $(TRASH)
	@echo "$(SUCCESS)\nCleanup complete!$(RESET)"

cache_clean:
	@echo "$(YELLOW)Initiating cache cleanup operation...$(RESET)"
	@echo "Deleting $(RED)$(UV_CACHE), $(HF_CACHE)$(RESET)"
	@rm -rf $(UV_CACHE_DIR) $(HF_CACHE_DIR)
	@echo "$(SUCCESS)\nCache cleanup complete!$(RESET)"