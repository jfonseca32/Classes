.PHONY: check clean format install lint test test-cpp test-java test-python install-me4710 update-me4710 kernel-me4710 jupyter-me4710


UV ?= uv
PYTEST ?= $(UV) run pytest
PREK ?= $(UV) run prek
RUFF ?= $(UV) run ruff
CXX ?= g++
JAVA ?= java
MAVEN ?= mvn
GRADLE ?= ./gradlew
ME4710_DIR := courses/ME-4710_Foundations_of_ML_for_Engineers
ME4710_ENV := $(ME4710_DIR)/environment.yml

install:
	$(UV) sync --dev
	git lfs install
	$(PREK) install

lint:
	$(PREK) run --all-files

format:
	$(RUFF) format .

test: test-python test-cpp test-java

test-python:
	@if git ls-files --cached --others --exclude-standard -- '*.py' | grep -Eq '(^|/)(test_[^/]*|[^/]*_test)\.py$$'; then \
		$(PYTEST); \
	else \
		echo "No Python tests found."; \
	fi

test-cpp:
	@if find . -path './.git' -prune -o \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print -quit | grep -q .; then \
		find . -path './.git' -prune -o \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0 | xargs -0 -n1 $(CXX) -std=c++20 -Wall -Wextra -pedantic -fsyntax-only; \
	else \
		echo "No C++ sources found."; \
	fi

test-java:
	@if [ -x ./gradlew ]; then \
		$(GRADLE) test; \
	elif [ -f pom.xml ]; then \
		$(MAVEN) test; \
	else \
		echo "No Java build found."; \
	fi

check: lint test

clean:
	find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name 'htmlcov' -o -name 'build' -o -name 'dist' -o -name 'target' -o -name '.gradle' \) -prune -exec rm -rf {} +
	find . -type f \( -name '.coverage' -o -name '*.pyc' -o -name '*.class' -o -name '*.o' -o -name '*.out' \) -delete

install-me4710:
	conda env create --file $(ME4710_ENV)
	conda run --name me4710 python -m ipykernel install --user --name me4710 --display-name "Python (ME 4710)"

update-me4710:
	conda env update --name me4710 --file $(ME4710_ENV) --prune
	conda run --name me4710 python -m ipykernel install --user --name me4710 --display-name "Python (ME 4710)"

kernel-me4710:
	conda run --name me4710 python -m ipykernel install --user --name me4710 --display-name "Python (ME 4710)"

jupyter-me4710:
	conda run --no-capture-output --name me4710 jupyter lab $(ME4710_DIR)/Homeworks/HW1
