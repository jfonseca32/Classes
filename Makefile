.PHONY: check clean format install lint test test-cpp test-java test-python

UV ?= uv
PYTEST ?= $(UV) run pytest
PREK ?= $(UV) run prek
RUFF ?= $(UV) run ruff
CXX ?= g++
JAVA ?= java
MAVEN ?= mvn
GRADLE ?= ./gradlew

install:
	$(UV) sync --dev
	git lfs install

lint:
	$(PREK) run --all-files

format:
	$(RUFF) format .

test: test-python test-cpp test-java

test-python:
	@if find . -path './.venv' -prune -o -path './.git' -prune -o -name 'test_*.py' -print -quit | grep -q .; then \
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
