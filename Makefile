.PHONY: all doc

#############
## Print help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
 match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
 if match:
  target, help = match.groups()
  print("%-25s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

##############
## python pkg

install_py_pkg: ## install package in debug mode
	@pip install -e .

py_test: ## test package
	@python -m pytest ./pycovid19xray/

