pre-commit:
	pre-commit run

pre-commit-all:
	pre-commit run --all-files

cli:
	xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" python cli.py repl
