VERSION?=0.2.0

.PHONY: clean docs format

docs:
	pydoctor \
	--project-name=nlp-project	\
	--project-version=$(VERSION) \
	--project-url=https://github.com/davidboening/nlp-project/ \
	--make-html \
	--html-output=docs \
	--project-base-dir="utils" \
	--docformat=numpy \
	--intersphinx=https://docs.python.org/3/objects.inv \
	./ddnn

format:
	black --preview .

clean:
	rm -rf docs/*
	python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"