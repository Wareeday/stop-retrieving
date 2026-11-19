.PHONY: install data run report test judge clean

install:
	pip install -r requirements-dev.txt

data:
	python -m stopret download

run:            # builds trajectories, runs every experiment, writes results/ (about 2 minutes)
	python -m stopret run

report:         # redraw figures and tables from results/metrics.json
	python -m stopret report

test:
	python -m pytest -q

BACKEND ?= ollama
judge:          # optional: make judge BACKEND=gemini MODEL=<id>   (needs GROQ_API_KEY or GEMINI_API_KEY)
	python -m stopret judge --backend $(BACKEND) --model $(MODEL) --n 100

clean:
	rm -rf results/*.png results/*.json results/results_tables.md .pytest_cache
