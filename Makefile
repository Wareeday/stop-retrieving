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

judge:          # optional, needs a running Ollama server: make judge MODEL=llama3.2:3b
	python -m stopret judge --model $(MODEL) --n 200

clean:
	rm -rf results/*.png results/*.json results/results_tables.md .pytest_cache
