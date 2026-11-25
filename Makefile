.PHONY: install data run report test lint estimate check judge clean

install:
	pip install -r requirements-dev.txt

data:           # downloads the 6 dataset files and verifies their SHA-256 hashes
	python -m stopret download

run:            # builds trajectories, runs every experiment, writes results/ (about 2 minutes)
	python -m stopret run

report:         # redraw figures and tables from results/metrics.json
	python -m stopret report

test:
	python -m pytest -q

lint:
	ruff check .

estimate:       # no API calls: make estimate N=100 LIMITS="--tokens-per-minute 8000 --tpd 200000"
	python -m stopret judge --dry-run --n $(or $(N),100) $(LIMITS)

check:          # test key + connection, lists model ids: make check BACKEND=groq MODEL=<id>
	python -m stopret check --backend $(BACKEND) $(if $(MODEL),--model $(MODEL),) $(OPTS)

judge:          # make judge BACKEND=cerebras MODEL=<id> N=100 OPTS='--rpm 30 --extra-body ...'
	python -m stopret judge --backend $(BACKEND) --model $(MODEL) --n $(or $(N),100) $(OPTS)

clean:
	rm -rf results/*.png results/metrics.json results/results_tables.md .pytest_cache
