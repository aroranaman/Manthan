.PHONY: setup demo test

# This target is now just for documentation, as you manage your own env.
setup:
	@echo "INFO: Setup assumes you have an active Conda env (e.g., 'manthan_gee')."
	@echo "INFO: Install dependencies with: pip install -r requirements.txt"

demo:
	# No need to activate, just run python directly from your active env.
	python intelligent_app.py --aoi assets/sample_aoi.geojson

test:
	# Same here, just run pytest directly.
	pytest -q tests/