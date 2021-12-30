download-dataset:
	rm -r .data
	mkdir -p .data
	wget -P .data/ http://opihi.cs.uvic.ca/sound/genres.tar.gz
	tar -xf .data/genres.tar.gz -C .data/
	rm .data/genres.tar.gz
	python scripts/create_dataset_csv.py

create-data-csv:
	python scripts/create_dataset_csv.py 

lint:
	black .
