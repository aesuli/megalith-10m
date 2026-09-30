# Megalith-10M Downloader

A resumable downloader for the [Megalith-10M](https://huggingface.co/datasets/madebyollin/megalith-10m) dataset. 
The dataset contains image URLs in ten Parquet shards; this project downloads the linked Flickr images locally and tracks progress in SQLite.

## Installation

Python 3.9 or newer is required.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Download a limited number of images:

```bash
python download_megalith10m.py --output-dir ./megalith_images --limit 50000
```

Download all available images:

```bash
python download_megalith10m.py --output-dir ./megalith_images
```

The command is safe to run again. 
Completed rows are read from `progress.sqlite3`, cached Parquet shards are reused, and existing valid image files are skipped.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--output-dir PATH` | `./megalith_images` | Directory for images, state, and Parquet cache |
| `--workers N` | `1` | Number of parallel download threads |
| `--limit N` | all | Stop after `N` successful downloads |
| `--shards N ...` | `0` through `9` | Process only the selected shard indices |
| `--highres` | disabled | Use the higher-resolution `url_highres` field |
| `--min-bytes N` | `2048` | Minimum response size accepted as an image |
| `--timeout SECONDS` | `15` | Per-request timeout |
| `--user-agent TEXT` | project default | HTTP user-agent sent to Flickr |

For example, download only shards 0 and 1 using higher-resolution URLs:

```bash
python download_megalith10m.py --output-dir ./megalith_images --shards 0 1 --highres
```

## Output layout

```text
megalith_images/
├── images/
│   └── 00/ ... 99/       # downloaded images, bucketed by photo ID
├── parquet_cache/        # cached Hugging Face Parquet shards
└── progress.sqlite3      # resumable download state
```

Failed requests are recorded and can be retried by running the command again. 

## Data and usage notices

This software and the Megalith-10M dataset are totally independent entities.
This software downloads third-party content referenced by the Megalith-10M dataset.
The dataset metadata, Flickr images, and image URLs may have separate licenses, attribution requirements, and access restrictions. 
Review the [dataset page](https://huggingface.co/datasets/madebyollin/megalith-10m), the original Flickr pages, and applicable terms before redistributing or using downloaded content.

Please use reasonable request rates and storage. 
The default concurrency is configurable, yet increase it only when appropriate for your network and the policies of the data sources.

## License

[See the LICENSE file](LICENSE).