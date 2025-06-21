# Dutch European Parliament Verbatim Report Scraper

This repository contains a small scraper that downloads the Dutch verbatim reports of the European Parliament and uploads them as a dataset to the [Hugging Face Hub](https://huggingface.co/).

The scraper starts from the first available report at:
```
https://www.europarl.europa.eu/doceo/document/CRE-4-1996-04-15-TOC_NL.html
```
and follows the "Volgende" links to iterate through the archive. For each page the `-TOC` part is removed to obtain the full report, which is then parsed and cleaned.

The dataset is pushed to the public hub repository **vGassen/Dutch-European-Parliament-Verbatim-Reports**. Set the environment variables `HF_USERNAME` and `HF_TOKEN` before running the script so it can authenticate with the hub.

## Usage

```bash
pip install -r requirements.txt
python scraper.py
```

## License

The code in this repository is released under the MIT License (see the `LICENSE` file). The scraped content remains subject to the European Parliament's reuse policy quoted in the repository description.
