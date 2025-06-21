# Dutch European Parliament Adopted Texts Scraper

This repository contains a simple scraper that downloads the adopted texts ("Aangenomen teksten") of the European Parliament in Dutch and uploads them to the [Hugging Face Hub](https://huggingface.co/).

The scraper starts from the first available adopted text at:
```
https://www.europarl.europa.eu/doceo/document/TA-5-1999-07-21-TOC_NL.html
```
and follows the "Volgende" links to iterate through the archive. For each page the `-TOC` part is removed to obtain the full text, which is then parsed and cleaned.

The dataset is pushed to the public hub repository **vGassen/Dutch-European-Parliament-Adopted-Texts**. Set the environment variables `HF_USERNAME` and `HF_TOKEN` before running the script so it can authenticate with the hub.

## Usage
```bash
pip install -r requirements.txt
python scraper.py
```

## License

The code in this repository is released under the MIT License (see the `LICENSE` file). The scraped content remains subject to the European Parliament's reuse policy quoted in the repository description.
