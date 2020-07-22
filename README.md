# paper-dark-fishing-fleets-in-north-korea
This repository contains the codes and figures used in the paper "Dark fishing fleets in North Korea", published in *Science Advances* on July 22, 2020. The methodologies and references can be found in the Supplementary materials of the paper
The current repository is orgainzed as per the following structure.

## Readme 
* Description of the paper, repository, and data sets used in the paper 

## Publication of the paper
* Manuscript [Link to be added]
* Supplementary materials [Link to be added]
* Supplementary materials **with page numbers**: Park_abb1197_SM_with_page_numbers.pdf (in this directory)

## Figures
* 1-figure1: Readme and code for Fig. 1A-1F
* 2-figure2: Readme and code for Fig. 2A-2D
* 3-figure3: Readme and code for Fig. 3
* 4-figure4: Readme and code for Fig. 4A-4D
* 5-pdfs_of_all_figures: Figs. 1-4 in PDF formats

## Supplementary-sections 
The codes and figures are stored under directories as per Sections in Supplementary materials. Sections 6, 7, and 9 have no codes used.
* 1-boundary 
* 2-daytime-optical-imagery
* 3-synthetic-aperture-radar
* 4-automatic-identification-system
* 5-nighttime-optical-imagery
* 8-number-of-vessels-and-days-of-fishing

## Libraries required
* All python packages used in the study are available in `requirements.txt`.
* You may want to use `pip install -r requirements.txt` or `conda install --file requirements.txt`.
* You may need to execute the following command first `conda config --append channels conda-forge`.
* Set this environment on Jupyter Notebook, you may need `python -m ipykernel install --user --name=[YOUR_NAME_CHOICE]`.
* If a problem is encountered while installing `gdal` and `basemap` packages, please refer to the following article: https://github.com/conda-forge/basemap-feedstock/issues/43
* Some codes are run on Python 2.7. We will migrate our codes / environment to Python 3.7 in the near future. 
