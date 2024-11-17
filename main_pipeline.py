import argparse

from grant_call_scraping import main as gcs_main
from abstracts_scraping import main as as_main
from embedding import main as emb_main
from create_qdrant_collections import main as cqc_main


def main():
    parser = argparse.ArgumentParser(description="Running this file will run the entire pipeline required for the data files creation (scraping -> embedding -> creating collections)"
                                                 "with the deafult options (default file names, paths, collection sizes, both models will be used, etc..).\n"
                                                 "Make sure you have the default grant calls list (12_11_2023-AllGuideResultsReport.xlsx) in the project folder before running.")
    parser.parse_args()

    gcs_main()
    as_main()
    emb_main(pipeline=True)
    cqc_main(pipeline=True)
    print("---finished data files creation pipeline---")


if __name__ == '__main__':
    main()
