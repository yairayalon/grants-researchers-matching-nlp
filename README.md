# Grant2Research: NLP-Powered Matching of Grants and Research Abstracts

<video src="media/grant2research_ui_demo.mp4" controls="controls" style="max-width: 100%;">
Your browser does not support the video tag.
</video>

## Overview

The Grant2Research system matches grant call solicitations from
the National Institutes of Health (NIH) with research abstracts from Semantic
Scholar, a free academic search engine, using natural language processing (NLP)
models. The process starts by scraping grant calls and abstracts, which are
then embedded using NLP models to capture their semantic meaning. These
embeddings are stored in a Qdrant vector database, which enables efficient
similarity-based matching. A GUI-based application allows users to input URLs
or free text, customize search parameters, and easily navigate the results,
streamlining the process of identifying grant applicants and funding
opportunities.

## Table of Contents

1. [Overview](#overview)
2. [How to Run the Matching System](#how-to-run-the-matching-system)
3. [Data Files Created During main_pipeline.py Run](#data-files-created-during-main_pipelinepy-run)
4. [How to Use the Matching System](#how-to-use-the-matching-system)
    - [Embedding Model Customization](#embedding-model-customization)
    - [Number of Matches Customization](#number-of-matches-customization)
    - [Grant Call Matching by URL](#grant-call-matching-by-url)
    - [Abstract Matching by URL](#abstract-matching-by-url)
    - [Free Text Matching](#free-text-matching)
    - [Navigation between Matches](#navigation-between-matches)
5. [Workflow of the Matching System](#workflow-of-the-matching-system)
6. [Learn More](#learn-more)

## How to Run the Matching System

Follow these steps to set up and run the GUI-based matching system:

- If `main_pipeline.py` was not run before:
    - Open an environment with Python version 3.8 and above on the Cluster (or
      on another high-speed processing computing infrastructure).
    - Install the required Python packages from the `requirements.txt` file.
    - Run `main_pipeline.py` to execute the entire pipeline required for the
      data files creation (scraping -> embedding -> creating collections). The
      data files that will be created during the pipeline run are listed in
      the 'Data Files Created During `main_pipeline.py` Run' section below.
- Transfer (if you have not already transferred) the project's files, including
  the data files (created by a previous run of `main_pipeline.py`), to a local
  computer.
- Open an environment with Python version 3.8 and above on the local computer.
- Install the required Python packages from the `requirements.txt` file.
- Run `matching.py` to use the GUI-based matching system.

> It is recommended to run `main_pipeline.py` on a high-speed processing
> computing infrastructure (such as the Cluster) because it takes a lot of
> processing power for `embedding.py`, which is part of the pipeline, to run.

> It is recommended to run `matching.py` on a local computer because this way
> the interactive GUI of the matching system is much faster for use than if it
> were run remotely (running a GUI remotely slows down its performance).

## Data Files Created During `main_pipeline.py` Run

The data files that will be created (chronologically) during
the `main_pipeline.py` run are:

- `grant_calls.json`
- `grant_calls.xlsx`
- `abstracts.json`
- `abstracts.xlsx`
- `SpecterGrantCallsEmbeddings.json`
- `SpecterAuthorsAbstractsEmbeddings.json`
- `AllMpnetGrantCallsEmbeddings.json`
- `AllMpnetAuthorsAbstractsEmbeddings.json`
- `qdrant_collections` directory - with directories and files in it

> The files `grant_calls.xlsx` and `abstracts.xlsx` neatly summarize the
> filtered grant calls and the filtered abstracts respectively that can be used
> within the matching system. Each of these files will be created if the code
> line responsible for its creation, in the bottom of `grant_call_scraping.py`
> or
> in the bottom of `abstracts_scraping.py` respectively, is uncommented - make
> sure a file named `grant_calls.xlsx` or a file named `abstracts.xlsx`
> respectively is not open on the computer before running this code line.

## How to Use the Matching System

After successfully running the matching system following the steps in the 'How
to Run the Matching System' section above, the GUI-based matching system will
be displayed, offering the following functionalities:

### Embedding Model Customization

Choose the embedding model for matching, in the 'Embeddings' field.

### Number of Matches Customization

Choose the number of matches, sorted by match rating, in the 'Matches #' field.

### Grant Call Matching by URL

Enter a grant call URL (found in `grant_calls.xlsx`) in the 'From NIH URL'
field. Click 'Match' to find matching abstracts.

### Abstract Matching by URL

Enter a paper URL (found in `abstracts.xlsx`) in the 'From SemScho URL' field.
Click 'Match' to find matching grant calls.

### Free Text Matching

Enter text in either of the two 'From text' fields - one for grant calls and
the other for abstracts. Click 'Match' to find matches.

### Navigation between Matches

Navigate through the matches using the 'Previous' and 'Next' buttons.

## Workflow of the Matching System

- `grant_call_scraping.py` and `abstracts_scraping.py`: Retrieve filtered grant
  calls and papers' data from the internet, and store it in JSON (and Excel if
  desired) files.
- `embedding.py`: Processes the generated JSON files and creates new files with
  embeddings for the textual parts, using SPECTER and/or all-mpnet-base-v2
  models (which are being used within wrapper classes in `models.py`).
- `create_qdrant_collections.py`: Creates and stores a Qdrant object with the
  desired embedding collections.
- `main_pipeline.py`: Runs the entire pipeline required for the data files
  creation (scraping -> embedding -> creating collections).
- `matching.py`: Runs the matching program supplied with the given Qdrant
  object.

## Learn More

For more information about some of the files and to change their execution
configurations, use the `--help` flag through the Terminal on each of the
following Python files:

- `grant_call_scraping.py`
- `abstracts_scraping.py`
- `embedding.py`
- `create_qdrant_collections.py`
- `main_pipeline.py`
- `matching.py`
