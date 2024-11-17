import requests
import json
import pandas as pd
import argparse
from constants import API_KEY, ABSTRACTS_PATH


def get_abstract_from_url(url, api_k=API_KEY):
    """
    Tries to get the abstract of a paper by its URL.
    :raise Exception when the paper itself of its abstract could not be retrieved
    :param url: The paper's URL
    :param api_k: API key for Semantic Scholar
    :return: The abstract of the paper
    """
    headers = {
        "x-api-key": api_k
    }
    paper_id = [url.split('/')[-1]]
    req = requests.post(
        'https://api.semanticscholar.org/graph/v1/paper/batch',
        headers=headers,
        params={'fields': "abstract"},
        json={"ids": paper_id}
    )
    if req.status_code == 200:
        cur_json = req.json()
        abstract = cur_json[0]['abstract']
        if abstract:
            return abstract
        else:
            raise Exception("Could not extract abstract from paper")
    else:
        raise Exception("Could not retrieve paper from URL")


def main():
    parser = argparse.ArgumentParser(description="Running this file will use the Semantic Scholar api to retrieve the data of the given minimum number of papers (and a little more).\n"
                                                 "Specifically, the papers' id, title, abstract, url and first and last authors ids, urls and names will be retrieved.\n"
                                                 "The papers will be stored in the given output file in json format. Each key: value pair is\n"
                                                 "SERIAL_NUMBER: {'paperId': ..., 'title': ..., 'abstract': ..., 'url': ..., 'authors': "
                                                 "[{'authorId': ..., 'url': ..., 'name': ...}, {'authorId': ..., 'url': ..., 'name': ...}]}")
    parser.add_argument('-k', '--api-key', default=API_KEY,
                        help='API key to use for the Semantic Scholar API. Defaults to my API key')
    parser.add_argument('-m', '--min-papers', default=20000,
                        help='Minimum number of papers to be scraped from Semantic Scholar (as long as not bigger than the total amount). Defaults to 20,000.\n'
                             'Only papers with an abstract and authors information in the fields of medicine or biology, with at least 10 citations and from 2019 onwards are filtered.')
    parser.add_argument('-o', '--output-path', default=ABSTRACTS_PATH,
                        help='Path to a json output file that will store the scraped title, abstract and authors of all filtered papers. Defaults to abstracts.json')
    args = parser.parse_args()

    api_key = args.api_key

    headers = {
        "x-api-key": api_key
    }

    # Papers in the field of medicine and/or biology from 2019 onwards with at least 10 citations are searched for
    r = requests.get(
        'https://api.semanticscholar.org/graph/v1/paper/search/bulk',
        headers=headers,
        params={'fields': "title,abstract,url,authors.name,authors.url,fieldsOfStudy",
                'fieldsOfStudy': 'Medicine,Biology',
                'year': '2019-',
                'minCitationCount': 10},
    )

    if r.status_code == 200:
        cur_json = r.json()
        data = cur_json["data"]
        total = cur_json["total"]
    else:
        print(f"Error: {r.status_code}")
        print(json.dumps(r.json(), indent=2))
        exit()

    to_json = {}
    for i, paper_dict in enumerate(data):
        if paper_dict['abstract'] and paper_dict['authors'] and len(paper_dict['abstract'].split()) > 10 and paper_dict['fieldsOfStudy'] \
                and ('Biology' in paper_dict['fieldsOfStudy'] or 'Medicine' in paper_dict['fieldsOfStudy']):
            if len(paper_dict['authors']) >= 2:
                paper_dict['authors'] = [paper_dict['authors'][0], paper_dict['authors'][-1]]
            del paper_dict['fieldsOfStudy']
            to_json[str(i)] = paper_dict

    # Iteration counter for given a unique identifier for each paper.
    # No more than 1000 papers can be retrieved with each call to requests.get.
    j = 1

    # As the number of papers that contain authors information and an abstract can change
    # each time, the while loop continues until the minimum number is met.
    while len(to_json) < args.min_papers and len(to_json) < total:
        print(f'{len(to_json)} abstracts added so far.')
        r = requests.get(
            'https://api.semanticscholar.org/graph/v1/paper/search/bulk',
            headers=headers,
            params={'fields': "title,abstract,url,authors.name,authors.url,fieldsOfStudy",
                    'fieldsOfStudy': 'Medicine,Biology',
                    'year': '2019-',
                    'minCitationCount': 10,
                    'token': cur_json['token']},
        )

        if r.status_code == 200:
            cur_json = r.json()
            data = cur_json["data"]
        else:
            print(f"Error: {r.status_code}")
            continue

        for i, paper_dict in enumerate(data):
            if paper_dict['abstract'] and paper_dict['authors'] and len(paper_dict['abstract'].split()) > 10 and paper_dict['fieldsOfStudy'] \
                    and ('Biology' in paper_dict['fieldsOfStudy'] or 'Medicine' in paper_dict['fieldsOfStudy']):
                if len(paper_dict['authors']) >= 2:
                    paper_dict['authors'] = [paper_dict['authors'][0], paper_dict['authors'][-1]]
                del paper_dict['fieldsOfStudy']
                to_json[str(i + j * 1000)] = paper_dict

        j += 1

    with open(args.output_path, "w") as outfile:
        json.dump(to_json, outfile)

    with open(args.output_path) as json_file:
        json_data = json.load(json_file)

    df = pd.DataFrame([
        {"Title": paper["title"],
         "URL": paper["url"],
         "Abstract": paper["abstract"],
         "Authors": ", ".join(
             [f"{author['name']} ({author['url']})" for author in
              paper["authors"]])} for paper in json_data.values()])

    # Uncomment to export the filtered abstracts to an Excel file as well
    # (if uncommented - make sure the file 'abstracts.xlsx' is not open on
    # the computer before running this code)
    # df.to_excel("abstracts.xlsx", engine='xlsxwriter', index=False)


if __name__ == '__main__':
    main()
