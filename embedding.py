import sys

from transformers import AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm

from models import SpecterModel, AllMpnetModel
from constants import *

# 'allenai/specter'
# 'sentence-transformers/all-mpnet-base-v2'


class DS:
    """
    Abstract Dataset class. Batches are created from reading the given json file,
    and are tokenized using the given tokenizer
    """
    def __init__(self, data_path, tokenizer, batch_size, max_length=MAX_INPUT_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        with open(data_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)


class GrantCallsDS(DS):
    """
    Dataset class for grant calls
    """
    def batches(self):
        batch = []
        batch_ids = []
        titles = []
        urls = []
        batch_size = self.batch_size
        i = 0
        for k, d in self.data.items():
            if i % batch_size != 0 or i == 0:
                titles.append(d['Title'])
                urls.append(d['URL'])
                batch_ids.append(k)
                batch.append((d.get('Purpose') or '') + ' ' + (d.get('Background') or ''))
            else:
                input_ids = self.tokenizer(batch, padding=True,
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.max_length)
                yield input_ids.to(device), batch_ids, titles, urls
                titles = [d['Title']]
                urls = [d['URL']]
                batch_ids = [k]
                batch = [(d.get('Purpose') or '') + ' ' + (d.get('Background') or '')]
            i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt",
                                       max_length=self.max_length)
            yield input_ids.to(device), batch_ids, titles, urls


class AuthorAbstractsDS(DS):
    """
    Dataset class for authors' abstracts
    """
    def batches(self):
        batch = []
        batch_ids = []
        titles = []
        urls = []
        authors = []
        batch_size = self.batch_size
        i = 0
        for k, d in self.data.items():
            if i % batch_size != 0 or i == 0:
                titles.append(d['title'])
                urls.append(d['url'])
                authors.append(d['authors'])
                batch_ids.append(k)
                batch.append(d['abstract'])
            else:
                input_ids = self.tokenizer(batch, padding=True,
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.max_length)
                yield input_ids.to(device), batch_ids, titles, urls, authors
                titles = [d['title']]
                urls = [d['url']]
                authors = [d['authors']]
                batch_ids = [k]
                batch = [d['abstract']]
            i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt",
                                       max_length=self.max_length)
            yield input_ids.to(device), batch_ids, titles, urls, authors


def embed_grant_calls_and_store(model, dataset, batch_size, output_name):
    """
    Embeds the batches in the given GrantCallsDS and stores them in the given output path
    :param model: The model to use for the embedding
    :param dataset: The dataset to extract the data from
    :param batch_size: The batch size
    :param output_name: Path of the created output file
    :return: None
    """
    assert type(dataset) == GrantCallsDS
    with torch.no_grad():
        results = {}

        for batch, batch_ids, titles, urls in tqdm(dataset.batches(), total=len(dataset) // batch_size):
            emb = model(batch)
            for grant_call_id, embedding, title, url in zip(batch_ids, emb.unbind(), titles, urls):
                results[grant_call_id] = {"Title": title,
                                          "id": grant_call_id,
                                          "url": url,
                                          "embedding": embedding.detach().cpu().numpy().tolist()}

        with open(output_name, 'w') as fout:
            fout.write(json.dumps(results))


def embed_author_abstracts_and_store(model, dataset, batch_size, output_name):
    """
    Embeds the batches in the given AuthorAbstractsDS and stores them in the given output path
    :param model: The model to use for the embedding
    :param dataset: The dataset to extract the data from
    :param batch_size: The batch size
    :param output_name: Path of the created output file
    :return: None
    """
    assert type(dataset) == AuthorAbstractsDS
    with torch.no_grad():
        results = {}

        for batch, batch_ids, titles, urls, authors_lists in tqdm(dataset.batches(), total=len(dataset) // batch_size):
            emb = model(batch)
            for paper_id, embedding, title, url, authors in zip(batch_ids, emb.unbind(), titles, urls, authors_lists):
                results[paper_id] = {"Title": title,
                                     "id": paper_id,
                                     "url": url,
                                     "authors": authors,
                                     "embedding": embedding.detach().cpu().numpy().tolist()}

        with open(output_name, 'w') as fout:
            fout.write(json.dumps(results))


def tokenize_input(inputs, tokenizer):
    """
    :param inputs: A list of strings for tokenization
    :param tokenizer: The tokenizer to use
    :return: The tokenized inputs
    """
    input_ids = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=MAX_INPUT_LEN)
    return input_ids


def main(pipeline=False):
    parser = argparse.ArgumentParser(description="Running this file will embed the grant calls and/or authors' abstracts using SPECTER and/or all-mpnet-base-v2.\n"
                                                 "The results will be saved in the relevant and appropriate given output files in json format.\n"
                                                 "Both grant calls and authors' abstracts will have a new 'embedding' filed and an 'id' filed."
                                                 "grant calls will not have their 'Purpose' and 'Background' fileds, and abstracts will not have their 'abstract' field")
    parser.add_argument('-gci', '--grant-calls-input', default="",
                        help='Path to a json file containing grant calls metadata created by grant_call_scraping.py')
    parser.add_argument('-aai', '--authors-abstracts-input', default="",
                        help='Path to a json file containing authors abstracts metadata created by abstracts_scraping.py')
    parser.add_argument('-sp', '--use-specter', action=argparse.BooleanOptionalAction,
                        help='Whether to use SPECTER for embedding or not. Output json file will be created.')
    parser.add_argument('-am', '--use-allmpnet', action=argparse.BooleanOptionalAction,
                        help='Whether to use all-mpnet-base-v2 for embedding or not. Output json file will be created.')
    parser.add_argument('-ol', '--output-paths-list', nargs=4,
                        default=(SPECTER_GC_EMB_PATH, SPECTER_AA_EMB_PATH, ALLMPNET_GC_EMB_PATH, ALLMPNET_AA_EMB_PATH),
                        help='List of paths for output json embeddings files. Irrelevant ones can be empty strings.\n'
                             'Order is:\n[specter_gc, specter_aa, allmpnet_gc, allmpnt_aa]\n'
                             '(gc = grant calls, aa = authors abstracts)\nDefaults to\n'
                             '["SpecterGrantCallsEmbeddings.json", "SpecterAuthorsAbstractsEmbeddings.json", "AllMpnetGrantCallsEmbeddings.json", "AllMpnetAuthorsAbstractsEmbeddings.json"]')
    parser.add_argument('-b', '--batch-size', type=int, default=8,
                        help='Batch size for embedding process for both models. Defaults to 8.')
    args = parser.parse_args()

    # Used by main_pipeline.py. Otherwise irrelevant
    if pipeline:
        args.grant_calls_input = GRANT_CALLS_PATH
        args.authors_abstracts_input = ABSTRACTS_PATH
        args.use_specter = True
        args.use_allmpnet = True

    if not (args.grant_calls_input or args.authors_abstracts_input):
        sys.exit("No input json files were given. Use --grant-calls-path and/or --author-abstracts-path flags.")
    if not (args.use_specter or args.use_allmpnet):
        sys.exit("Chosen model/s to use were not provided. Use --use-specter or --use-allmpnet flags.")

    if args.use_specter:
        print("Using SPECTER model")
        specter_tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        specter_model = SpecterModel()
        if args.grant_calls_input:
            print(f"Embedding grant calls from {args.grant_calls_input} and storing in {SPECTER_GC_EMB_PATH}")
            specter_gc_dataset = GrantCallsDS(data_path=args.grant_calls_input, batch_size=args.batch_size, tokenizer=specter_tokenizer)
            embed_grant_calls_and_store(specter_model, specter_gc_dataset, args.batch_size, args.output_paths_list[0])
            print("---finished embedding---")
        if args.authors_abstracts_input:
            print(f"Embedding authors abstracts from {args.authors_abstracts_input} and storing in {SPECTER_AA_EMB_PATH}")
            specter_aa_dataset = AuthorAbstractsDS(data_path=args.authors_abstracts_input, batch_size=args.batch_size, tokenizer=specter_tokenizer)
            embed_author_abstracts_and_store(specter_model, specter_aa_dataset, args.batch_size, args.output_paths_list[1])
            print("---finished embedding---")
    if args.use_allmpnet:
        print("Using all-mpnet-base-v2 model")
        allmpnet_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        allmpnet_model = AllMpnetModel()
        if args.grant_calls_input:
            print(f"Embedding grant calls from {args.grant_calls_input} and storing in {ALLMPNET_GC_EMB_PATH}")
            allmpnet_gc_dataset = GrantCallsDS(data_path=args.grant_calls_input, batch_size=args.batch_size, tokenizer=allmpnet_tokenizer)
            embed_grant_calls_and_store(allmpnet_model, allmpnet_gc_dataset, args.batch_size, args.output_paths_list[2])
            print("---finished embedding---")
        if args.authors_abstracts_input:
            print(f"Embedding authors abstracts from {args.authors_abstracts_input} and storing in {ALLMPNET_AA_EMB_PATH}")
            allmpnet_aa_dataset = AuthorAbstractsDS(data_path=args.authors_abstracts_input, batch_size=args.batch_size, tokenizer=allmpnet_tokenizer)
            embed_author_abstracts_and_store(allmpnet_model, allmpnet_aa_dataset, args.batch_size, args.output_paths_list[3])
            print("---finished embedding---")


if __name__ == '__main__':
    main()
