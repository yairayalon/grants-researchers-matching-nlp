from qdrant_client import models, QdrantClient
from tqdm import tqdm
from constants import *
import argparse
import json


def create_qdrant_collection(qdrant, col_name, json_data, id_key="id", embedding_key="embedding"):
    """
    Creates a new qdrant collection with the given parameters
    :param qdrant: The qdrant object to create the collection for
    :param col_name: The name for the new collection
    :param json_data: Json data that contains the entries to the collection.
                      Should be in the same format as the files created by embedding.py
    :param id_key: Identifier key for the entries in the collection
    :param embedding_key: Embedding key for the entries in the collection
    :return: None
    """
    print(f'---Creating qdrant collection {col_name}---')
    qdrant.recreate_collection(
        collection_name=col_name,
        vectors_config=models.VectorParams(
            size=SENT_EMB_DIM,
            distance=models.Distance.COSINE
        )
    )
    print('---Uploading records---')
    qdrant.upload_records(
        collection_name=col_name,
        records=[
            models.Record(
                id=int(value[id_key]),
                vector=value[embedding_key],
                payload=value
            ) for value in tqdm(json_data.values())
        ]
    )
    print('---finished---')


def main(pipeline=False):
    parser = argparse.ArgumentParser(description="Running this file will create and save a new qdrant object in the given path with collections created using "
                                                 "the specified models for the specified grant calls and/or authors' abstracts files.")
    parser.add_argument('-p', '--qdrant-path', default=QDRANT_PATH,
                        help='Path to the directory in which to save the qdrant collection/s. Defaults to ./qdrant_collections')
    parser.add_argument('-gc', '--grant-calls', action=argparse.BooleanOptionalAction,
                        help='Whether to create a grant calls qdrant collection/s or not.\n'
                             'Collection/s name/s will be "(specter/allmpnet)_grant_calls_col"')
    parser.add_argument('-aa', '--authors-abstracts', action=argparse.BooleanOptionalAction,
                        help='Whether to create an authors abstracts qdrant collection/s or not.\n'
                             'Collection/s name/s will be "(specter/allmpnet)_authors_abstracts_col"')
    parser.add_argument('-sp', '--use-specter', action=argparse.BooleanOptionalAction,
                        help='Whether to create collection/s using SPECTER embeddings or not.')
    parser.add_argument('-am', '--use-allmpnet', action=argparse.BooleanOptionalAction,
                        help='Whether to create collection/s using all-mpnet-base-v2 embeddings or not.')
    parser.add_argument('-il', '--input-paths-list', nargs=4,
                        default=(SPECTER_GC_EMB_PATH, SPECTER_AA_EMB_PATH, ALLMPNET_GC_EMB_PATH, ALLMPNET_AA_EMB_PATH),
                        help='List of paths for input json embeddings files. Irrelevant ones can be empty strings.\n'
                             'Order is:\n[specter_gc, specter_aa, allmpnet_gc, allmpnt_aa]\n'
                             '(gc = grant calls, aa = authors abstracts)\nDefaults to\n'
                             '["SpecterGrantCallsEmbeddings.json", "SpecterAuthorAbstractsEmbeddings.json", "AllMpnetGrantCallsEmbeddings.json", "AllMpnetAuthorAbstractsEmbeddings.json"]')
    args = parser.parse_args()

    # Used by main_pipeline.py. Otherwise irrelevant
    if pipeline:
        args.grant_calls = True
        args.authors_abstracts = True
        args.use_specter = True
        args.use_allmpnet = True

    if args.qdrant_path != ":memory:":
        qdrant = QdrantClient(path=args.qdrant_path)
    else:
        qdrant = QdrantClient(":memory:")
    print(f'qdrant collections will be saved in {args.qdrant_path}')

    if args.use_specter:
        print("Using SPECTER embeddings")
        if args.grant_calls:
            with open(args.input_paths_list[0]) as f:
                specter_grantcall_data = json.load(f)
            create_qdrant_collection(qdrant, SPECTER_GC_COL, specter_grantcall_data)
        if args.authors_abstracts:
            with open(args.input_paths_list[1]) as f:
                specter_authorsabstracts_data = json.load(f)
            create_qdrant_collection(qdrant, SPECTER_AA_COL, specter_authorsabstracts_data)
    if args.use_allmpnet:
        print("Using all-mpnet-base-v2 embeddings")
        if args.grant_calls:
            with open(args.input_paths_list[2]) as f:
                allmpnet_grantcall_data = json.load(f)
            create_qdrant_collection(qdrant, ALLMPNET_GC_COL, allmpnet_grantcall_data)
        if args.authors_abstracts:
            with open(args.input_paths_list[3]) as f:
                allmpnet_authorsabstracts_data = json.load(f)
            create_qdrant_collection(qdrant, ALLMPNET_AA_COL, allmpnet_authorsabstracts_data)


if __name__ == '__main__':
    main()
