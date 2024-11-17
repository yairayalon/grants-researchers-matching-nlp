import argparse
import webbrowser

from qdrant_client import QdrantClient
from embedding import tokenize_input
from transformers import AutoTokenizer
from models import SpecterModel, AllMpnetModel
from grant_call_scraping import get_text_from_url
from abstracts_scraping import get_abstract_from_url
from constants import *
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

print(f"Creating local variables")
specter_tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
specter_model = SpecterModel()

allmpnet_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
allmpnet_model = AllMpnetModel()

grant_calls_col = SPECTER_GC_COL
authors_abstracts_col = SPECTER_AA_COL
cur_tokenizer = specter_tokenizer
cur_model = specter_model
matches_num = 1
grant_hits = []
grant_hit_idx = 0
abstract_hits = []
abstract_hit_idx = 0


def find_best_match_in_collection(qdrant, col_name, input_str, model, tokenizer, limit=1):
    """
    Finds the best matches in a given collection for a given input
    :param qdrant: The qdrant object that contains the collection
    :param col_name: The name of the collection used for matching
    :param input_str: The string input to match for
    :param model: The model used for embedding the input
    :param tokenizer: The tokenizer used for the embedding process
    :param limit: The number of matches to be returned from the collection
    :return: A list of the matches found
    """
    hits = qdrant.search(
        collection_name=col_name,
        query_vector=model(
            tokenize_input(input_str, tokenizer=tokenizer)).tolist()[0],
        limit=limit
    )
    return hits


def main():
    parser = argparse.ArgumentParser(description="Running this file will run the grant calls to authors' abstract matching GUI using the collections of the qdrant object from the given path.\n"
                                                 "Finding matching authors (by their abstracts) for a given grant call can be done by inserting its URL from NIH's grants website,"
                                                 "or by writing a textual description of the grant call.\n Finding matching grant calls for a the authors of a paper can be done\n"
                                                 "Matches will be searched for within the loaded qdrant collections.\n"
                                                 "by inserting the paper's URL from the Semantic Scholar website or by writing a textual description of the paper's abstract.\n"
                                                 "An API-KEY is for Semantic Scholar's API is required for URL based matches of authors' abstracts.")
    parser.add_argument('-p', '--qdrant-path', default=QDRANT_PATH,
                        help='Path to the directory from which to load the qdrant collection/s. Defaults to ./qdrant_collections')
    parser.add_argument('-k', '--api-key', default=API_KEY,
                        help='API key to use for querying the Semantic Scholar API when searching for matches for an abstract with a URL. Defaults to my API key')
    args = parser.parse_args()

    print(f"Loading qdrant collection/s from {args.qdrant_path}")
    qdrant = QdrantClient(path=args.qdrant_path)

    print("---finished---")

    # Functions used by the Tkinter GUI

    def insert_clickable_url(widget, url, tag_name):
        widget.tag_config(tag_name, foreground="blue", underline=True)
        widget.insert(tk.END, url, tag_name)
        widget.tag_bind(tag_name, "<Button-1>", lambda _: webbrowser.open_new(url))

    def insert_authors(authors, scrolledtxt):
        scrolledtxt.delete('1.0', tk.END)
        scrolledtxt.insert(tk.END, f'First author\n\nAuthor name:\n'
                                   f'{authors[0]["name"]}\n\nAuthor URL:\n')
        insert_clickable_url(scrolledtxt, authors[0]["url"], "author_url_tag1")
        if len(authors) == 2:
            scrolledtxt.insert(tk.END, f'\n\n\nLast author\n\nAuthor name:\n'
                                       f'{authors[1]["name"]}\n\nAuthor URL:\n')
            insert_clickable_url(scrolledtxt, authors[1]["url"], "author_url_tag2")

    def inc_grant_hit_idx():
        global grant_hits, grant_hit_idx
        if not grant_hits or grant_hit_idx >= len(grant_hits) - 1:
            return
        else:
            grant_hit_idx += 1
        show_grant_match()

    def dec_grant_hit_idx():
        global grant_hits, grant_hit_idx
        if not grant_hits or grant_hit_idx <= 0:
            return
        else:
            grant_hit_idx -= 1
        show_grant_match()

    def inc_abstract_hit_idx():
        global abstract_hits, abstract_hit_idx
        if not abstract_hits or abstract_hit_idx >= len(abstract_hits) - 1:
            return
        else:
            abstract_hit_idx += 1
        show_abstract_match()

    def dec_abstract_hit_idx():
        global abstract_hits, abstract_hit_idx
        if not abstract_hits or abstract_hit_idx <= 0:
            return
        else:
            abstract_hit_idx -= 1
        show_abstract_match()

    def show_grant_match():
        global grant_hits, grant_hit_idx
        hit = grant_hits[grant_hit_idx]
        grant_match_title.delete('1.0', tk.END)
        grant_match_url.delete('1.0', tk.END)
        grant_match_authors.delete('1.0', tk.END)
        grant_match_title.insert(tk.END, hit.payload['Title'])
        insert_clickable_url(grant_match_url, hit.payload['url'], "grant_match_url_tag")
        insert_authors(hit.payload['authors'], grant_match_authors)

    def show_abstract_match():
        global abstract_hits, abstract_hit_idx
        hit = abstract_hits[abstract_hit_idx]
        abstract_match_title.delete('1.0', tk.END)
        abstract_match_url.delete('1.0', tk.END)
        abstract_match_title.insert(tk.END, hit.payload['Title'])
        insert_clickable_url(abstract_match_url, hit.payload['url'], "abstract_match_url_tag")

    def on_grant_url_search():
        global grant_hits, grant_hit_idx, authors_abstracts_col, cur_model, cur_tokenizer, matches_num
        grant_match_title.delete('1.0', tk.END)
        grant_match_url.delete('1.0', tk.END)
        grant_match_authors.delete('1.0', tk.END)
        url = entry_url1.get().strip()
        try:
            text = get_text_from_url(url)
        except Exception as e:
            messagebox.showerror('Error', str(e))
            return
        try:
            grant_hits = find_best_match_in_collection(qdrant, authors_abstracts_col, text, cur_model, cur_tokenizer, limit=matches_num)
        except Exception:
            messagebox.showerror('Error', 'The collection with the chosen embeddings does not exist')
            return
        grant_hit_idx = 0
        show_grant_match()

    def on_grant_text_search():
        global grant_hits, grant_hit_idx, authors_abstracts_col, cur_model, cur_tokenizer, matches_num
        grant_match_title.delete('1.0', tk.END)
        grant_match_url.delete('1.0', tk.END)
        grant_match_authors.delete('1.0', tk.END)
        text = entry_text1.get()
        try:
            grant_hits = find_best_match_in_collection(qdrant, authors_abstracts_col, text, cur_model, cur_tokenizer, limit=matches_num)
        except Exception:
            messagebox.showerror('Error', 'The collection with the chosen embeddings does not exist')
            return
        grant_hit_idx = 0
        show_grant_match()

    def on_abstract_url_search():
        global abstract_hits, abstract_hit_idx, grant_calls_col, cur_model, cur_tokenizer, matches_num
        abstract_match_title.delete('1.0', tk.END)
        abstract_match_url.delete('1.0', tk.END)
        url = entry_url2.get().strip()
        try:
            text = get_abstract_from_url(url, args.api_key)
        except Exception as e:
            messagebox.showerror('Error', str(e))
            return
        try:
            abstract_hits = find_best_match_in_collection(qdrant, grant_calls_col, text, cur_model, cur_tokenizer, limit=matches_num)
        except Exception:
            messagebox.showerror('Error', 'The collection with the chosen embeddings does not exist')
            return
        abstract_hit_idx = 0
        show_abstract_match()

    def on_abstract_text_search():
        global abstract_hits, abstract_hit_idx, grant_calls_col, cur_model, cur_tokenizer, matches_num
        abstract_match_title.delete('1.0', tk.END)
        abstract_match_url.delete('1.0', tk.END)
        text = entry_text2.get()
        try:
            abstract_hits = find_best_match_in_collection(qdrant, grant_calls_col, text, cur_model, cur_tokenizer, limit=matches_num)
        except Exception:
            messagebox.showerror('Error', 'The collection with the chosen embeddings does not exist')
            return
        abstract_hit_idx = 0
        show_abstract_match()

    def on_dropdown_select_emb(*event):
        global grant_calls_col, authors_abstracts_col, cur_tokenizer, cur_model, specter_tokenizer, specter_model
        selected_option = dropdown_emb.get()
        if selected_option == options_emb[0]:
            grant_calls_col = SPECTER_GC_COL
            authors_abstracts_col = SPECTER_AA_COL
            cur_tokenizer = specter_tokenizer
            cur_model = specter_model
        else:
            grant_calls_col = ALLMPNET_GC_COL
            authors_abstracts_col = ALLMPNET_AA_COL
            cur_tokenizer = allmpnet_tokenizer
            cur_model = allmpnet_model

    def on_dropdown_select_matches_num(*event):
        global matches_num
        selected_option = dropdown_mat.get()
        matches_num = int(selected_option)

    # Tkinter GUI creation
    root = tk.Tk()
    root.title("GrantsToAbstractsMatcher")
    root.state("normal")

    frame_padding = (10, 10, 10, 10)
    frame_grid_sticky = tk.W + tk.E + tk.N + tk.S

    # Main frame
    main_frame = ttk.Frame(root, padding=frame_padding)
    main_frame.grid(row=0, column=0, sticky=frame_grid_sticky)

    # Grant frame
    grant_frame = ttk.LabelFrame(main_frame, padding=frame_padding)
    grant_frame.grid(row=0, column=0, padx=10, pady=10, sticky=frame_grid_sticky)

    grant_title = ttk.Label(grant_frame, text="Match To Grant Call", font=('Arial', 10, 'bold', 'underline'))
    grant_title.grid(row=0, column=0, columnspan=3, pady=5)

    ttk.Label(grant_frame, text="From NIH URL:").grid(row=1, column=0, sticky=tk.W, pady=5)
    entry_url1 = ttk.Entry(grant_frame, width=36)
    entry_url1.grid(row=1, column=1, padx=10, pady=5)
    ttk.Button(grant_frame, text="Match", command=on_grant_url_search).grid(row=1, column=2, pady=5)

    ttk.Label(grant_frame, text="From text:").grid(row=2, column=0, sticky=tk.W, pady=5)
    entry_text1 = ttk.Entry(grant_frame, width=36)
    entry_text1.grid(row=2, column=1, padx=10, pady=5)
    ttk.Button(grant_frame, text="Match", command=on_grant_text_search).grid(row=2, column=2, pady=5)

    ###

    ttk.Separator(main_frame, orient="vertical").grid(row=0, column=2, sticky="ns", padx=8)

    ###

    # Abstract frame
    abstract_frame = ttk.LabelFrame(main_frame, padding=frame_padding)
    abstract_frame.grid(row=0, column=3, padx=10, pady=10, sticky=frame_grid_sticky)

    abstract_title = ttk.Label(abstract_frame, text="Match To Abstract", font=('Arial', 10, 'bold', 'underline'))
    abstract_title.grid(row=0, column=0, columnspan=3, pady=5)

    ttk.Label(abstract_frame, text="From SemScho URL:").grid(row=1, column=0, sticky=tk.W, pady=5)
    entry_url2 = ttk.Entry(abstract_frame, width=36)
    entry_url2.grid(row=1, column=1, padx=10, pady=5)
    ttk.Button(abstract_frame, text="Match", command=on_abstract_url_search).grid(row=1, column=2, pady=5)

    ttk.Label(abstract_frame, text="From text:").grid(row=2, column=0, sticky=tk.W, pady=5)
    entry_text2 = ttk.Entry(abstract_frame, width=36)
    entry_text2.grid(row=2, column=1, padx=10, pady=5)
    ttk.Button(abstract_frame, text="Match", command=on_abstract_text_search).grid(row=2, column=2, pady=5)

    # Embeddings dropdown
    ttk.Label(main_frame, text="Embeddings:").grid(row=3, column=0, sticky=tk.E, pady=5)
    options_emb = ["SPECTER", "all-mpnet-base-v2"]
    dropdown_emb = ttk.Combobox(main_frame, values=options_emb, state="readonly")
    dropdown_emb.set(options_emb[0])
    dropdown_emb.grid(row=3, column=1, pady=5, columnspan=2, padx=1, sticky=tk.W)
    dropdown_emb.bind("<<ComboboxSelected>>", on_dropdown_select_emb)

    # Matches # dropdown
    ttk.Label(main_frame, text="Matches #:").grid(row=4, column=0, pady=5, sticky=tk.E)
    options_mat = [str(i + 1) for i in range(10)]
    dropdown_mat = ttk.Combobox(main_frame, values=options_mat, state="readonly")
    dropdown_mat.set(options_mat[0])
    dropdown_mat.grid(row=4, column=1, pady=5, columnspan=2, padx=1, sticky=tk.W)
    dropdown_mat.bind("<<ComboboxSelected>>", on_dropdown_select_matches_num)

    # Matches frame
    matches_frame = ttk.Frame(main_frame, padding=frame_padding)
    matches_frame.grid(row=5, column=0, columnspan=4, pady=10, sticky=frame_grid_sticky)

    # Grant matches frame
    grant_matches_frame = ttk.Frame(matches_frame, padding=frame_padding)
    grant_matches_frame.grid(row=0, column=0, padx=10, pady=10, sticky=frame_grid_sticky)

    ttk.Label(grant_matches_frame, text="Matches", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)

    ttk.Label(grant_matches_frame, text="Title:").grid(row=1, column=0, sticky=tk.E, pady=5)
    grant_match_title = scrolledtext.ScrolledText(grant_matches_frame, wrap="char", width=50, height=4)
    grant_match_title.grid(row=1, column=1, sticky=tk.W, padx=1, pady=5)

    ttk.Label(grant_matches_frame, text="URL:").grid(row=2, column=0, sticky=tk.E, pady=5)
    grant_match_url = scrolledtext.ScrolledText(grant_matches_frame, wrap="char", width=50, height=4)
    grant_match_url.grid(row=2, column=1, sticky=tk.W, padx=1, pady=5)

    ttk.Label(grant_matches_frame, text="Authors:").grid(row=3, column=0, sticky=tk.W, pady=5)
    grant_match_authors = scrolledtext.ScrolledText(grant_matches_frame, wrap="char", width=50, height=8)
    grant_match_authors.grid(row=3, column=1, sticky=tk.W, padx=1, pady=5)

    grant_buttons_frame = ttk.Frame(grant_matches_frame)
    ttk.Button(grant_buttons_frame, text="Next", command=inc_grant_hit_idx).grid(row=0, column=1, padx=5, sticky=tk.W)
    ttk.Button(grant_buttons_frame, text="Previous", command=dec_grant_hit_idx).grid(row=0, column=0, padx=5, sticky=tk.E)
    grant_buttons_frame.grid(row=4, column=0, columnspan=2, pady=2)

    ###

    ttk.Separator(matches_frame, orient="vertical").grid(row=0, column=1, sticky="ns", padx=8)

    ###

    # Abstract matches frame
    abstract_matches_frame = ttk.Frame(matches_frame, padding=frame_padding)
    abstract_matches_frame.grid(row=0, column=2, padx=10, pady=10, sticky=frame_grid_sticky)

    ttk.Label(abstract_matches_frame, text="Matches", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)

    ttk.Label(abstract_matches_frame, text="Title:").grid(row=1, column=0, sticky=tk.W, pady=5)
    abstract_match_title = scrolledtext.ScrolledText(abstract_matches_frame, wrap="char", width=50, height=4)
    abstract_match_title.grid(row=1, column=1, sticky=tk.W, padx=1, pady=5)

    ttk.Label(abstract_matches_frame, text="URL:").grid(row=2, column=0, sticky=tk.W, pady=5)
    abstract_match_url = scrolledtext.ScrolledText(abstract_matches_frame, wrap="char", width=50, height=4)
    abstract_match_url.grid(row=2, column=1, sticky=tk.W, padx=1, pady=5)

    abstract_buttons_frame = ttk.Frame(abstract_matches_frame)
    ttk.Button(abstract_buttons_frame, text="Next", command=inc_abstract_hit_idx).grid(row=0, column=1, padx=5, sticky=tk.W)
    ttk.Button(abstract_buttons_frame, text="Previous", command=dec_abstract_hit_idx).grid(row=0, column=0, padx=5, sticky=tk.E)
    abstract_buttons_frame.grid(row=3, column=0, columnspan=2, pady=2, sticky=tk.N)

    root.mainloop()


if __name__ == '__main__':
    main()
