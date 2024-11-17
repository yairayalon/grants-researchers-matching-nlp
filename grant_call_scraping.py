from bs4 import BeautifulSoup, element
import requests
import pandas as pd
from tqdm import tqdm
import argparse
from constants import GRANT_CALLS_PATH, GRANT_CALLS_SHEET


def __get_text_from_header(cur_header):
    """
    Tries to extract the textual information of a header based on some curated heuristics.
    :param cur_header: The header
    :return: The text extracted
    """
    nav_text = ""
    for sib in cur_header.next_siblings:
        if type(sib) == element.NavigableString and len(sib.text.split()) > 5:
            nav_text += sib.text

    text = ""
    if len(cur_header.text.split()) >= 15:
        text = cur_header.text + '\n'
    splt_len = 0
    ct = 0
    m_elem = cur_header
    while splt_len < 300 and ct < 4:
        nxt_divs = m_elem.find_all_next("div", limit=2)
        nxt_ps = m_elem.find_all_next("p", limit=5)
        if nxt_divs and nxt_ps:
            if nxt_divs[0].sourceline <= nxt_ps[0].sourceline:
                m_elem = nxt_divs[-1]
                elems = nxt_divs
            else:
                m_elem = nxt_ps[-1]
                elems = nxt_ps
        elif nxt_divs:
            m_elem = nxt_divs[-1]
            elems = nxt_divs
        elif nxt_ps:
            m_elem = nxt_ps[-1]
            elems = nxt_ps
        else:
            if ct == 0:
                return None, nav_text
            else:
                break
        for elem in elems:
            text += elem.text
        splt_len = len(text.split())
        if not splt_len:
            if ct == 0:
                return None, nav_text
            else:
                break
        ct += 1
    return text, nav_text


def concat_text_and_update(dframe, cur_headers, indexes, j):
    """
    Updates the "Purpose" and "Background" of the given dataframe in the given line with text extracted from the appropriate headers for that line.
    Also updates indexes with the line number if headers were indeed found.
    :param dframe: The dataframe to update
    :param cur_headers: The headers extracted for that entry
    :param indexes: Indexes of lines for which headers were successfully extracted
    :param j: The line number in the dataframe
    :return: None
    """
    def check_if_in_and_add(df, txt, cur_head):
        if txt not in df["Purpose"][j] and txt not in df["Background"][j]:
            if cur_head.text.split()[0] == "Background":
                df["Background"][j] += txt
            else:
                df["Purpose"][j] += txt

    for cur_header in cur_headers:
        text, nav_text = __get_text_from_header(cur_header)
        if text:
            check_if_in_and_add(dframe, text, cur_header)
            indexes.add(j)
        if nav_text:
            check_if_in_and_add(dframe, nav_text, cur_header)
            indexes.add(j)


def get_headers(url):
    """
    :param url: The NIH grant call URL to get the headers from
    :return: Headers in the grant that contain the important summarized information about it
    """
    def contains_a_section_i_child_or_is_one(cur_tag):
        if cur_tag.name == 'a' and cur_tag.text.startswith("Section I."):
            return True
        children = cur_tag.children
        for child in children:
            if child.name == 'a' and child.text.startswith("Section I."):
                return True
        return False

    try:
        page = requests.get(url)
    except Exception:
        return
    soup = BeautifulSoup(page.text, "html.parser")
    cur_headers = soup.find_all(lambda tag: "Purpose" == tag.text or "Background" == tag.text or "Purpose and Background" == tag.text
                                        or (tag.text.startswith("Section I.") and not contains_a_section_i_child_or_is_one(tag)) or (tag.name == "div" and
                                        10 >= len(tag.text.split()) > 0 and (tag.text.split()[0] == "Purpose" or tag.text.split()[0] == "Background")))
    header_contents = set()
    headers = []
    for head in cur_headers:
        if tuple(head.text.split()) in header_contents:
            continue
        header_contents.add(tuple(head.text.split()))
        headers.append(head)
    return headers


def get_text_from_url(url):
    """
    Tries to extract the relevant text from a URL of an NIH grant call.
    :param url: The URL
    :return: The extracted text
    """
    cur_headers = get_headers(url)
    if cur_headers is None:
        raise Exception("NIH connection error")
    elif not cur_headers:
        raise Exception("Could not extract text")
    text = ""
    nav_text = ""
    for cur_header in cur_headers:
        cur_text, cur_nav_text = __get_text_from_header(cur_header)
        if cur_text and cur_text not in text:
            text += (cur_text + '\n')
        if cur_nav_text and cur_nav_text not in nav_text:
            nav_text += (cur_nav_text + '\n')
    if not text.split() and not nav_text.split():
        raise Exception("Could not extract text")
    return text + '\n' + nav_text


def main():
    parser = argparse.ArgumentParser(description='Running this file will perform web scraping on the grant URLs in the given input excel file and store the results in the given output file.\n'
                                                 'The script performs basic parsing on the html code, looking for the "Purpose" and "Background" sections in the page.\n'
                                                 'The grant calls will be stored in the given output file in json format. Each key: value pair is\n'
                                                 'SERIAL_NUMBER: {"Title": ..., "Purpose": ..., "Background": ..., "URL": ...}')
    parser.add_argument('-i', '--input-path', default=GRANT_CALLS_SHEET,
                        help='Path to a xlsx input file containing a table of grants with a \"Title\" column and a \"URL\" column. Defaults to 12_11_2023-AllGuideResultsReport.xlsx')
    parser.add_argument('-o', '--output-path', default=GRANT_CALLS_PATH,
                        help='Path to a json output file that will store the scraped Title, Purpose and Background of all filtered grants. Defaults to grant_calls.json')
    args = parser.parse_args()

    df = pd.read_excel(args.input_path)
    df["Purpose"] = ["" for _ in range(len(df))]
    df["Background"] = ["" for _ in range(len(df))]
    idxs = set()
    for i, url in tqdm(enumerate(df["URL"])):
        headers = get_headers(url)
        if headers is None:
            print("Connection error")
            continue
        elif not headers:
            print(f"No fitting headers found for {i}, {url}")
            continue
        else:
            concat_text_and_update(df, headers, idxs, i)

    print(f"Number of grants with filtered/total {len(idxs)}/{len(df)}")
    filtered_df = df[["Title", "Purpose", "Background", "URL"]].loc[list(idxs)]
    filtered_df.to_json(args.output_path, orient="index")

    # Uncomment to export the filtered grant calls to an Excel file as well
    # (if uncommented - make sure the file 'grant_calls.xlsx' is not open on
    # the computer before running this code)
    # filtered_df.to_excel('grant_calls.xlsx', engine='xlsxwriter', index=False)


if __name__ == '__main__':
    main()
