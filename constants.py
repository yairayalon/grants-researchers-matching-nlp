import torch

MAX_INPUT_LEN = 512
SENT_EMB_DIM = 768

GRANT_CALLS_SHEET = "12_11_2023-AllGuideResultsReport.xlsx"

GRANT_CALLS_PATH = "grant_calls.json"
ABSTRACTS_PATH = "abstracts.json"
QDRANT_PATH = "qdrant_collections"

SPECTER_GC_EMB_PATH = "SpecterGrantCallsEmbeddings.json"
SPECTER_AA_EMB_PATH = "SpecterAuthorsAbstractsEmbeddings.json"
ALLMPNET_GC_EMB_PATH = "AllMpnetGrantCallsEmbeddings.json"
ALLMPNET_AA_EMB_PATH = "AllMpnetAuthorsAbstractsEmbeddings.json"

SPECTER_GC_COL = "specter_grant_calls_col"
SPECTER_AA_COL = "specter_authors_abstracts_col"
ALLMPNET_GC_COL = "allmpnet_grant_calls_col"
ALLMPNET_AA_COL = "allmpnet_authors_abstracts_col"

API_KEY = "BX4NoBpKLw4dFiAEy8GDG3sSuAYXi8Vo9VlS5llQ"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
