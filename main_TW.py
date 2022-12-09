# %% [markdown]
# ### TypeWriter
# This notebook runs the TypeWriter method for predicting types of Python methods.

# %%
import json
from typewriter import config_TW
from os.path import join, exists, isdir
import os
import re
import pickle
import pandas as pd
import numpy as np
import result_proc
from importlib import reload

# %%
import nltk

nltk.download("omw-1.4")

# %% [markdown]
# ### Cleaning

# %%
from result_proc import copy_results, clean_output

# Copying the results and cleaning the output of last run
# copy_results('./output/reports/', './results/')

# Delete all the files in the output folder
# clean_output('./output/')

# %% [markdown]
# ### Select Python projects

# %%
from gh_query import load_json, gen_json_file, find_current_repos

# %% [markdown]
# Here, we only select Python projects that has `mypy` as one of its dependencies

# %%
# repos = load_json('./data/mypy-dependents-by-stars.json')

repo_dir = "./data/paper-dataset/Repos"
repos_on_disk = find_current_repos(repo_dir, author_repo=True)
print(f"{len(repos_on_disk)} repositories in {repo_dir}")
# gen_json_file('./data/py_projects_all.json', repos, repos_on_disk)

repo_json = list()
for repo_on_disk in repos_on_disk:
    author, repo = repo_on_disk.split("/")
    repo_json.append({"author": author, "repo": repo})
with open("./data/py_projects_all.json", "w") as f:
    json.dump(repo_json, f)


# %% [markdown]
# Loads selected repos

# %%
repos = load_json("./data/py_projects_all.json")
print("number of projects:", len(repos))

# %% [markdown]
# ### Create output folder and dirs

# %% [markdown]
# Give a name to output directory. It'll be created automatically.

# %%
OUTPUT_DIR = "./dataset/"

# %%
OUTPUT_EMBEDDINGS_DIRECTORY = join(OUTPUT_DIR, "embed")
OUTPUT_DIRECTORY_TW = join(OUTPUT_DIR, "funcs")
AVAILABLE_TYPES_DIR = join(OUTPUT_DIR, "avl_types")
RESULTS_DIR = join(OUTPUT_DIR, "results")

ML_INPUTS_PATH_TW = join(OUTPUT_DIR, "ml_inputs")
ML_PARAM_TW_TRAIN = join(ML_INPUTS_PATH_TW, "_ml_param_train.csv")
ML_PARAM_TW_TEST = join(ML_INPUTS_PATH_TW, "_ml_param_test.csv")
ML_RET_TW_TRAIN = join(ML_INPUTS_PATH_TW, "_ml_ret_train.csv")
ML_RET_TW_TEST = join(ML_INPUTS_PATH_TW, "_ml_ret_test.csv")

VECTOR_OUTPUT_DIR_TW = join(OUTPUT_DIR, "vectors")
VECTOR_OUTPUT_TRAIN = join(VECTOR_OUTPUT_DIR_TW, "train")
VECTOR_OUTPUT_TEST = join(VECTOR_OUTPUT_DIR_TW, "test")

W2V_MODEL_TOKEN_DIR = join(OUTPUT_EMBEDDINGS_DIRECTORY, "w2v_token_model.bin")
W2V_MODEL_COMMENTS_DIR = join(OUTPUT_EMBEDDINGS_DIRECTORY, "w2v_comments_model.bin")

DATA_FILE_TW = join(ML_INPUTS_PATH_TW, "_all_data.csv")

LABEL_ENCODER_PATH_TW = join(ML_INPUTS_PATH_TW, "label_encoder.pkl")
TYPES_FILE_TW = join(ML_INPUTS_PATH_TW, "_most_frequent_types.csv")

dirs = [
    OUTPUT_EMBEDDINGS_DIRECTORY,
    OUTPUT_DIRECTORY_TW,
    AVAILABLE_TYPES_DIR,
    RESULTS_DIR,
    ML_INPUTS_PATH_TW,
    VECTOR_OUTPUT_DIR_TW,
    VECTOR_OUTPUT_TRAIN,
    VECTOR_OUTPUT_TEST,
]

# %%
if not isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

config_TW.create_dirs(dirs)

# %% [markdown]
# ### Step 1: Extracting functions
# Extract natural language information and preprocess functions

# %%
from dltpy.preprocessing.pipeline import Pipeline

p = Pipeline("./data/paper-dataset/Repos/", OUTPUT_DIRECTORY_TW, AVAILABLE_TYPES_DIR)
p.run_pipeline_manual(repos, 28)

# %% [markdown]
# Generates dataframe or loads an existing one

# %%
from dltpy.input_preparation.generate_df import list_files, parse_df

# %%
if config_TW.CACHE_TW and os.path.exists(DATA_FILE_TW):
    print("Loading cached copy")
    df = pd.read_csv(DATA_FILE_TW)
else:
    DATA_FILES = list_files(OUTPUT_DIRECTORY_TW)
    print("Found %d datafiles" % len(DATA_FILES))
    # print(DATA_FILES)
    df = parse_df(DATA_FILES, batch_size=4098)
    print("Dataframe loaded writing it to CSV")
    df.to_csv(DATA_FILE_TW, index=False)

# %% [markdown]
# Initial dataset before processing parameter and returns data

# %%
df.head(25)

# %%
df["arg_names"].head(20)

# %% [markdown]
# stats of the repositories

# %%
print("Number of source files: ", len(df.file.unique()))
print("Number of functions: ", len(df.name))

# %%
print(
    "Number of functions with comments: ",
    df[(~df["return_descr"].isnull()) | (~df["func_descr"].isnull())].shape[0],
)
print("Number of functions with return types: ", df["return_type"].count())
print(
    "Number of functions with both: ",
    df[
        ((~df["return_descr"].isnull()) | (~df["func_descr"].isnull()))
        & (~df["return_type"].isnull())
    ].shape[0],
)

# %% [markdown]
# Splits the intial dataset based on source files. Later on, we use this to split the datasets.

# %%
from sklearn.model_selection import train_test_split

train_files, test_files = train_test_split(
    pd.DataFrame(df["file"].unique(), columns=["file"]), test_size=0.2
)

# %%
df_train = df[df["file"].isin(train_files.to_numpy().flatten())]
print("Number of functions in train set: ", df_train.shape[0])

# %%
df_test = df[df["file"].isin(test_files.to_numpy().flatten())]
print("Number of functions in test set: ", df_test.shape[0])

# %% [markdown]
# ### Step 2: Processing functions

# %%
from typewriter import prepocessing

reload(prepocessing)
from typewriter.prepocessing import (
    filter_functions,
    gen_argument_df_TW,
    gen_most_frequent_avl_types,
    encode_aval_types_TW,
)
from dltpy.input_preparation.generate_df import (
    filter_return_dp,
    format_df,
    encode_types,
)

# %% [markdown]
# Filters trivial functions such as `__str__` and `__len__`

# %%
df = filter_functions(df)

# %% [markdown]
# Extracts informations for functions' arguments

# %%
df_params = gen_argument_df_TW(df)

# %% [markdown]
# Ignore `self` arguments and args with type of `Any` or `None`.

# %%
args_count = df_params["arg_name"].count()
args_with_annot = df_params[df_params["arg_type"] != ""].shape[0]
print("Number of arguments: ", args_count)
print("Args with type annotations: ", args_with_annot)
df_params = df_params[
    (df_params["arg_name"] != "self")
    & ((df_params["arg_type"] != "Any") & (df_params["arg_type"] != "None"))
]
print("Ignored trivial types: ", (args_count - df_params.shape[0]))

# %% [markdown]
# Ignore arguments without a type

# %%
df_params = df_params[df_params["arg_type"] != ""]
print("Number of arguments with types: ", df_params.shape[0])

# %% [markdown]
# Filters out functions:
# - without a return type
# - with the return type of `Any` or `None`
# - without a return expression

# %%
df = filter_return_dp(df)

# %%
df.head(10)

# %%
df["arg_names"]

# %%
df = format_df(df)

# %%
df.head(10)

# %% [markdown]
# Encode types as int

# %%
df, df_params, label_encoder, uniq_types = encode_types(df, df_params, TYPES_FILE_TW)

# %% [markdown]
# Add argument names as a string except self

# %%
df["arg_names_str"] = df["arg_names"].apply(
    lambda l: " ".join([v for v in l if v != "self"])
)

# %% [markdown]
# Add return expressions as a string, replace self. and self within expressions

# %%
df["return_expr_str"] = df["return_expr"].apply(
    lambda l: " ".join([re.sub(r"self\.?", "", v) for v in l])
)

# %% [markdown]
# Drop all columns useless for the ML algorithms

# %%
df = df.drop(
    columns=[
        "author",
        "repo",
        "has_type",
        "arg_names",
        "arg_types",
        "arg_descrs",
        "return_expr",
    ]
)

# %% [markdown]
# Extracts top 1000 available types

# %%
if config_TW.CACHE_TW and os.path.exists(
    os.path.join(
        AVAILABLE_TYPES_DIR, "top_%d_types.csv" % (config_TW.AVAILABLE_TYPES_NUMBER - 1)
    )
):
    df_types = pd.read_csv(
        os.path.join(
            AVAILABLE_TYPES_DIR,
            "top_%d_types.csv" % (config_TW.AVAILABLE_TYPES_NUMBER - 1),
        )
    )
else:
    df_types = gen_most_frequent_avl_types(
        AVAILABLE_TYPES_DIR, config_TW.AVAILABLE_TYPES_NUMBER - 1, True
    )

# %%
df_params, df = encode_aval_types_TW(df_params, df, df_types)

# %%
df["ret_aval_enc"].head(10)

# %% [markdown]
# Calculates the datapoints and types coverage

# %%
all_enc_types = np.concatenate(
    (df_params["arg_type_enc"].values, df["return_type_enc"].values)
)
other_type_count = np.count_nonzero(
    all_enc_types == label_encoder.transform(["other"])[0]
)
print("Number of datapoints with other types: ", other_type_count)
print(
    "The percentage of covered unique types: %.2f%%"
    % ((config_TW.AVAILABLE_TYPES_NUMBER / len(uniq_types)) * 100)
)
print(
    "The percentage of all datapoints covered by considered types: %.2f%%"
    % ((1 - other_type_count / all_enc_types.shape[0]) * 100)
)

# %% [markdown]
# Final arguments data before embedding step

# %%
df_params.head(10)

# %% [markdown]
# Final return data before embedding step

# %%
df.head(10)

# %% [markdown]
# Splits parameters and returns type dataset by file into a train and test sets

# %%
df_params_train = df_params[df_params["file"].isin(train_files.to_numpy().flatten())]
df_params_test = df_params[df_params["file"].isin(test_files.to_numpy().flatten())]
print(df_params_train.shape, df_params_test.shape)

# %%
df_ret_train = df[df["file"].isin(train_files.to_numpy().flatten())]
df_ret_test = df[df["file"].isin(test_files.to_numpy().flatten())]
print(df_ret_train.shape, df_ret_test.shape)

# %% [markdown]
# Make sure that there is no overlap between the train and test sets.

# %%
list(
    set(df_params_train["file"].tolist()).intersection(
        set(df_params_test["file"].tolist())
    )
)

# %%
list(set(df_ret_train["file"].tolist()).intersection(set(df_ret_test["file"].tolist())))

# %% [markdown]
# Store the dataframes and the label encoder

# %%
if not os.path.exists(ML_INPUTS_PATH_TW):
    os.makedirs(ML_INPUTS_PATH_TW)

with open(LABEL_ENCODER_PATH_TW, "wb") as file:
    pickle.dump(label_encoder, file)

# df.to_csv(config.ML_RETURN_DF_PATH_TW, index=False)
# df_params.to_csv(config.ML_PARAM_DF_PATH_TW, index=False)
df_params_train.to_csv(ML_PARAM_TW_TRAIN, index=False)
df_params_test.to_csv(ML_PARAM_TW_TEST, index=False)
df_ret_train.to_csv(ML_RET_TW_TRAIN, index=False)
df_ret_test.to_csv(ML_RET_TW_TEST, index=False)

# %% [markdown]
# Plots 20 most frequent types in the dataset

# %%
result_proc.plot_top_n_types(TYPES_FILE_TW, 20)

# %% [markdown]
# ### Step 3: Embeddings

# %%
from typewriter import extraction
from typewriter.extraction import EmbeddingTypeWriter
from gensim.models import Word2Vec

reload(extraction)

# %% [markdown]
# Loads dataset for parametes and return types

# %%
param_df = pd.read_csv(ML_PARAM_TW_TRAIN)
return_df = pd.read_csv(ML_RET_TW_TRAIN)

print("Number of parameters types:", param_df.shape[0])
print("Number of returns types", return_df.shape[0])

# %% [markdown]
# Train embeddings

# %%
embedder = EmbeddingTypeWriter(
    param_df, return_df, W2V_MODEL_COMMENTS_DIR, W2V_MODEL_TOKEN_DIR
)
embedder.train_token_model()
embedder.train_comment_model()

# %% [markdown]
# Loads pre-trained W2V models for TypeWriter

# %%
w2v_token_model = Word2Vec.load(W2V_MODEL_TOKEN_DIR)
w2v_comments_model = Word2Vec.load(W2V_MODEL_COMMENTS_DIR)

# %% [markdown]
# stats of the W2V models:

# %%
print("W2V statistics: ")
print(
    "W2V token model total amount of words : " + str(w2v_token_model.corpus_total_words)
)
print(
    "W2V comments model total amount of words : "
    + str(w2v_comments_model.corpus_total_words)
)
print("\n Top 20 words for token model:")
print(w2v_token_model.wv.index_to_key[:20])
print("\n Top 20 words for comments model:")
print(w2v_comments_model.wv.index_to_key[:20])

# %% [markdown]
# ### Step 4: Vector Representation

# %%
from typewriter.extraction import (
    IdentifierSequence,
    TokenSequence,
    CommentSequence,
    process_datapoints_TW,
    gen_aval_types_datapoints,
)

# %% [markdown]
# Process parameter datapoints

# %%
id_trans_func_param = lambda row: IdentifierSequence(
    w2v_token_model, row.arg_name, row.other_args, row.func_name
)
token_trans_func_param = lambda row: TokenSequence(
    w2v_token_model, 7, 3, row.arg_occur, None
)

cm_trans_func_param = lambda row: CommentSequence(
    w2v_comments_model, row.func_descr, row.arg_comment, None
)

# %% [markdown]
# Identifiers

# %%
dp_ids_param_X_train = process_datapoints_TW(
    ML_PARAM_TW_TRAIN,
    VECTOR_OUTPUT_TRAIN,
    "identifiers_",
    "param_train",
    id_trans_func_param,
)

# %%
dp_ids_param_X_test = process_datapoints_TW(
    ML_PARAM_TW_TEST,
    VECTOR_OUTPUT_TEST,
    "identifiers_",
    "param_test",
    id_trans_func_param,
)

# %% [markdown]
# Tokens

# %%
dp_tokens_param_X_train = process_datapoints_TW(
    ML_PARAM_TW_TRAIN,
    VECTOR_OUTPUT_TRAIN,
    "tokens_",
    "param_train",
    token_trans_func_param,
)

# %%
dp_tokens_param_X_test = process_datapoints_TW(
    ML_PARAM_TW_TEST,
    VECTOR_OUTPUT_TEST,
    "tokens_",
    "param_test",
    token_trans_func_param,
)

# %% [markdown]
# Comments

# %%
dp_cms_param_X_train = process_datapoints_TW(
    ML_PARAM_TW_TRAIN,
    VECTOR_OUTPUT_TRAIN,
    "comments_",
    "param_train",
    cm_trans_func_param,
)

# %%
dp_cms_param_X_test = process_datapoints_TW(
    ML_PARAM_TW_TEST, VECTOR_OUTPUT_TEST, "comments_", "param_test", cm_trans_func_param
)

# %%
print("Identifiers' train set parameters: ", dp_ids_param_X_train.shape)
print("Tokens' train set parameters: ", dp_tokens_param_X_train.shape)
print("Comments' train parameters: ", dp_cms_param_X_train.shape)
print("Identifiers' test set parameters: ", dp_ids_param_X_test.shape)
print("Tokens' test set parameters: ", dp_tokens_param_X_test.shape)
print("Comments' test set parameters: ", dp_cms_param_X_test.shape)

# %% [markdown]
# Process returns datapoints

# %%
id_trans_func_ret = lambda row: IdentifierSequence(
    w2v_token_model, None, row.arg_names_str, row.name
)

token_trans_func_ret = lambda row: TokenSequence(
    w2v_token_model, 7, 3, None, row.return_expr_str
)

cm_trans_func_ret = lambda row: CommentSequence(
    w2v_comments_model, row.func_descr, None, row.return_descr
)

# %% [markdown]
# Identifiers

# %%
dp_ids_ret_X_train = process_datapoints_TW(
    ML_RET_TW_TRAIN, VECTOR_OUTPUT_TRAIN, "identifiers_", "ret_train", id_trans_func_ret
)

# %%
dp_ids_ret_X_test = process_datapoints_TW(
    ML_RET_TW_TEST, VECTOR_OUTPUT_TEST, "identifiers_", "ret_test", id_trans_func_ret
)

# %% [markdown]
# Tokens

# %%
dp_tokens_ret_X_train = process_datapoints_TW(
    ML_RET_TW_TRAIN, VECTOR_OUTPUT_TRAIN, "tokens_", "ret_train", token_trans_func_ret
)

# %%
dp_tokens_ret_X_test = process_datapoints_TW(
    ML_RET_TW_TEST, VECTOR_OUTPUT_TEST, "tokens_", "ret_test", token_trans_func_ret
)

# %% [markdown]
# Comments

# %%
dp_cms_ret_X_train = process_datapoints_TW(
    ML_RET_TW_TRAIN, VECTOR_OUTPUT_TRAIN, "comments_", "ret_train", cm_trans_func_ret
)

# %%
dp_cms_ret_X_test = process_datapoints_TW(
    ML_RET_TW_TEST, VECTOR_OUTPUT_TEST, "comments_", "ret_test", cm_trans_func_ret
)

# %%
print("Identifiers' train set return: ", dp_ids_ret_X_train.shape)
print("Identifiers' test set return: ", dp_ids_ret_X_test.shape)
print("Tokens' train set return: ", dp_tokens_ret_X_train.shape)
print("Tokens' test set return: ", dp_tokens_ret_X_test.shape)
print("Comments' train set return:", dp_cms_ret_X_train.shape)
print("Comments' test set return:", dp_cms_ret_X_test.shape)

# %% [markdown]
# Generates datapoints for available types

# %%
dp_params_train_aval_types, dp_ret_train_aval_types = gen_aval_types_datapoints(
    ML_PARAM_TW_TRAIN, ML_RET_TW_TRAIN, "train", VECTOR_OUTPUT_TRAIN
)

# %%
dp_params_test_aval_types, dp_ret_test_aval_types = gen_aval_types_datapoints(
    ML_PARAM_TW_TEST, ML_RET_TW_TEST, "test", VECTOR_OUTPUT_TEST
)

# %%
print("Available types-parameters-train:", dp_params_train_aval_types.shape)
print("Available types-returns-train:", dp_ret_train_aval_types.shape)
print("Available types-parameters-test:", dp_params_test_aval_types.shape)
print("Available types-returns-test:", dp_ret_test_aval_types.shape)

# %% [markdown]
# Generates type vectors.

# %%
from dltpy.input_preparation.df_to_vec import generate_labels

# %%
params_y_train, ret_y_train = generate_labels(
    ML_PARAM_TW_TRAIN, ML_RET_TW_TRAIN, "train", VECTOR_OUTPUT_TRAIN
)

# %%
params_y_test, ret_y_test = generate_labels(
    ML_PARAM_TW_TEST, ML_RET_TW_TEST, "test", VECTOR_OUTPUT_TEST
)

# %% [markdown]
# ### Step 5: Learning the neural model

# %%
from typewriter.model import (
    load_data_tensors_TW,
    TWModel,
    train_loop_TW,
    evaluate_TW,
    report_TW,
    load_label_tensors_TW,
    TWModelA,
    EnhancedTWModel,
    BaseLineModel,
)
from statistics import mean
from torch.utils.data import DataLoader, TensorDataset
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-- Using {device} for training.")

# %% [markdown]
# Loads parameters' data vectors

# %%
def load_param_train_data():
    return (
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TRAIN, "identifiers_param_train_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TRAIN, "tokens_param_train_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TRAIN, "comments_param_train_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TRAIN, "params_train_aval_types_dp.npy")
        ),
        load_label_tensors_TW(
            join(VECTOR_OUTPUT_TRAIN, "params_train_datapoints_y.npy")
        ),
    )


def load_param_test_data():
    return (
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TEST, "identifiers_param_test_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TEST, "tokens_param_test_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TEST, "comments_param_test_datapoints_x.npy")
        ),
        load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, "params_test_aval_types_dp.npy")),
        load_label_tensors_TW(join(VECTOR_OUTPUT_TEST, "params_test_datapoints_y.npy")),
    )


# %% [markdown]
# Loads return' data vectors

# %%
def load_ret_train_data():
    return (
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TRAIN, "identifiers_ret_train_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TRAIN, "tokens_ret_train_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TRAIN, "comments_ret_train_datapoints_x.npy")
        ),
        load_data_tensors_TW(join(VECTOR_OUTPUT_TRAIN, "ret_train_aval_types_dp.npy")),
        load_label_tensors_TW(join(VECTOR_OUTPUT_TRAIN, "ret_train_datapoints_y.npy")),
    )


def load_ret_test_data():
    return (
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TEST, "identifiers_ret_test_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TEST, "tokens_ret_test_datapoints_x.npy")
        ),
        load_data_tensors_TW(
            join(VECTOR_OUTPUT_TEST, "comments_ret_test_datapoints_x.npy")
        ),
        load_data_tensors_TW(join(VECTOR_OUTPUT_TEST, "ret_test_aval_types_dp.npy")),
        load_label_tensors_TW(join(VECTOR_OUTPUT_TEST, "ret_test_datapoints_y.npy")),
    )


# %% [markdown]
# Concatanates parameters and return data vectors for combined prediction

# %%
def load_combined_train_data():
    return (
        torch.cat(
            (
                load_data_tensors_TW(
                    join(
                        VECTOR_OUTPUT_TRAIN, "identifiers_param_train_datapoints_x.npy"
                    )
                ),
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "identifiers_ret_train_datapoints_x.npy")
                ),
            )
        ),
        torch.cat(
            (
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "tokens_param_train_datapoints_x.npy")
                ),
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "tokens_ret_train_datapoints_x.npy")
                ),
            )
        ),
        torch.cat(
            (
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "comments_param_train_datapoints_x.npy")
                ),
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "comments_ret_train_datapoints_x.npy")
                ),
            )
        ),
        torch.cat(
            (
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "params_train_aval_types_dp.npy")
                ),
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "ret_train_aval_types_dp.npy")
                ),
            )
        ),
        torch.cat(
            (
                load_label_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "params_train_datapoints_y.npy")
                ),
                load_label_tensors_TW(
                    join(VECTOR_OUTPUT_TRAIN, "ret_train_datapoints_y.npy")
                ),
            )
        ),
    )


def load_combined_test_data():
    return (
        torch.cat(
            (
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "identifiers_param_test_datapoints_x.npy")
                ),
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "identifiers_ret_test_datapoints_x.npy")
                ),
            )
        ),
        torch.cat(
            (
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "tokens_param_test_datapoints_x.npy")
                ),
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "tokens_ret_test_datapoints_x.npy")
                ),
            )
        ),
        torch.cat(
            (
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "comments_param_test_datapoints_x.npy")
                ),
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "comments_ret_test_datapoints_x.npy")
                ),
            )
        ),
        torch.cat(
            (
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "params_test_aval_types_dp.npy")
                ),
                load_data_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "ret_test_aval_types_dp.npy")
                ),
            )
        ),
        torch.cat(
            (
                load_label_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "params_test_datapoints_y.npy")
                ),
                load_label_tensors_TW(
                    join(VECTOR_OUTPUT_TEST, "ret_test_datapoints_y.npy")
                ),
            )
        ),
    )


# %% [markdown]
# Datasets

# %%
datasets_train = {
    "combined": load_combined_train_data,
    "return": load_ret_train_data,
    "argument": load_param_train_data,
}
datasets_test = {
    "combined": load_combined_test_data,
    "return": load_ret_test_data,
    "argument": load_param_test_data,
}

# %% [markdown]
# Learning parameters

# %%
input_size = config_TW.W2V_VEC_LENGTH
hidden_size = 768
output_size = 1000
num_layers = 1
learning_rate = 0.002
dropout_rate = 0.25
epochs = 25
top_n_pred = [1, 3, 5]
n_rep = 1
batch_size = 2048
train_split_size = 0.8
data_loader_workers = 5

params_dict = {
    "epochs": epochs,
    "lr": learning_rate,
    "dr": dropout_rate,
    "batches": batch_size,
    "layers": num_layers,
    "hidden_size": hidden_size,
}

# %% [markdown]
# Complete neural model of TypeWriter

# %%
model = TWModel(
    input_size, hidden_size, config_TW.AVAILABLE_TYPES_NUMBER, num_layers, output_size
).to(device)

# %% [markdown]
# The neural model of TypeWriter without available types

# %%
model = TWModelA(input_size, hidden_size, num_layers, output_size).to(device)

# %% [markdown]
# The complete neurel model of TypeWriter with Dropout

# %%
model = EnhancedTWModel(
    input_size,
    hidden_size,
    config_TW.AVAILABLE_TYPES_NUMBER,
    num_layers,
    output_size,
    dropout_rate,
).to(device)

# %% [markdown]
# Data parallesim for mutli-GPUs

# %%
model = torch.nn.DataParallel(model)

# %%
model.module.__class__.__name__

# %%
idx_of_other = pickle.load(open(LABEL_ENCODER_PATH_TW, "rb")).transform(["other"])[0]

for d in datasets_train:
    print(f"Loading {d} data for model {model.module.__class__.__name__}")
    # X_id, X_tok, X_cm, X_type, Y = datasets[d]
    load_data_t = time.time()
    X_id_train, X_tok_train, X_cm_train, X_type_train, Y_train = datasets_train[d]()
    X_id_test, X_tok_test, X_cm_test, X_type_test, Y_test = datasets_test[d]()

    train_loader = DataLoader(
        TensorDataset(X_id_train, X_tok_train, X_cm_train, X_type_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=data_loader_workers,
    )

    test_loader = DataLoader(
        TensorDataset(X_id_test, X_tok_test, X_cm_test, X_type_test, Y_test),
        batch_size=batch_size,
    )
    print("Loaded train and test sets in %.2f min" % ((time.time() - load_data_t) / 60))

    for i in range(1, n_rep + 1):

        train_t = time.time()
        train_loop_TW(model, train_loader, learning_rate, epochs)
        print("Training finished in %.2f min" % ((time.time() - train_t) / 60))
        eval_t = time.time()
        y_true, y_pred = evaluate_TW(model, test_loader, top_n=max(top_n_pred))
        print("Prediction finished in %.2f min" % ((time.time() - eval_t) / 60))

        # Ignore other type
        idx = (y_true != idx_of_other) & (y_pred[:, 0] != idx_of_other)
        f1_score_top_n = []
        for top_n in top_n_pred:
            filename = f"{model.module.__class__.__name__}_{d}_{i}_{top_n}"

            report_TW(
                y_true,
                y_pred,
                top_n,
                f"{filename}_unfiltered",
                RESULTS_DIR,
                params_dict,
            )
            report = report_TW(
                y_true[idx],
                y_pred[idx],
                top_n,
                f"{filename}_filtered",
                RESULTS_DIR,
                params_dict,
            )
            f1_score_top_n.append(report["macro avg"]["f1-score"])
        print("Mean f1_score:", mean(f1_score_top_n))

        model.module.reset_model_parameters()

# %% [markdown]
# ### Naive Baseline Model

# %%
idx_of_other = pickle.load(open(LABEL_ENCODER_PATH_TW, "rb")).transform(["other"])[0]

baseline_model = BaseLineModel(TYPES_FILE_TW)

for d in datasets_train:
    print(f"Loading {d} data for model {baseline_model.__class__.__name__}")

    X_id_test, X_tok_test, X_cm_test, X_type_test, y_test = datasets_test[d]()
    y_test = y_test.numpy()

    y_pred = baseline_model.predict(X_id_test)

    # Ignore other type
    idx = (y_test != idx_of_other) & (y_pred[:, 0] != idx_of_other)
    f1_score_top_n = []
    for top_n in top_n_pred:
        filename = f"{baseline_model.__class__.__name__}_{d}_{1}_{top_n}"
        report_TW(
            y_test,
            y_pred,
            top_n,
            f"{filename}_unfiltered",
            RESULTS_DIR,
            params={"model": "baseline"},
        )
        report = report_TW(
            y_test[idx],
            y_pred[idx],
            top_n,
            f"{filename}_filtered",
            RESULTS_DIR,
            params={"model": "baseline"},
        )
        print(report["weighted avg"])
        f1_score_top_n.append(report["weighted avg"]["f1-score"])
    print("Mean f1_score:", mean(f1_score_top_n))

# %% [markdown]
# ## Results

# %%
import result_proc

reload(result_proc)

# %%
res = result_proc.eval_result(
    RESULTS_DIR, "EnhancedTWModel", "return", "filtered", True
)

# %% [markdown]
# Plotting the results

# %%
result_proc.plot_result(res, "NaiveBaseline-Return-MacroAvg")

# %%
result_proc.copy_results(RESULTS_DIR, "./results/")

# %% [markdown]
# ## RayTune

# %%
from ray import tune
import ray

ray.init(memory=16 * 1024 * 1024 * 1024, object_store_memory=8 * 1024 * 1024 * 1024)

# %%
train_loader, test_loader = tune.utils.pin_in_object_store(
    train_loader
), tune.utils.pin_in_object_store(test_loader)

# %%
idx_of_other = tune.utils.pin_in_object_store(idx_of_other)

# %%
# @ray.remote
def train_TW(config):
    top_n_pred = [1, 3, 5]
    model = TWModel(
        input_size,
        config["hidden_size"],
        X_types_param.shape[1],
        config["num_layers"],
        output_size,
        True,
    ).to(device)

    # for i in range(1, n_rep+1):
    i = 1

    train_loop_TW(
        model, config["train_loader"], config["learning_rate"], config["epochs"]
    )
    y_true, y_pred = evaluate_TW(
        model, config["test_loader"], top_n=max(top_n_pred)
    )
    #     learn.train_loop_TW(model, train_loader, config['learning_rate'], config['epochs'])
    #     y_true, y_pred = learn.evaluate_TW(model, test_loader, top_n=max(top_n_pred))

    # Ignore other type
    # idx_of_other = pickle.load(open(f'./output/ml_inputs/label_encoder.pkl', 'rb')).transform(['other'])[0]
    idx = (y_true != tune.utils.get_pinned_object(idx_of_other)) & (
        y_pred[:, 0] != tune.utils.get_pinned_object(idx_of_other)
    )
    f1_score_top_n = []
    for top_n in top_n_pred:
        filename = f"{TWModel.__name__}_complete_{i}_{top_n}"
        # learn.report_TW(y_true, y_pred, top_n, f"{filename}_unfiltered")
        report = report_TW(
            y_true[idx], y_pred[idx], top_n, y_true.shape[0], f"{filename}_filtered",
            para
        )
        f1_score_top_n.append(report["weighted avg"]["f1-score"])
    print("Mean f1_score:", mean(f1_score_top_n))
    ray.tune.track.log(mean_f1_score=mean(f1_score_top_n))


# %%
analysis = tune.run(
    train_TW,
    config={
        "hidden_size": tune.grid_search([32, 64, 128]),
        "num_layers": tune.grid_search([1]),
        "learning_rate": tune.grid_search([0.002]),
        "epochs": tune.grid_search([5]),
        "train_loader": train_loader,
        "test_loader": test_loader,
    },
    name="TypeWriter_model",
    resources_per_trial={"cpu": 2, "gpu": 2},
    verbose=1,
)
print("Best config: ", analysis.get_best_config(metric="mean_f1_score"))

# %%
ray.shutdown()

# %%
