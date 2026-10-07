"""Label mappings for datasets."""

# The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
# Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
# The dictionary consists of 1433 unique words.
label_dict_cora: dict[int, str] = {
    0: "Theory",
    1: "Reinforcement Learning",
    2: "Genetic Algorithms",
    3: "Neural Networks",
    4: "Probabilistic Methods",
    5: "Case Based",
    6: "Rule Learning",
}

# The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes.
# The citation network consists of 4732 links.
# Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
# The dictionary consists of 3703 unique words.
label_dict_citeseer: dict[int, str] = {
    0: "Agents",
    1: "Artificial Intelligence",
    2: "Database",
    3: "Information Retrieval",
    4: "Machine Learning",
    5: "Human Computer Interaction",
}

# The Pubmed Diabetes dataset consists of 19.717 scientific publications from the PubMed database pertaining to diabetes classified into one of three classes.
# The citation network consists of 44.338 links.
# Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.
label_dict_pubmed: dict[int, str] = {
    0: "Diabetes Mellitus, Experimental",
    1: "Diabetes Mellitus Type 1",
    2: "Diabetes Mellitus Type 2",
}

# Aggregate helper mapping
label_dicts: dict[str, dict[int, str]] = {
    "cora": label_dict_cora,
    "citeseer": label_dict_citeseer,
    "pubmed": label_dict_pubmed,
}
