## propStar example use, skrlj 2020 use at own discretion

import pandas as pd
import queue
import networkx as nx
import tqdm
from collections import defaultdict, OrderedDict
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn import preprocessing
import re

from neural import * ## DRMs
from learning import * ## starspace
from vectorizers import * ## ConjunctVectorizer

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

class OrderedDictList(OrderedDict):
    def __missing__(self,k):
        self[k] = []
        return self[k]

def cleanp(stx):

    """
    Simple string cleaner
    """
    
    return stx.replace("(","").replace(")","").replace(",","")

def interpolate_nans(X):

    """
    Simply replace nans with column means for numeric variables.
    input: matrix X with present nans
    output: a filled matrix X
    """
    
    for j in range(X.shape[1]):
        mask_j = np.isnan(X[:,j])
        X[mask_j,j] = np.mean(np.flatnonzero(X))
    return X

def discretize_candidates(df, types,ratio_threshold = 0.20, n_bins = 20):

    """
    Continuous variables are discretized if more than 30% of the rows are unique.
    """
    
    ratio_storage= {}
    for enx, type_var in enumerate(types):
        if "int" in type_var or "decimal" in type_var or "float" in type_var:
            ratio_storage = 1.*df[enx].nunique()/df[enx].count()
            if ratio_storage > ratio_threshold and ratio_storage != 1.0:
                to_validate = df[enx].values                
                parsed_array = np.array([np.nan if x == "NULL" else float(x) for x in to_validate])
                parsed_array = interpolate_nans(parsed_array.reshape(-1,1))
                to_be_discretized = parsed_array.reshape(-1,1)
                var = KBinsDiscretizer(encode = "ordinal", n_bins = n_bins).fit_transform(to_be_discretized)
                df[enx] = var
                if np.isnan(var).any():
                    continue ## discretization fail
                df[enx] = df[enx].astype(str) ## cast back to str.                   
    return df

def clear(stx):
    """
    Clean the unneccesary parenthesis
    """
    
    return stx.replace("`","").replace("`","")

def table_generator(sql_file,variable_types):

    """
    A simple SQLite parser. This is inspired by the official SQL library, yet keeps only minimal overhead.
    input: a .sql data dump from e.g., relational.fit.cz
    output: Pandas represented-linked dataframe
    """
    
    table_trigger = False
    table_header = False
    current_table = None
    sqt = defaultdict(list)
    tabu = ["KEY","PRIMARY","CONSTRAINT"]
    table_keys = defaultdict(list)
    primary_keys = {}
    foreign_key_graph = []
    fill_table = False
    tables = dict()
    header_init = False
    col_types = []

    ## Read the file table-by-table (This could be done in a lazy manner if needed)
    with open (sql_file,"r", encoding = "utf-8", errors = "ignore") as sqf:
        for line in sqf:
            
            if "CREATE TABLE" in line:
                header_init = True

            if header_init:
                if "DEFAULT" in line:
                    if "ENGINE" in line:
                        continue
                    
                    ctype = line.split()[1]
                    col_types.append(ctype)
                
            if "INSERT INTO" in line:

                ## Do some basic cleaning and create the dataframe
                table_header=False
                header_init = False
                vals = line.strip().split()
                vals_real = " ".join(vals[4:]).split("),(")
                vals_real[0] = vals_real[0].replace("(","")
                vals_real[len(vals_real)-1] = vals_real[len(vals_real)-1].replace(");","")
                col_num = len(sqt[current_table])
                
                vx = list(filter(lambda x: len(x) == col_num,[re.split(r",(?=(?:[^\']*\'[^\']*\')*[^\']*$)", x) for x in vals_real]))

                
                if len(vx) == 0:

                    ## this was added for the movies.sql case
                    vx = []
                    
                    for x in vals_real:
                        parts = x.split(",")
                        vx.append(parts[len(parts)-col_num :])
               
                dfx = pd.DataFrame(vx)

                ## Discretize continuous attributes.
#                if dfx.shape[1] == len(col_types):
#                    dfx = discretize_candidates(dfx,col_types)
                    
                col_types = []

                try:
                    assert dfx.shape[1] == len(sqt[current_table])
                    
                except:
                    logging.info(sqt[current_table])
                    logging.info(col_num,re.split(r",(?=(?:[^\']*\'[^\']*\')*[^\']*$)", vals_real[0]))


                try:
                    dfx.columns = [clear(x) for x in sqt[current_table]] ## some name reformatting.
                except:
                    dfx.columns = [x for x in sqt[current_table]] ## some name reformatting.                 

                tables[current_table] = dfx

            ## get the foreign key graph.
            if table_trigger and table_header:
                line = line.strip().split()
                if len(line) > 0:
                    if line[0] not in tabu:
                        if line[0] != "--":                            
                            if re.sub(r'\([^)]*\)', '', line[1]).lower() in variable_types:
                                sqt[current_table].append(clear(line[0]))
                    else:
                        if line[0] == "KEY":
                            table_keys[current_table].append(clear(line[2]))
                            
                        if line[0] == "PRIMARY":
                            primary_keys[current_table] = cleanp(clear(line[2]))
                            table_keys[current_table].append(clear(line[2]))
                            
                        if line[0] == "CONSTRAINT":
                            ## Structure in the form of (t1 a1 t2 a2) is used.
                            foreign_key_quadruplet = [clear(cleanp(x)) for x in [current_table,line[4],line[6],line[7]]]
                            foreign_key_graph.append(foreign_key_quadruplet)

            if "CREATE TABLE" in line:
                table_trigger = True
                table_header = True
                current_table = clear(line.strip().split(" ")[2])
                
    return tables,foreign_key_graph,primary_keys

def get_table_keys(quadruplet):

    """
    A basic method for gaining a given table's keys.
    """
    
    tk = defaultdict(set)
    for entry in quadruplet:
        tk[entry[0]].add(entry[1])
        tk[entry[2]].add(entry[3])
    return tk

def relational_words_to_matrix(fw,relation_order, vectorization_type = "tfidf", max_features = 10000):

    """
    Employ the conjuncVectorizer to obtain zero order features.
    input: documents
    output: a sparse matrix
    """
    
    docs = []

    if vectorization_type == "tfidf" or vectorization_type == "binary": 
        if vectorization_type == "tfidf":            
            vectorizer = conjunctVectorizer(max_atoms=relation_order, max_features = max_features)
        elif vectorization_type == "binary":
            vectorizer = conjunctVectorizer(max_atoms=relation_order, binary = True, max_features = max_features)            
        for k,v in fw.items():
            docs.append(set(v))
        mtx = vectorizer.fit_transform(docs)

    elif vectorization_type == "sklearn_tfidf" or vectorization_type == "sklearn_binary" or vectorization_type == "sklearn_hash":

        if vectorization_type == "sklearn_tfidf":
            vectorizer = TfidfVectorizer(max_features = max_features, binary = True)
        elif vectorization_type == "sklearn_binary":
            vectorizer = TfidfVectorizer(max_features = max_features, binary = False)
        elif vectorization_type == "sklearn_hash":
            vectorizer = HashingVectorizer()
        
        for k,v in fw.items():
            docs.append(" ".join(v))
            
        mtx = vectorizer.fit_transform(docs)

    return mtx, vectorizer

def relational_words_to_matrix_with_vec(fw,vectorizer, vectorization_type = "tfidf"):

    """
    Just do the transformation. This is for proper cross-validation (on the test set)
    """
    
    docs = []
    if vectorization_type == "tfidf" or vectorization_type == "binary":
        for k,v in fw.items():
            docs.append(set(v))            
        mtx = vectorizer.transform(docs)
        
    else:
        for k,v in fw.items():
            docs.append(" ".join(v))
        mtx = vectorizer.transform(docs)

    return mtx

def generate_relational_words(tables,fkg,target_table=None,target_attribute=None,relation_order=(2,4), indices=None, vectorizer = None, vectorization_type = "tfidf", num_features = 10000):

    """
    Key method for generation of relational words and documents. 
    It traverses individual tables in path, and consequantially appends the witems to a witem set. This method is a rewritten, non exponential (in space) version of the original Wordification algorithm (Perovsek et al, 2014).
    input: a collection of tables and a foreign key graph
    output: a representation in form of a sparse matrix.
    """
    
    fk_graph = nx.Graph() ## a simple undirected graph as the underlying fk structure
    core_foreign_keys = set()
    all_foreign_keys = set()

    for foreign_key in fkg:

        ## foreing key mapping
        t1, k1, t2, k2 = foreign_key
        
        if t1 == target_table:
            core_foreign_keys.add(k1)
            
        elif t2 == target_table:
            core_foreign_keys.add(k2)
            
        all_foreign_keys.add(k1)
        all_foreign_keys.add(k2)

        ## add link, note that this is in fact a typed graph now
        fk_graph.add_edge((t1,k1),(t2,k2))

    ## this is more efficient than just orderedDict object
    feature_vectors = OrderedDictList()
    if not indices is None:
        core_table = tables[target_table].iloc[indices,:]
    else:
        core_table = tables[target_table]
    all_table_keys = get_table_keys(fkg)
    core_foreign = None
    target_classes = core_table[target_attribute]

    ## This is a remnant of one of the experiment, left here for historical reasons :)
    if target_attribute == "Delka_hospitalizace":
        tars = []
        for tc in target_classes:
            if int(tc) >= 10:
                tars.append(0)
            else:
                tars.append(1)
        target_classes = pd.DataFrame(np.array(tars))
        print(np.sum(tars)/len(target_classes))
        
    total_witems = set()
    num_witems = 0    

    ## The main propositionalization routine
    logging.info("Propositionalization of core table ..")
    for index, row in tqdm.tqdm(core_table.iterrows(), total = core_table.shape[0]):
        for i in range(len(row)):
            column_name = row.index[i]
            if column_name != target_attribute and not column_name in core_foreign_keys:
                witem = "-".join([target_table, column_name, row[i]])
                feature_vectors[index].append(witem)
                num_witems += 1
                total_witems.add(witem)
                
    logging.info("Traversing other tables ..")
    for core_fk in core_foreign_keys: ## this is normaly a single key.
        bfs_traversal = dict(nx.bfs_successors(fk_graph,(target_table,core_fk)))

        ## Traverse the row space
        for index, row in tqdm.tqdm(core_table.iterrows(), total = core_table.shape[0]):
                             
            current_depth = 0
            to_traverse = queue.Queue()
            to_traverse.put(target_table) ## seed table
            max_depth = 2
            tables_considered = 0
            parsed_tables = set()

            ## Perform simple search
            while current_depth < max_depth:
                current_depth += 1
                origin = to_traverse.get()
                if current_depth == 1:
                    successor_tables = bfs_traversal[(origin,core_fk)]
                else:
                    if origin in bfs_traversal:
                        successor_tables = bfs_traversal[origin]
                    else: continue                
                for succ in successor_tables:
                    to_traverse.put(succ)                    
                for table in successor_tables:
                    if (table) in parsed_tables:
                        continue                    
                    parsed_tables.add(table)
                    first_table_name, first_table_key = origin, core_fk
                    next_table_name, next_table_key = table
                    if not first_table_name in tables or not next_table_name in tables:
                        continue

                    ## link and generate witems
                    first_table = tables[first_table_name]
                    second_table = tables[next_table_name]                  
                    if first_table_name == target_table:
                        key_to_compare = row[first_table_key]                    
                    elif first_table_name != target_table and current_depth == 2:
                        key_to_compare = None
                        for edge in fk_graph.edges():
                            if edge[0][0] == target_table and edge[1][0] == first_table_name:
                                key_to_compare = first_table[first_table[edge[1][1]] == row[edge[0][1]]][first_table_key]
                        if not key_to_compare is None:
                            pass
                        else:
                            continue
                        
                    ## The second case
                    trow = second_table[second_table[next_table_key] == key_to_compare]
                    for x in trow.columns:                        
                        if not x in all_foreign_keys and x != target_attribute:
                            for value in trow[x]:
                                witem = "-".join(str(x) for x in [next_table_name, x, value])
                                total_witems.add(witem)
                                num_witems += 1
                                feature_vectors[index].append(witem)

    ## Summary of the output
    logging.info("Stored {} witems..".format(num_witems))
    logging.info("Learning representation from {} unique witems.".format(len(total_witems)))

    ## Vectorizer is an arbitrary vectorizer, some of the well known ones are implemented here, it's simple to add your own!
    if vectorizer:
        matrix = relational_words_to_matrix_with_vec(feature_vectors, vectorizer, vectorization_type = vectorization_type)
        return matrix, target_classes    
    else:
        matrix, vectorizer = relational_words_to_matrix(feature_vectors,relation_order, vectorization_type,  max_features = num_features)
        logging.info("Stored sparse representation of the witemsets.")
        return matrix, target_classes, vectorizer

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", default="starspace")
    parser.add_argument("--learning_rate", default = 0.001, type = float, help = "Learning rate of starspace")
    parser.add_argument("--epochs", default = 10, type = int, help = "Number of epochs")
    parser.add_argument("--dropout", default = 0.1, type = float, help = "Dropout rate")
    parser.add_argument("--num_features", default = 30000, type = int, help = "Number of features")
    parser.add_argument("--hidden_size", default = 16, type = int, help ="Embedding dimension")
    parser.add_argument("--negative_samples_limit", default = 10, type = int, help = "Max number of negative samples")
    parser.add_argument("--negative_search_limit", default = 10, type = int, help = "Negative search limit (see starspace docs for extensive description)")
    parser.add_argument("--representation_type", default = "tfidf", type = str, help = "Type of representation and weighting. tfidf or binary, also supports scikit's implementations (ordered patterns)")
    args = parser.parse_args()

    variable_types_file = open("variable_types.txt") ## types to be considered.
    variable_types = [line.strip().lower() for line in variable_types_file.readlines()]
    variable_types_file.close()
    learner = args.learner
    import os

    ## IMPORTANT: a tmp folder must be possible to construct, as the intermediary embeddings are stored here.
    directory = "tmp"
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    ## Traverse the data set space
    with open('datasets.txt') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip()[0] != "#":
                line = line.strip().split()
                example_sql = "./sql_data/"+line[0]
                target_table = line[1]
                target_attribute = line[2]
                
                logging.info("Running for example_sql: "+example_sql+", target_table: "+target_table+", target_attribute "+target_attribute)

                tables, fkg, primary_keys = table_generator(example_sql, variable_types)
                
                if learner == "DRM":
                    drm_grid = []
                    drm_grid.append([args.epochs,args.learning_rate,args.hidden_size,args.dropout,args.representation_type, args.num_features])

                    for pars in tqdm.tqdm(drm_grid):
                        perf= []
                        perf_roc = []
                        logging.info("Evaluation of {} - {}".format(pars, target_attribute))
                        split_gen = preprocess_and_split(tables[target_table], num_fold=10, target_attribute = target_attribute)
                        for train_index, test_index in split_gen:
                            ## higher relation orders result in high memory load, thread with caution!
                            train_features, train_classes, vectorizer = generate_relational_words(tables,fkg,target_table,target_attribute, relation_order=(1,2), indices = train_index, vectorization_type = pars[4], num_features = args.num_features)
                            test_features, test_classes = generate_relational_words(tables,fkg,target_table,target_attribute, relation_order=(1,2), vectorizer = vectorizer, indices = test_index, vectorization_type = pars[4], num_features = args.num_features)
                            
                            model = E2EDNN(num_epochs = pars[0],
                                           learning_rate = pars[1],
                                           hidden_layer_size = pars[2],
                                           dropout = pars[3])
                            le = preprocessing.LabelEncoder()
                            le.fit(train_classes.values)
                            
                            train_classes = le.transform(train_classes)
                            test_classes = le.transform(test_classes)

                            ## standard fit predict
                            model.fit(train_features, train_classes)
                            preds = model.predict(test_features)
                            acc1 = accuracy_score(preds,test_classes)
                            logging.info(acc1)
                            perf.append(acc1)
                            
                            if len(np.unique(test_classes)) == 2:
                                preds = model.predict(test_features, return_proba = True) 
                                roc = roc_auc_score(test_classes, preds)
                                logging.info(roc)
                                perf_roc.append(roc)
                                
                            else:
                                
                                perf_roc.append(0)

                        stx = "|".join(str(x) for x in pars)
                        mp = np.round(np.mean(perf),4)
                        mp_roc = np.round(np.mean(perf_roc), 4)
                        if mp != "nan" and mp != np.nan:
                            print("RESULT_LINE {} {} {} {} {} {} {}".format("DRM",mp_roc,mp,line[0],line[1],line[2], stx))
                        else:
                            pass
                            
                elif learner == "starspace":
                    starspace_grid = []
                    starspace_grid.append([args.epochs,args.learning_rate,args.negative_samples_limit,args.hidden_size,args.negative_search_limit, args.representation_type, args.num_features])

                    for pars in tqdm.tqdm(starspace_grid):
                        perf= []
                        perf_roc = []
                        logging.info("Evaluation of {}".format(pars))                      
                        split_gen = preprocess_and_split(tables[target_table], num_fold=10, target_attribute = target_attribute)
                        for train_index, test_index in split_gen:
                            train_features, train_classes, vectorizer = generate_relational_words(tables,fkg,target_table,target_attribute, relation_order=(1,2), indices = train_index, vectorization_type = pars[5], num_features = pars[6])
                            
                            test_features, test_classes = generate_relational_words(tables,fkg,target_table,target_attribute, relation_order=(1,2), vectorizer = vectorizer, indices = test_index, vectorization_type = pars[5], num_features = pars[6])
                            le = preprocessing.LabelEncoder()
                            le.fit(train_classes.values)                            
                            train_classes = le.transform(train_classes)
                            test_classes = le.transform(test_classes)
                            model = starspaceLearner(epoch=pars[0],
                                                     learning_rate = pars[1],
                                                     neg_search_limit = pars[2],
                                                     dim=pars[3],
                                                     max_neg_samples = pars[4])
                            

                            ## standard fit predict
                            model.fit(train_features, train_classes)
                            preds = model.predict(test_features, clean_tmp = False)

                            if len(preds)  == 0:
                                perf_roc.append(0)
                                perf.append(0)
                                continue

                            try:
                                acc1 = accuracy_score(test_classes, preds)
                                
                                logging.info(acc1)
                                perf.append(acc1)
                                
                                preds_scores = model.predict(test_features, clean_tmp = True, return_int_predictions = False, return_scores = True) ## use scores for auc.

                                if len(np.unique(test_classes)) == 2:
                                    roc = roc_auc_score(test_classes, preds_scores)
                                    perf_roc.append(roc)
                                    logging.info(roc)
                                else:
                                    ## not reported.
                                    perf_roc.append(0)
                                                                    
                            except Exception as es:
                                print(es)
                                continue

                        stx = "|".join(str(x) for x in pars)
                        mp = np.round(np.mean(perf),4)
                        mp_roc = np.round(np.mean(perf_roc),4)
                        if mp != "nan" and mp != np.nan:
                            print("RESULT_LINE {} {} {} {} {} {} {}".format("StarSpaceDirect",mp_roc,mp,line[0],line[1],line[2], stx))
                            
                        else:
                            pass
