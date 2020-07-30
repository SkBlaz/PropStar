### get conjunct features efficiently
from itertools import combinations
import numpy as np
import scipy.sparse as sps
import queue
import tqdm
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


class conjunctVectorizer:
    def __init__(self, max_atoms=(1, 1), max_features=10000, binary=False):
        """
        Instantiation class -> set default values + queue size.
        """

        self.max_atoms = max_atoms
        self.max_features = max_features
        self.binary = binary
        self.overall_counts = {}
        self.queue_maxsize = 20000  ## increase this for richer feature space
        self.selected_witems = queue.PriorityQueue(self.queue_maxsize)
        self.min_score = 0
        self.overall_witem_counts = {}

    def fit(self, witem_documents):
        """
        A standard fit method (sklearn-style). It operates on a list of sets.
        """

        self.all_witems = set()
        for doc in witem_documents:
            for witem in doc:
                if not witem in self.overall_counts:
                    self.overall_counts[witem] = 1
                else:
                    self.overall_counts[witem] += 1

        for witem_order in range(self.max_atoms[0], self.max_atoms[1] + 1):
            logging.info("Now computing witem order {}".format(witem_order))
            all_combinations_of_order = list(
                combinations(self.overall_counts.keys(), witem_order))
            for candidate_feature in tqdm.tqdm(
                    all_combinations_of_order,
                    total=len(list(all_combinations_of_order))):
                if self.selected_witems.qsize() == self.queue_maxsize:
                    logging.info(
                        "Max priority queue size achieved. Breaking ..")
                    break

                all_counts = [
                    self.overall_counts[x] for x in candidate_feature
                ]
                score = 1 / np.mean(all_counts)
                priority_score = np.mean(all_counts)
                self.overall_witem_counts[candidate_feature] = score
                self.selected_witems.put((-priority_score, candidate_feature))

        self.top_witems = []
        for x in range(self.max_features):
            if self.selected_witems.empty():
                break
            witem = self.selected_witems.get()[1]
            self.top_witems.append(witem)

    def get_feature_names(self):

        if self.top_witems:
            names = [" AND ".join(x) for x in self.top_witems]
            return names

    def transform(self, witem_documents):
        """
        Once fitted, this method transforms a list of witemsets to a sparse matrix.
        """

        logging.info("Preparing to transform witem documents ..")
        query = []
        key = []
        values = []
        doc_count = 0
        for idx, doc in enumerate(witem_documents):
            doc_count += 1
            for enx, witemset in enumerate(self.top_witems):
                if doc.intersection(set(witemset)):
                    query.append(idx)
                    key.append(enx)
                    if self.binary:
                        values.append(1)
                    else:
                        values.append(self.overall_witem_counts[witemset])

        assert len(query) == len(key)
        m = sps.csr_matrix((values, (query, key)),
                           shape=(doc_count, self.max_features))
        return m

    def fit_transform(self, witem_documents):
        """
        Fit and transform in a single call.
        """

        self.fit(witem_documents)
        return self.transform(witem_documents)
