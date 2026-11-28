"""Item-based CF from training-only binary user/item incidence vectors."""
import numpy as np
from scipy import sparse


def interaction_matrix(histories,num_items):
    rows=[]; cols=[]
    for row,items in enumerate(histories.values()):
        for item in set(items):
            if not 1<=item<=num_items:
                raise ValueError("Invalid item ID.")
            rows.append(row); cols.append(item-1)
    return sparse.csr_matrix((np.ones(len(rows),dtype=np.float32),(rows,cols)),
                             shape=(len(histories),num_items))


class ItemCF:
    def __init__(self,similarity):
        self.similarity=similarity.tocsr()
        if similarity.shape[0]!=similarity.shape[1]:
            raise ValueError("Expected square item similarities.")
        self.num_items=similarity.shape[0]

    @classmethod
    def fit(cls,histories,num_items,metric="jaccard"):
        if metric not in ("jaccard","cosine"):
            raise ValueError("Unknown similarity metric.")
        x=interaction_matrix(histories,num_items)
        counts=np.asarray(x.sum(axis=0)).ravel()
        cooccurrence=(x.T@x).tocoo()
        keep=cooccurrence.row!=cooccurrence.col
        row,col,overlap=cooccurrence.row[keep],cooccurrence.col[keep],cooccurrence.data[keep]
        if metric=="jaccard":
            denominator=counts[row]+counts[col]-overlap
        else:
            denominator=np.sqrt(counts[row]*counts[col])
        values=overlap/denominator
        similarities=sparse.csr_matrix((values,(row,col)),shape=(num_items,num_items))
        similarities.eliminate_zeros()
        return cls(similarities)

    def score_all(self,users,histories):
        # Sum similarity to every observed item. All neighbors are retained.
        return (interaction_matrix(dict(enumerate(histories)),self.num_items)@self.similarity).toarray()
