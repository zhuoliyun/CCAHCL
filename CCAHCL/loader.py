from .dataset import (
    CoraCocitationDataset,
    CiteseerCocitationDataset,
    PubmedCocitationDataset,   
    CoraCoauthorshipDataset,
    DBLPCoauthorshipDataset,
    ModelNet40Dataset,
)


class DatasetLoader(object):
    def __init__(self):
        pass

    def load(self, dataset_name: str = 'cora'):
        if dataset_name == 'cora':
            return CoraCocitationDataset() 
        elif dataset_name == 'citeseer':
            return CiteseerCocitationDataset()
        elif dataset_name == 'pubmed':
            return PubmedCocitationDataset()
        elif dataset_name == 'cora_coauthor':
            return CoraCoauthorshipDataset()
        elif dataset_name == 'dblp_coauthor':
            return DBLPCoauthorshipDataset()
        elif dataset_name == 'ModelNet40':
            return ModelNet40Dataset()
        else:
            assert False
