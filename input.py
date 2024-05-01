from numpy import integer
import torch
from torch_geometric.data import Dataset, Data
import os
import json

from torch_geometric.data.data import BaseData

class ModelNet40(Dataset):
    def __init__(self,root,transform=None, pre_transform=None, partition="train", data_dir = '/CWE121/labeled_graphs'):
        self.partition = partition
        super(ModelNet40,self).__init__(root, transform, pre_transform)

    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.json')] 
    
    def process(self):
        for idx, name in enumerate(self.raw_file,names):
             with open(os.path.join(self.raw_dir,name),'r') as f:
                  graph_data = json.load(f)

        edge_index = torch.tensor(graph_data['edges'],dtype=torch.long).t().contiguous()
        x = torch.tensor(graph_data['edges'],dtype=torch.long).t().contiguous()
        y = torch.tensor([graph_data['label']],dtype = torch.long)
        edge_attr = torch.tensor(graph_data['edge_features'],dtype=torch.float)

        data = Data(x=x,edge_index=edge_index,y=y,edge_attr=edge_attr)
        torch.save(data,os.path.join(self.processed_dir, f'graph_{idx}.pt'))

    def __len__(self):
        return len(self.proessed_file_names)
    
    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.processed_dir,f'graph_{idx}.pt'))
        return data

    if __name__ == '__main__':
        root = '/CWE121/labeled_graphs'
        dataset = ModelNet40(root = root,partition='train')

        for data in dataset:
            print(data)
            break

             