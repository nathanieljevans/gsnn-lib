import torch

def augment_edge_index(edge_index, N, seed=0):
    
    num_nodes = torch.max(edge_index) + 1 

    edge_set = set([(i.item(),j.item()) for i,j in zip(*edge_index)])

    row = []; col=[]; n=0
    torch.random.manual_seed(seed)
    while n < N: 
        print(f'generating false edges: {n+1}/{N}', end='\r')
        i = torch.randint(0, num_nodes, size=(1,))
        j = torch.randint(0, num_nodes, size=(1,))

        if ((i.item(), j.item()) in edge_set) or (i == j):
            continue 
        else: 
            row.append(i.item())
            col.append(j.item())
            n+=1
    print()

    
    new_edges = torch.stack((torch.tensor(row, dtype=torch.long),
                             torch.tensor(col, dtype=torch.long)), dim=0)
    
    edge_index = torch.cat((edge_index, new_edges), dim=1)

    mask = torch.ones((edge_index.size(1),), dtype=torch.bool)
    mask[-N:] = False 

    return edge_index, mask
