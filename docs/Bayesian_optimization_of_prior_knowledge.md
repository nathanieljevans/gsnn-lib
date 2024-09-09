
# Bayesian optimization of prior knowledge selection and inference of new interactions 

One of the noted limitations of the GSNN method is the difficulty of selecting appropriate prior knowledge. This is particularly problematic since the computational complexity limits application to large graphs, so inclusion of all prior knowledge is intractable and inefficient.  To remedy this, we propose a novel data-driven approach to optimal selection of prior knowledge. Notably, this approach could also be applied to inference of new (unknown) edges in the graph, which is another limitation of the method. 

## Method 

-- This might work better as reinforcement learning -- 
-- could still use KG embeddings as input to the policy -- 

Recent methods have shown remarkable ability of knowledge graphs to infer new edges and create embeddings informative for many predictive tasks. We propose to generate a GSNN-specific knowledge graph that describes the known relationships between drugs, proteins, rnas as well as auxillary relationships that may be predictive of entity inclusion such as pathway membership, gene ontologies, disease associations, drug-drug similarity, etc. Given a knowledge graph $\mathcal{G}_{KG}$, which must include all drugs, proteins and RNAs that we intend to optimize over for GSNN graph ($\mathcal{G}$), we will then use an embedding method to create node or edge embeddings that encode KG relationship information. Potential options for this approach include DeepWalk, Node2vec, GNN autoencoders, or KG embedding methods like Complex^2, TransE, etc. 

Defining the KG node $i$ embedding as $e_i$, we can then train a model to predict GSNN validation performance, given a set of node embeddings, $E={e_1, e_2, ..., e_n}. 

$$ \bar{\rho}_{gsnn} = f_{\phi}(E) $$ 

where $$ \bar{\rho}_{gsnn}$ is the average pearson correlation over all the gene nodes when evaluated on the GSNN. 

Using traditional bayesian optimization methods, we can iteratively explore the prior knowledge space by suggesting likely node embeddings and then training the respective GSNN model to get resulting performance information to update $\phi$ for the next suggestions. Appropriate implementation of this should converge to the optimal selection of nodes. 

Note, however, that this approach is conditional on an unchanging set of observations (drugs, omic features, cell-lines, and gene outputs). So the drugs and genes must be specified by the user prior to prior knowledge optimization. We can get around fixed omic features by only including the respective omic connections if they are selected by the bayesian optimization procedure. For instance, many omic features can be isolates and therefore not predict outcome. 

Interestingly, this approach could also be implemented to select edges instead of nodes (use edge embeddings rather than node embeddings), which would allow edge specific knowledge selection. If we constrain the edge selection to only known interactions, than we will optimize the selection of available prior knowledge. If, however, we allow selection of unknown interactions, we can infer novel interactions via bayesian optimization. Notably, this has the potential to overfit quickly and relies on informative edge representations, but could mitigate issues of missing prior knowledge. 

## Limitations 

### time complexity 

Notably, it can take several hours to train a GSNN model, to effectively explore prior knowledge selection we would either need to parrallelize this or accelerate the training process significantly. 

potential solutions: 
    - use small models; few channels, few layers 
    - select small subsets of drugs (inputs), genes (outputs)
    - implement drug-specific forward passes (potential significant gains ~ 2-10x est.)
    - compile pyg convolution 

NOTE: I think the best approach is to choose only a few drugs, this will drastically limit the number of observations. 

### node/edge embedding information content 

For this to work, we expect the node/edge embeddings to be representative of the inclusion criteria, this should be predictive of pathway membership, role, function, local neighborhood, etc. 

Additionally, we need a model to parameterize $\phi$ that can effectively operate on a set of varying size. This could be done with simple permutation invariant aggregations (mean, sum, max, etc), or we could use a method like RNNs, LSTM, Transformer, etc. My intuition is a Transformer would actually work fairly well, but definitely start with averaging embeddings with a vanilla nn. 

## Extensions 

- There is potential that we could "reuse" nodes, and therefore explore the selection of prior knowledge in a single training run. This might be done by a "selection" model, which chooses which function nodes to include and a GSNN model, where we load only the selected function nodes into memory and evaluate performance. This would be a lot faster, but the challenge is that function nodes are likely to learn within context of the nodes that are selected, so random selection of nodes may not appropriately represent the utility of a node set. 

- performing this process within a single cell line would remove the need for omic features and produce cell line specific graphs. 

- there is potential to train the KG node embeddings directly on the GSNN validation performances... unlikely to be effective without KG embedding but could be a fine tuning step. 



