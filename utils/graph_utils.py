import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from node2vec import Node2Vec
from torch_geometric.nn import Node2Vec as node2vec_gpu
import networkx as nx
import math
import cugraph
import cudf
import cupy as cp
from torch_geometric.nn import GATConv
from torch.optim import Adam
from proco.proco import ProCoLoss


def point_to_graph(point_cloud_distance, point_cloud_attr, k):
    kdtree = KDTree(point_cloud_distance, leaf_size=30, metric='euclidean')
    distances, indices = kdtree.query(point_cloud_distance, k)
    edge_index = []
    edge_attr = []
    for i in range(len(point_cloud_distance)):
        for index, j in enumerate(indices[i]):
            if i != j:
                edge_index.append([i, j])
                edge_attr.append(distances[i, index])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    data = Data(x=torch.tensor(point_cloud_attr, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    return data

def compute_modularity(graph,community,node_attr):
    score = 0
    for i in graph.nodes:
        for j in graph.nodes:
            # print(node_attr[i])
            if community[i] == community[j]:
                r1, g1, b1 = node_attr[i][0],node_attr[i][1],node_attr[i][2]
                r2, g2, b2 = node_attr[j][0],node_attr[j][1],node_attr[j][2]
                color_score= math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
                color_score = color_score / math.sqrt(3 * (255 ** 2))
                opacity_score = abs(node_attr[i][3] - node_attr[j][3])
                score += color_score + opacity_score
    print(score)
    return score

def frequency_clustering(features,pos,frozen_labels,label):
    result, max_score = None, None
    origin_id = torch.zeros(features.shape[0]).cuda()
    for i in range(features.shape[0]-1):
        origin_id[i] = i
    part_feature = features[frozen_labels == label]
    part_pos = pos[frozen_labels == label]
    origin_id = origin_id[frozen_labels == label]
    graph = to_networkx(point_to_graph(part_pos.detach().cpu().numpy(), part_feature.detach().cpu().numpy(), 3),to_undirected=True)
    for i in graph.nodes:
        for j in graph.neighbors(i):
            r1, g1, b1 = part_feature[i][0],part_feature[i][1],part_feature[i][2]
            r2, g2, b2 = part_feature[j][0],part_feature[j][1],part_feature[j][2]
            color_score= math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
            color_score = color_score / math.sqrt(3 * (255 ** 2))
            opacity_score = abs(part_feature[i][3] - part_feature[j][3])
            score = color_score + opacity_score
        if max_score is None or score > max_score:
            max_score = score
            result = origin_id[i]
    return result.item()

# def frequency_clustering(features,pos,frozen_labels,label):
#     result, max_score = None, None
#     origin_id = torch.zeros(features.shape[0]).cuda()
#     for i in range(features.shape[0]-1):
#         origin_id[i] = i
#     part_feature = features[frozen_labels == label]
#     part_pos = pos[frozen_labels == label]
#     origin_id = origin_id[frozen_labels == label]
#     graph = to_networkx(point_to_graph(part_pos.detach().cpu().numpy(), part_feature.detach().cpu().numpy(), 3),to_undirected=True)
#     for i in graph.nodes:
#         for j in graph.neighbors(i):
#             r1, g1, b1 = part_feature[i][0],part_feature[i][1],part_feature[i][2]
#             r2, g2, b2 = part_feature[j][0],part_feature[j][1],part_feature[j][2]
#             color_score= math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
#             color_score = color_score / math.sqrt(3 * (255 ** 2))
#             opacity_score = abs(part_feature[i][3] - part_feature[j][3])
#             score = color_score + opacity_score
#         if max_score is None or score > max_score:
#             max_score = score
#             result = origin_id[i]
#     return result.item()
            
#     optimized, community, modularity, best_community, best_modularity = False, dict(), 0, dict(), 0
#     for i in range(len(graph.nodes)):
#         community[i] = i
#     best_modularity, best_community = compute_modularity(graph, community,part_feature), community
#     while not optimized:
#         optimized = True
#         for node in graph.nodes:
#             temp_community = best_community
#             for neighbor in graph.neighbors(node):
#                 temp_community[node] = temp_community[neighbor]
#                 modularity = compute_modularity(graph,community,part_feature)
#                 if modularity > best_modularity:
#                     best_community = community
#                     best_modularity = modularity
#                     optimized = False
#         print(optimized)
#     return best_community
    
def nodeEmbedding_node2vec(graph):
    G = to_networkx(graph, to_undirected=True)
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=16)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)    # 这一步慢
    # fit 函数用来训练 Node2Vec 模型，参数说明：
    results = {str(node): model.wv[str(node)] for node in G.nodes()} # 这一行通过模型 model 获取每个节点的嵌入向量。
    embeddings = None
    embeddings = torch.stack([torch.tensor(x) for x in results.values()])
    # for x in results.values():
    #     embeddings = torch.tensor(x).unsqueeze(0) if embeddings is None else torch.cat((embeddings, torch.tensor(x).unsqueeze(0)))
    return embeddings


def nodeEmbedding_node2vec_new(graph):
    # 将 PyG 图转换为 NetworkX 图
    G = to_networkx(graph, to_undirected=True)
    
    # 使用 cudf 将 NetworkX 图转换为 CuGraph 所需的格式
    edges = cudf.DataFrame({'src': cp.array([e[0] for e in G.edges()]), 
                            'dst': cp.array([e[1] for e in G.edges()])})
    
    # 将图创建为 CuGraph 图
    G_cugraph = cugraph.Graph()
    G_cugraph.from_cudf_edgelist(edges, source='src', destination='dst')
    
    # 执行随机游走
    start_vertices = cp.array([node for node in G.nodes()])
    walk_length = 30
    num_walks = 200
    
    # 执行随机游走
    rw_result = cugraph.random_walk(G_cugraph, start_vertices, length=walk_length, max_depth=num_walks)
    
    # 生成节点嵌入（可以通过统计每个节点的游走轨迹来生成嵌入）
    embeddings = []
    for node in G.nodes():
        # 获取该节点在所有随机游走中的位置
        node_walks = rw_result[rw_result['vertex'] == node]
        # 计算每个节点的嵌入，可以通过统计游走历史来实现
        embedding = cp.mean(node_walks['walk'], axis=0)  # 计算每个节点的嵌入
        embeddings.append(embedding)
    
    # 将嵌入转换为 PyTorch 张量
    embeddings = torch.tensor(cp.array(embeddings), dtype=torch.float32)
    
    return embeddings


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        # feature fusion
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * 8, output_dim, heads=1, dropout=0.6)

        # # Regression layer for 6-dimensional output
        # self.regression = nn.Linear(output_dim, 3)
        #
        # zeroinit = True
        # if zeroinit:
        #     # 零初始化权重和偏置
        #     with torch.no_grad():
        #         self.regression.weight.zero_()  # 将权重初始化为 0
        #         self.regression.bias.zero_()  # 将偏置初始化为 0


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # 第一层 GAT
        x = self.conv2(x, edge_index)  # 第二层 GAT
        return x

        # embedding = F.relu(x)
        # out = self.regression(embedding)
        # return embedding, out

# def nodeEmbedding_gat(point_cloud_distance, point_cloud_attr, k, input_dim=11, hidden_dim=64, output_dim=64):
#     device = torch.device("cuda")
#     graph_data = point_to_graph(point_cloud_distance, point_cloud_attr, k)
#     graph_data = graph_data.to(device)

#     # 创建 GAT 模型
#     model = GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
#     model = model.to(device)
#     optimizer = Adam(model.parameters(), lr=0.005)

#     model.train()
#     # optimizer.zero_grad()
#     embeddings = model(graph_data)
    
#     return embeddings, optimizer


def nodeEmbedding_gat(gaussian, point_cloud_distance, point_cloud_attr, k):
    device = torch.device("cuda")

    graph_data = point_to_graph(point_cloud_distance, point_cloud_attr, k)
    graph_data = graph_data.to(device)
    # print("Outside: input size", graph_data.size())

    embeddings = gaussian.GAT(graph_data)

    return embeddings

    # embeddings, out = gaussian.GAT(graph_data)

    # return embeddings, out



def nodeEmbedding_gat_new(gaussians, point_cloud_distance, k):
    device = torch.device("cuda")

    # gpu = int(os.environ["LOCAL_RANK"])
    # define loss function (criterion)
    # criterion_ce = LogitAdjust(15).cuda(gpu)
    criterion_scl = ProCoLoss(contrast_dim=64, temperature=0.1, num_classes=16).to('cuda')

    point_cloud_attr = torch.cat(
                (gaussians._xyz, gaussians._features_dc.squeeze(1), gaussians._opacity, gaussians._rotation), dim=1)
                # (gaussians._xyz, gaussians._features_dc.squeeze(1), gaussians._opacity, gaussians._rotation, gaussians._scaling), dim=1)

    targets = torch.argmax(gaussians._objects_dc.squeeze(1), dim=1)

    graph_data = point_to_graph(point_cloud_distance, point_cloud_attr, k)
    graph_data = graph_data.to(device)

    embeddings = gaussians.GAT(graph_data)

    contrastive_logits = criterion_scl(embeddings, targets)

    return contrastive_logits




def node_clustering(point_cloud, point_cloud_attr, k=3):
    graph = point_to_graph(point_cloud, point_cloud_attr, 3).cuda()
    data = nodeEmbedding_node2vec(graph)
    affinity_propagation = AffinityPropagation(damping=0.9, preference=-50, max_iter=200, convergence_iter=15)
    affinity_propagation.fit(data)
    return affinity_propagation.labels_

def equal_node(node1,node2):
    pass

def equal_edges(edge1,edge2):
    pass

def graph_similar(graph1,graph2):
    graph1 = to_networkx(graph1, to_undirected=True)
    graph2 = to_networkx(graph2, to_undirected=True)
    distance = nx.optimize_graph_edit_distance(graph1, graph2)
    for dist in distance:
        print(dist)
    for dist in distance:
        return torch.tensor(dist,dtype=torch.float)