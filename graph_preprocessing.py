# graph_preprocessing.py
import torch
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from torch_geometric.utils import dense_to_sparse
from slic_simple import SP_SLIC


def prepare_graph_data_from_files(data1_path, data2_path, scale=400):

    img1 = sio.loadmat(data1_path)['HypeRvieW']
    img2 = sio.loadmat(data2_path)['HypeRvieW']
    h, w, b = img1.shape

    img1 = img1.reshape(-1, b)
    scaler = preprocessing.StandardScaler()
    img1 = scaler.fit_transform(img1).reshape(h, w, b)
    img2 = img2.reshape(-1, b)
    img2 = scaler.fit_transform(img2).reshape(h, w, b)

    dummy_gt = np.zeros((h, w), dtype=np.int32)
    slic_obj = SP_SLIC(img1, img2, dummy_gt, n_component=1)
    Q1, S1, A1, Q2, S2, A2, _ = slic_obj.simple_superpixel_no_LDA(scale=scale)

    A1_t = torch.from_numpy(A1).float()
    A2_t = torch.from_numpy(A2).float()
    I = torch.eye(A1_t.shape[0])
    edge_index1, edge_weight1 = dense_to_sparse(A1_t + I)
    edge_index2, edge_weight2 = dense_to_sparse(A2_t + I)

    return {
        'x1': torch.from_numpy(img1).float(),
        'x2': torch.from_numpy(img2).float(),
        'Q1': torch.from_numpy(Q1).float(),
        'edge_index1': edge_index1,
        'edge_weight1': edge_weight1,
        'Q2': torch.from_numpy(Q2).float(),
        'edge_index2': edge_index2,
        'edge_weight2': edge_weight2,
        'height': h, 'width': w, 'channel': b
    }


def prepare_graph_data_demo():

    print("=== Running in DEMO mode (synthetic data) ===")
    H, W, C = 64, 64, 30  
    N_superpixels = 200


    x1 = torch.randn(H, W, C).float()
    x2 = torch.randn(H, W, C).float()


    Q1 = torch.zeros(H * W, N_superpixels)
    for i in range(H * W):
        Q1[i, np.random.randint(0, N_superpixels)] = 1
    Q2 = Q1.clone()  


    A = torch.rand(N_superpixels, N_superpixels)
    A = (A + A.T) / 2
    A = A > 0.7
    A = A.float()
    I = torch.eye(N_superpixels)
    edge_index1, edge_weight1 = dense_to_sparse(A + I)
    edge_index2, edge_weight2 = edge_index1.clone(), edge_weight1.clone()

    return {
        'x1': x1, 'x2': x2,
        'Q1': Q1, 'edge_index1': edge_index1, 'edge_weight1': edge_weight1,
        'Q2': Q2, 'edge_index2': edge_index2, 'edge_weight2': edge_weight2,
        'height': H, 'width': W, 'channel': C
    }



if __name__ == "__main__":
    import sys


    if len(sys.argv) == 3:
        print("Using real data from provided files...")
        data = prepare_graph_data_from_files(sys.argv[1], sys.argv[2])
    else:
        data = prepare_graph_data_demo()


    from model import ImprovedNet


    net = ImprovedNet(
        height=data['height'], width=data['width'], channel=data['channel'], class_count=2,
        Q1=data['Q1'], edge_index1=data['edge_index1'], edge_weight1=data['edge_weight1'],
        Q2=data['Q2'], edge_index2=data['edge_index2'], edge_weight2=data['edge_weight2']
    )


    with torch.no_grad():
        output = net(data['x1'], data['x2'])
    print(f"Forward pass successful. Output shape: {output.shape}")
    print(f"Model has {sum(p.numel() for p in net.parameters())} parameters.")
