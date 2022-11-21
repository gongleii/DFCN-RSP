from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import DFCNRSP.opt as opt
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from DFCNRSP.utils import adjust_learning_rate
from DFCNRSP.utils import eva, target_distribution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
acc_reuslt = []
acc_reuslt.append(0)
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = ['acm', 'dblp', 'cite']


def Train(epoch, model, data, adj, label, lr, pre_model_save_path, final_model_save_path, n_clusters,
          original_acc, gamma_value, lambda_value, device):
    optimizer = Adam(model.parameters(), lr=lr)
    model.load_state_dict(torch.load(pre_model_save_path, map_location='cpu'))
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde = model(data, adj)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(label, cluster_id, 'Initialization')
    lr_s = StepLR(optimizer, step_size=50, gamma=1)
    for ep in range(epoch+1):

        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde = model(data, adj)
        tmp_q = q.data
        p = target_distribution(tmp_q)
        loss_ae = F.mse_loss(x_hat, data)
        loss_w = F.mse_loss(z_hat, torch.mm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj)
        loss_igae = loss_w + gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_igae + lambda_value * loss_kl
        print('{} loss: {}'.format(ep, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

        acc, nmi, ari, f1 = eva(label, kmeans.labels_, ep)
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

        if acc > original_acc:
            original_acc = acc
            torch.save(model.state_dict(), final_model_save_path)
        lr_s.step()
        print(epoch, lr_s.get_last_lr()[0])

