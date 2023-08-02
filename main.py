import DFCNRSP.opt as opt
import torch
import numpy as np
from DFCNRSP.DFCN import DFCN
from DFCNRSP.utils import setup_seed
from sklearn.decomposition import PCA
from DFCNRSP.load_data import LoadDataset, load_graph, construct_graph
from DFCNRSP.train import Train, acc_reuslt, nmi_result, f1_result, ari_result

setup_seed(np.random.randint(100))

print("network setting…")

if opt.args.name == 'acm':
    opt.args.k = None
    opt.args.n_clusters = 3
    opt.args.n_input = 100
elif opt.args.name == 'dblp':
    opt.args.k = None
    opt.args.n_clusters = 4
    opt.args.n_input = 50
else:
    print("error!")

### cuda
print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

### root
opt.args.data_path = 'data/{}.txt'.format(opt.args.name)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.name)
opt.args.graph_k_save_path = 'graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
opt.args.graph_save_path = 'graph/{}_graph.txt'.format(opt.args.name)
opt.args.pre_model_save_path = 'model/model_pretrain/{}_pretrain.pkl'.format(opt.args.name)
opt.args.final_model_save_path = 'model/model_final/{}_final.pkl'.format(opt.args.name)
opt.args.walk_length = 2
opt.args.num_walk = 20



### data pre-processing
print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))

graph = ['acm', 'dblp']

x = np.loadtxt(opt.args.data_path, dtype=float)
y = np.loadtxt(opt.args.label_path, dtype=int)

pca = PCA(n_components=opt.args.n_input)
x1= pca.fit_transform(x)
n = x.shape[0]
if opt.args.name == "acm":
    dataset = LoadDataset(x1)
elif opt.args.name == "dblp":
    dataset = LoadDataset(x)
else:
    print('error dataset')
data = torch.Tensor(dataset.x).to(device)
label = y

%这一注释掉的部分可以用来生成基于随机游走构造的邻接矩阵，即，我将这些矩阵存储为 数据集名称_dw的格式
"""
adj,walker = load_graph(opt.args.k, opt.args.graph_k_save_path, opt.args.graph_save_path, opt.args.data_path,opt.args.walk_length,opt.args.num_walk)
adj = adj.to_dense()
adj = adj.to(device)

freq_mat = np.zeros([n,n])

for key in walker.walks_dict: #初始节点i
  for i in range(len(walker.walks_dict[key])):#从该节点出发的路径
    for j in range(len(walker.walks_dict[key][i])):#该路径上的点j
      freq_mat[int(key),int(walker.walks_dict[key][i][j])] +=1#f(i,j)++
freq_mat /= opt.args.num_walk

for i in range(len(freq_mat)):
    if freq_mat[i,i] == 0:
        freq_mat[i,i] = 1



dot_prod = np.matmul(freq_mat, np.transpose(freq_mat))
dot_sum = np.sum(dot_prod,axis=-1)

dot_kernel = dot_prod / dot_sum   #归一化？？？ kernel T
#print(dot_kernel)

dot_kernel = (torch.from_numpy(dot_kernel)).to(torch.float32)"""

%就是上面注释掉的部分生成的，可以自己生成
dw = torch.load("{}_dw".format(opt.args.name))
###  model definition
model = DFCN(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
             ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
             gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
             gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
             n_input=data.shape[1],
             n_z=opt.args.n_z,
             n_clusters=opt.args.n_clusters,
             v=opt.args.freedom_degree,
             n_node=data.size()[0],
             device=device).to(device)

### training
print("Training on {}…".format(opt.args.name))

if opt.args.name == "acm":
    lr = opt.args.lr_acm
elif opt.args.name == "dblp":
    lr = opt.args.lr_dblp
else:
    print("missing lr!")

Train(opt.args.epoch, model, data, dw.to(device), label, lr, opt.args.pre_model_save_path, opt.args.final_model_save_path,
      opt.args.n_clusters, opt.args.acc, opt.args.gamma_value, opt.args.lambda_value, device)


print("ACC: {:.4f}".format(max(acc_reuslt)))
print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])

