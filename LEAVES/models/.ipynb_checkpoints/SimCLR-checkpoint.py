import math
import torch
import torch.nn as nn
from models.viewmaker import ViewMaker
from models.resnet import ResNetEncoder
import torch.nn.functional as F

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

class SimCLRObjective(torch.nn.Module):
    
    def __init__(self, outputs1, outputs2, t, push_only=False):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t
        self.push_only = push_only

    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        witness_score = torch.sum(self.outputs1 * self.outputs2, dim=1)
        if self.push_only:
            # Don't pull views together.
            witness_score = 0
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        witness_norm = self.outputs1 @ outputs12.T
        witness_norm = torch.logsumexp(witness_norm / self.t, dim=1) - math.log(2 * batch_size)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss

class SimCLR(nn.Module):
    def __init__(self, viewmaker_config, encoder_config):
        super().__init__()
        self.viewmaker_config = viewmaker_config
        if self.viewmaker_config['use_leaves']:
            self.view = self.create_viewmaker(viewmaker_config)
        self.encoder = self.create_encoder(encoder_config)
        
    def create_viewmaker(self, viewmaker_config):
        view_model = ViewMaker(num_channels = viewmaker_config['num_channels'],
                               distortion_budget = viewmaker_config['view_bound_magnitude'],
                               clamp = viewmaker_config['clamp'])
        return view_model
    
    def create_encoder(self, encoder_config):
        encoder = ResNetEncoder(
                        in_channels=encoder_config['in_channels'], 
                        base_filters=encoder_config['base_filters'],
                        kernel_size=encoder_config['kernel_size'], 
                        stride=encoder_config['stride'], 
                        groups=1, 
                        n_block=encoder_config['n_block'], 
                        downsample_gap=encoder_config['downsample_gap'], 
                        increasefilter_gap=encoder_config['increasefilter_gap'], 
                        use_do=True)
        return encoder
    
    def forward(self, x1, x2):
        if self.viewmaker_config['use_leaves']:
            x1 = self.view(x1)
            x2 = self.view(x2)
        
        view1_emb = self.encoder(x1)
        view2_emb = self.encoder(x2)
        
        return view1_emb, view2_emb

