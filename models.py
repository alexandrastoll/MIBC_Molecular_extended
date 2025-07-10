### modified and adapted from: https://www.kaggle.com/code/zznznb/wsi-train

### original publication ABMIL: https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf
### original publication DSMIL: https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    
    def __init__(self, in_dim, feat_dim, attn_dim, num_classes):
        super().__init__()
        self.downlinear = nn.Sequential(nn.Linear(in_dim, feat_dim), nn.ReLU())
        self.attention_V = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(feat_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downlinear(x)

        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = torch.softmax(A, dim=1)
        x = torch.mm(A, x)

        scores = self.classifier(x)

        return scores

class IClassifier(nn.Module):

    def __init__(self, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, feats):
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):

    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(nn.Module):

    def __init__(self, i_classifier, b_classifier):
        super(DSMIL, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B


class MeanPoolingMILREG(nn.Module):
    """MLP with meanpooling and multi-point regression"""

    import torch.nn as nn
    
    def __init__(self, in_dim, hidden_dim, num_outputs=3):
        super(MeanPoolingMILREG, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_outputs),
            nn.Sigmoid(),  
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: [batch_size, num_instances, in_dim]
        
        # Pool across instances casewise (dim=1)
        pooled = torch.mean(x, dim=1)  # Result: [batch_size, in_dim]
        
        predictions = self.regressor(pooled) * 0.8  # Scale between 0 and 0.8 (according to Pearson correlation values)
        return predictions


class MeanPoolingMIL(nn.Module):
    """MLP with meanpooling"""
    
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(MeanPoolingMIL, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: [batch_size, num_instances, in_dim]
        
        # Pool across instances casewise (dim=1)
        pooled = torch.mean(x, dim=1)  # Result: [batch_size, in_dim]
        
        scores = self.classifier(pooled)
        return scores
