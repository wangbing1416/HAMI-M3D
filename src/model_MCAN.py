"""
    Partial code is copied from MCAN (Multimodal Fusion with Co-Attention Networks for Fake News Detection)
    their released code: https://github.com/wuyang45/MCAN_code, thanks for their efforts.
    (We directly use a pre-trained InceptionNetV3 to encoder the frequency-domain images.)
"""
import torchvision
from torchvision.transforms import Resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import fft, dct

from transformers import BertModel, BertTokenizer

def process_dct_img(img):
    img = img.numpy()  # size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    N = 8
    step = int(height / N)  # 28

    dct_img = np.zeros((1, N * N, step * step, 1), dtype=np.float32)  # [1,64,784,1]
    fft_img = np.zeros((1, N * N, step * step, 1))

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row + step), col:(col + step)], dtype=np.float32)
            block1 = block.reshape(-1, step * step, 1)  # [batch_size,784,1]
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]

            i += 1

    # for i in range(64):
    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real  # [batch_size,64, 784,1]

    fft_img = torch.from_numpy(fft_img).float()  # [batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250, 1])  # [batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1)  # torch.size = [64, 250]

    return new_img

class multimodal_attention(nn.Module):
    """
    dot-product attention mechanism
    """

    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(1, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(attention).squeeze(-1)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class PositionalWiseFeedForward(nn.Module):
    """
    Fully-connected network
    """

    def __init__(self, model_dim=256, ffn_dim=2048, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        output = x
        return output


class multimodal_fusion_layer(nn.Module):
    """
    A layer of fusing features
    """

    def __init__(self, model_dim=256, num_heads=8, ffn_dim=2048, dropout=0.5):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):
        output_1 = self.attention_1(image_output, text_output, text_output, attn_mask)

        output_2 = self.attention_2(text_output, image_output, image_output, attn_mask)

        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)

        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)

        return output


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args, shared_dim=128, sim_dim=64):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19
        # bert
        if self.args.dataset == 'weibo':
            bert_model = BertModel.from_pretrained('../../huggingface/bert-base-chinese')
        else:
            bert_model = BertModel.from_pretrained('../../huggingface/bert-base-uncased')

        self.bert_hidden_size = args.bert_hidden_dim
        self.shared_text_linear = nn.Sequential(
            nn.Linear(self.bert_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

        # TODO: whether bert need fine-tuning
        self.bertModel = bert_model.requires_grad_(False)
        for name, param in self.bertModel.named_parameters():
            if name.startswith("encoder.layer.11") or \
                    name.startswith("encoder.layer.10") or \
                    name.startswith("encoder.layer.9"):
                param.requires_grad = True

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        resnet = torchvision.models.resnet34(pretrained=True)
        num_ftrs = resnet.fc.out_features
        self.visualmodal = resnet
        self.shared_image = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

        self.dct_img = torchvision.models.inception_v3(pretrained=True)
        self.shared_dct = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

         # fusion
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.dct_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(model_dim=sim_dim, num_heads=8, ffn_dim=2048, dropout=args.dropout)
            for _ in range(1)
        ])

        self.sim_classifier = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, 2)
        )

        self.resize = Resize([299, 299])

    def forward(self, text, image, mask):
        # IMAGE
        image_resnet = self.visualmodal(image)  # [N, 512]
        image_z = self.shared_image(image_resnet)
        image_z = self.image_aligner(image_z)
        # dct
        image_dct = self.resize(image)
        image_inception = self.dct_img(image_dct)[0]
        if len(image_inception.shape) == 1: image_inception = image_inception.unsqueeze(0)
        image_dct_z = self.shared_dct(image_inception)
        image_dct_z = self.dct_aligner(image_dct_z)
        # Text
        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        text_z = self.shared_text_linear(last_hidden_state)
        text_z = self.text_aligner(text_z)

        for fusion_layer in self.fusion_layers:
            output = fusion_layer(image_z, image_dct_z, attn_mask=None)
        for fusion_layer in self.fusion_layers:
            output = fusion_layer(output, text_z, attn_mask=None)

        # Fake or real
        class_output = self.sim_classifier(output)
        class_output = self.dropout(class_output)
        return class_output, output, image_z, text_z
