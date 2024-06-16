"""
    Partial code is copied from BMR (Bootstrapping Multi-view Representations for Fake News Detection)
    their released code: https://github.com/yingqichao/fnd-bootstrap, thanks for their efforts.
"""
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
from timm.models.vision_transformer import Block
from transformers import BertModel, BertTokenizer


class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs)
        scores = scores.repeat(1, inputs.shape[1])
        outputs = scores * inputs
        return outputs, scores


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args, shared_dim=64, sim_dim=64, num_expert=3):
        super(CNN_Fusion, self).__init__()
        self.args = args
        self.num_expert = num_expert
        self.depth = 1
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

         # fusion
        self.text_attention = TokenAttention(sim_dim)
        self.image_attention = TokenAttention(sim_dim)
        self.mm_attention = TokenAttention(sim_dim * 2)
        self.final_attention = TokenAttention(sim_dim * 7)

        self.image_gate_mae = nn.Sequential(nn.Linear(sim_dim, sim_dim),
                                            nn.BatchNorm1d(sim_dim),
                                            nn.Linear(sim_dim, num_expert),
                                            )
        self.text_gate = nn.Sequential(nn.Linear(sim_dim, sim_dim),
                                       nn.BatchNorm1d(sim_dim),
                                       nn.Linear(sim_dim, num_expert),
                                       )
        self.mm_gate = nn.Sequential(nn.Linear(sim_dim * 2, sim_dim),
                                     nn.BatchNorm1d(sim_dim),
                                     nn.Linear(sim_dim, num_expert),
                                     )

        image_expert_list, text_expert_list, mm_expert_list = [], [], []
        for i in range(self.num_expert):
            image_expert = []
            for j in range(self.depth):
                image_expert.append(Block(dim=sim_dim, num_heads=8))  # note: need to output model[:,0]
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)

        for i in range(self.num_expert):
            text_expert = []
            mm_expert = []
            for j in range(self.depth):
                text_expert.append(Block(dim=sim_dim, num_heads=8))  # Block(dim=sim_dim, num_heads=8)
                mm_expert.append(Block(dim=sim_dim * 2, num_heads=8))
            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
            mm_expert = nn.ModuleList(mm_expert)
            mm_expert_list.append(mm_expert)

        self.image_experts = nn.ModuleList(image_expert_list)
        self.text_experts = nn.ModuleList(text_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)
        out_dim = 1
        self.aux_trim = nn.Sequential(
            nn.Linear(sim_dim * 2, 128),
            nn.BatchNorm1d(128),
        )
        self.aux_classifier = nn.Sequential(
            nn.Linear(128, out_dim),
        )

        self.text_trim = nn.Sequential(
            nn.Linear(sim_dim, 128),
            nn.BatchNorm1d(128),
        )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(128, out_dim),
        )

        self.image_trim = nn.Sequential(
            nn.Linear(sim_dim, 128),
            nn.BatchNorm1d(128),
        )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(128, out_dim),
        )

        self.vgg_trim = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.BatchNorm1d(128),
        )
        self.vgg_alone_classifier = nn.Sequential(
            nn.Linear(128, out_dim),
        )

        self.mapping_IS_MLP = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
        )
        self.mapping_T_MLP = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
        )
        self.mapping_IP_MLP = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
        )
        self.mapping_CC_MLP = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
        )
        self.irrelevant_tensor = nn.Parameter(torch.ones((1, sim_dim * 2)), requires_grad=True)

        self.fusion_SE_network_main_task = nn.Sequential(nn.Linear(sim_dim * 7, sim_dim),
                                                         nn.BatchNorm1d(sim_dim),
                                                         nn.Linear(sim_dim, self.num_expert),
                                                         nn.Sigmoid(),
                                                         )
        final_fusing_expert = []
        for i in range(self.num_expert):
            fusing_expert = []
            for j in range(self.depth):
                fusing_expert.append(Block(dim=sim_dim * 7, num_heads=8))
            fusing_expert = nn.ModuleList(fusing_expert)
            final_fusing_expert.append(fusing_expert)
        self.final_fusing_experts = nn.ModuleList(final_fusing_expert)

        self.mix_trim = nn.Sequential(
            nn.Linear(sim_dim * 7, 128),
            nn.BatchNorm1d(128),
        )
        self.mix_classifier = nn.Sequential(
            nn.Linear(128, 2),
        )


    def forward(self, text, image, mask):
        # IMAGE
        image = self.visualmodal(image)  # [N, 512]
        image_z = self.shared_image(image)
        image_atn_feature, _ = self.image_attention(image_z)
        # TEXT
        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        text_z = self.shared_text_linear(last_hidden_state)
        text_atn_feature, _ = self.text_attention(text_z)
        mm_atn_feature, _ = self.mm_attention(torch.cat((image_z, text_z), dim=1))

        gate_image_feature = self.image_gate_mae(image_atn_feature)
        gate_text_feature = self.text_gate(text_atn_feature)  # 64 320
        gate_mm_feature = self.mm_gate(mm_atn_feature)

        shared_image_feature, shared_image_feature_1 = 0, 0
        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            tmp_image_feature = image_z
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](tmp_image_feature.unsqueeze(1))
            shared_image_feature += (tmp_image_feature * gate_image_feature[:, i].unsqueeze(1).unsqueeze(1))
        shared_image_feature = shared_image_feature.squeeze()

        shared_text_feature, shared_text_feature_1 = 0, 0
        for i in range(self.num_expert):
            text_expert = self.text_experts[i]
            tmp_text_feature = text_z
            for j in range(self.depth):
                tmp_text_feature = text_expert[j](tmp_text_feature.unsqueeze(1))  # text_feature: 64, 170, 768
            shared_text_feature += (tmp_text_feature * gate_text_feature[:, i].unsqueeze(1).unsqueeze(1))
        shared_text_feature = shared_text_feature.squeeze()

        mm_feature = torch.cat((image_z, text_z), dim=1)
        shared_mm_feature = 0
        for i in range(self.num_expert):
            mm_expert = self.mm_experts[i]
            tmp_mm_feature = mm_feature
            for j in range(self.depth):
                tmp_mm_feature = mm_expert[j](tmp_mm_feature.unsqueeze(1))
            shared_mm_feature += (tmp_mm_feature * gate_mm_feature[:, i].unsqueeze(1).unsqueeze(1))
        shared_mm_feature = shared_mm_feature.squeeze()

        shared_mm_feature_lite = self.aux_trim(shared_mm_feature)
        aux_output = self.aux_classifier(shared_mm_feature_lite)

        vgg_feature_lite = self.vgg_trim(image_z)
        shared_image_feature_lite = self.image_trim(shared_image_feature)
        shared_text_feature_lite = self.text_trim(shared_text_feature)

        vgg_only_output = self.vgg_alone_classifier(vgg_feature_lite)
        image_only_output = self.image_alone_classifier(shared_image_feature_lite)
        text_only_output = self.text_alone_classifier(shared_text_feature_lite)

        aux_atn_score = 1 - torch.sigmoid(aux_output).clone().detach()
        image_atn_score = self.mapping_IS_MLP(torch.sigmoid(image_only_output).clone().detach())
        text_atn_score = self.mapping_T_MLP(torch.sigmoid(text_only_output).clone().detach())
        vgg_atn_score = self.mapping_IP_MLP(torch.sigmoid(vgg_only_output).clone().detach())
        irre_atn_score = self.mapping_CC_MLP(aux_atn_score.clone().detach())

        shared_image_feature = shared_image_feature * (image_atn_score)
        shared_text_feature = shared_text_feature * (text_atn_score)
        shared_mm_feature = shared_mm_feature
        vgg_feature = image_z * (vgg_atn_score)
        irr_score = torch.ones_like(shared_mm_feature) * self.irrelevant_tensor
        irrelevant_token = irr_score * (irre_atn_score)

        concat_feature_main_biased = torch.cat([shared_image_feature, shared_text_feature, shared_mm_feature, vgg_feature, irrelevant_token], dim=1)
        fusion_tempfeat_main_task, _ = self.final_attention(concat_feature_main_biased)
        gate_main_task = self.fusion_SE_network_main_task(concat_feature_main_biased)

        # this part of MoE always make results unexpected (very large loss or loss is consistently unchanged), so we remove this part
        final_feature_main_task = 0
        for i in range(self.num_expert):
            fusing_expert = self.final_fusing_experts[i]
            tmp_fusion_feature = concat_feature_main_biased
            for j in range(self.depth):
                tmp_fusion_feature = fusing_expert[j](tmp_fusion_feature.unsqueeze(1))
            final_feature_main_task += (tmp_fusion_feature * gate_main_task[:, i].unsqueeze(1).unsqueeze(1))
        final_feature_main_task = final_feature_main_task.squeeze()

        final_feature_main_task_lite = self.mix_trim(concat_feature_main_biased)
        mix_output = self.mix_classifier(final_feature_main_task_lite)

        mix_output = self.dropout(mix_output)
        return mix_output, final_feature_main_task_lite, image_z, text_z
