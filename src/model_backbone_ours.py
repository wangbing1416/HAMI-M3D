import torchvision
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args, shared_dim=128, sim_dim=64):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num
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

        # TEXT
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

        self.fusion = nn.Sequential(
            nn.Linear(sim_dim * 2, sim_dim * 2),
            nn.BatchNorm1d(sim_dim * 2),
            nn.ReLU(),
            nn.Linear(sim_dim * 2, sim_dim),
            nn.ReLU()
        )

        self.sim_classifier = nn.Sequential(
            nn.Linear(sim_dim * 5, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, 2)
        )

        # OURS
        self.forgery_extractor = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, sim_dim),
            nn.ReLU()
        )
        self.forgery_classifier = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, 2),
            nn.ReLU()
        )
        self.intent_extractor = nn.Sequential(
            nn.Linear(sim_dim * 3, sim_dim * 3),
            nn.BatchNorm1d(sim_dim * 3),
            nn.ReLU(),
            nn.Linear(sim_dim * 3, sim_dim),
            nn.ReLU()
        )
        self.intent_classifier = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, 2),
            nn.ReLU()
        )

    def forward(self, text, image, mask):
        # IMAGE
        image = self.visualmodal(image)  # [N, 512]
        image_z = self.shared_image(image)
        image_z = self.image_aligner(image_z)
        # TEXT
        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        text_z = self.shared_text_linear(last_hidden_state)
        text_z = self.text_aligner(text_z)
        # OURS
        forgery_z = self.forgery_extractor(image_z)
        forgery_pred = self.forgery_classifier(forgery_z)
        text_image_forgery = torch.cat((text_z, image_z, forgery_z), 1)
        intent_z = self.intent_extractor(text_image_forgery)
        intent_pred = self.intent_classifier(intent_z)
        # FUSION
        text_image = torch.cat((text_z, image_z), 1)
        text_image = self.fusion(text_image)
        text_image_fusion = torch.cat((text_z, text_image, image_z), dim=1)
        final = torch.cat((text_image_fusion, forgery_z, intent_z), dim=1)
        # CLASSIFICATION
        class_output = self.sim_classifier(final)
        class_output = self.dropout(class_output)
        return class_output, final, image_z, text_z, forgery_pred, intent_pred, forgery_z, intent_z
