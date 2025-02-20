import torch
import math
import torch.nn as nn
from pytorch_pretrained import BertModel


class DCNNCell(nn.Module):
    def __init__(
            self,
            cell_number=1,
            sent_length=7,
            conv_kernel_size=(3, 1),
            conv_input_channels=1,
            conv_output_channels=2,
            conv_stride=(1, 1),
            k_max_number=5,
            folding_kernel_size=(1, 2),
            folding_stride=(1, 1)
    ):
        super().__init__()
        self.cell_number = cell_number
        self.sent_length = sent_length
        self.conv_kernel_size = conv_kernel_size
        self.conv_input_channels = conv_input_channels
        self.conv_output_channels = conv_output_channels
        self.conv_stride = conv_stride
        self.k_max_number = k_max_number
        self.folding_kernel_size = folding_kernel_size
        self.folding_stride = folding_stride
        self.pad_0_direction = math.ceil(self.conv_kernel_size[0] - 1)
        self.pad_1_direction = math.ceil(self.conv_kernel_size[1] - 1)

        self.conv_layer = nn.Conv2d(
            in_channels=self.conv_input_channels,
            out_channels=self.conv_output_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=(self.pad_0_direction, self.pad_1_direction)
        )

        if cell_number == -1:
            self.fold = nn.AvgPool2d(kernel_size=self.folding_kernel_size, stride=self.folding_stride)

    def forward(self, inp):
        conved = self.conv_layer(inp)

        if self.cell_number == -1:
            conved = self.fold(conved)
        k_maxed = torch.tanh(torch.topk(conved, self.k_max_number, dim=2, largest=True)[0])
        return k_maxed


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class DCNN_SST(nn.Module):
    def __init__(
            self, parameter_dict, config
    ):
        super().__init__()
        self.parameter_dict = parameter_dict

        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True


        self.dcnn_first_cell = DCNNCell(
            cell_number=-1,
            sent_length=self.parameter_dict["cell_one_parameter_dict"]["sent_length"],
            conv_kernel_size=self.parameter_dict["cell_one_parameter_dict"]["conv_kernel_size"],
            conv_input_channels=self.parameter_dict["cell_one_parameter_dict"]["conv_input_channels"],
            conv_output_channels=self.parameter_dict["cell_one_parameter_dict"]["conv_output_channels"],
            conv_stride=self.parameter_dict["cell_one_parameter_dict"]["conv_stride"],
            k_max_number=self.parameter_dict["cell_one_parameter_dict"]["k_max_number"],
            folding_kernel_size=self.parameter_dict["cell_one_parameter_dict"]["folding_kernel_size"],
            folding_stride=self.parameter_dict["cell_one_parameter_dict"]["folding_stride"],
        )
        self.dcnn_last_cell = DCNNCell(
            cell_number=-1,
            sent_length=self.parameter_dict["cell_two_parameter_dict"]["sent_length"],
            conv_kernel_size=self.parameter_dict["cell_two_parameter_dict"]["conv_kernel_size"],
            conv_input_channels=self.parameter_dict["cell_two_parameter_dict"]["conv_input_channels"],
            conv_output_channels=self.parameter_dict["cell_two_parameter_dict"]["conv_output_channels"],
            conv_stride=self.parameter_dict["cell_two_parameter_dict"]["conv_stride"],
            k_max_number=self.parameter_dict["cell_two_parameter_dict"]["k_max_number"],
            folding_kernel_size=self.parameter_dict["cell_two_parameter_dict"]["folding_kernel_size"],
            folding_stride=self.parameter_dict["cell_two_parameter_dict"]["folding_stride"],
        )
        self.fc_layer_input = self.parameter_dict["cell_two_parameter_dict"]["k_max_number"] * \
                              self.parameter_dict["cell_two_parameter_dict"]["conv_output_channels"] * \
                              math.floor(self.parameter_dict["embedding_dim"] / 4)

        self.dropout = nn.Dropout(self.parameter_dict["dropout_rate"])
        self.flatten = Flatten()
        self.fc = nn.Linear(self.fc_layer_input, self.parameter_dict["output_dim"])

    def forward(self, inp):
        context = inp[0]
        mask = inp[2]
        embedded, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        embedded = embedded.unsqueeze(1)
        out = self.dcnn_first_cell(embedded)
        out = self.dcnn_last_cell(out)
        out = self.dropout(self.flatten(out))
        out = self.fc(out)
        return out


class DCNN_TREC(nn.Module):
    def __init__(
            self,
            parameter_dict, config
    ):
        super().__init__()
        self.parameter_dict = parameter_dict

        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dcnn_first_cell = DCNNCell(
            cell_number=-1,
            sent_length=self.parameter_dict["cell_one_parameter_dict"]["sent_length"],
            conv_kernel_size=self.parameter_dict["cell_one_parameter_dict"]["conv_kernel_size"],
            conv_input_channels=self.parameter_dict["cell_one_parameter_dict"]["conv_input_channels"],
            conv_output_channels=self.parameter_dict["cell_one_parameter_dict"]["conv_output_channels"],
            conv_stride=self.parameter_dict["cell_one_parameter_dict"]["conv_stride"],
            k_max_number=self.parameter_dict["cell_one_parameter_dict"]["k_max_number"],
            folding_kernel_size=self.parameter_dict["cell_one_parameter_dict"]["folding_kernel_size"],
            folding_stride=self.parameter_dict["cell_one_parameter_dict"]["folding_stride"],
        )
        self.fc_layer_input = self.parameter_dict["cell_one_parameter_dict"]["k_max_number"] * \
                              self.parameter_dict["cell_one_parameter_dict"]["conv_output_channels"] * \
                              math.floor(self.parameter_dict["embedding_dim"] / 2)

        self.dropout = nn.Dropout(self.parameter_dict["dropout_rate"])
        self.flatten = Flatten()
        self.fc = nn.Linear(self.fc_layer_input, self.parameter_dict["output_dim"])

    def forward(self, inp):
        context = inp[0]
        mask = inp[2]
        embedded, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        embedded = embedded.unsqueeze(1)
        out = self.dcnn_first_cell(embedded)
        out = self.dropout(self.flatten(out))
        out = self.fc(out)
        return out