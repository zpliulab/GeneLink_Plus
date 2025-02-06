import torch
import torch.nn as nn
import torch.nn.functional as F

class GENELink(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, hidden4_dim, output_dim, num_head1, num_head2, num_head3, alpha, device, type, reduction):
        super(GENELink, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.num_head3 = num_head3
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
            self.hidden3_dim = hidden3_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1 * hidden1_dim
            self.hidden2_dim = num_head2 * hidden2_dim
            self.hidden3_dim = num_head3 * hidden3_dim

        self.ConvLayer1 = [AttentionLayer(input_dim, hidden1_dim, alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i), attention)
        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim, hidden2_dim, alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i), attention)
        self.ConvLayer3 = [AttentionLayer(self.hidden2_dim, hidden3_dim, alpha) for _ in range(num_head3)]
        for i, attention in enumerate(self.ConvLayer3):
            self.add_module('ConvLayer3_AttentionHead{}'.format(i), attention)

        self.tf_linear1 = nn.Linear(hidden3_dim, hidden4_dim)
        self.target_linear1 = nn.Linear(hidden3_dim, hidden4_dim)

        self.tf_linear2 = nn.Linear(hidden4_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden4_dim, output_dim)

        if self.type == 'MLP':
            self.linear1 = nn.Linear(2 * output_dim, 16)
            self.linear2 = nn.Linear(16, 1)
        elif self.type == 'b_dot':
            # Add a learnable weight matrix for biased dot product
            self.bias_weight = nn.Parameter(torch.FloatTensor(output_dim, output_dim))
            nn.init.xavier_uniform_(self.bias_weight.data, gain=1.414)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()

        for attention in self.ConvLayer2:
            attention.reset_parameters()
        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

    def encode(self, x, adj):
        if self.reduction == 'concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1)
            x = F.elu(x)
        elif self.reduction == 'mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0)
            x = F.elu(x)
        else:
            raise TypeError

        if self.reduction == 'concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer2], dim=1)
            x = F.elu(x)
        elif self.reduction == 'mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]), dim=0)
            x = F.elu(x)
        else:
            raise TypeError

        out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer3]), dim=0)

        return out

    def decode(self, tf_embed, target_embed):
        if self.type == 'dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob, dim=1).view(-1, 1)
            return prob

        elif self.type == 'cosine':
            prob = torch.cosine_similarity(tf_embed, target_embed, dim=1).view(-1, 1)
            return prob

        elif self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed], dim=1)
            prob1 = self.linear1(h)
            prob1 = F.elu(prob1)
            prob = self.linear2(prob1)
            prob = F.elu(prob)
            return prob

        elif self.type == 'b_dot':
            # Biased dot product
            prob = torch.sum(torch.matmul(tf_embed, self.bias_weight) * target_embed, dim=1).view(-1, 1)
            return prob

        else:
            raise TypeError(r'{} is not available'.format(self.type))

    def forward(self, x, adj, train_sample):
        embed = self.encode(x, adj)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01, training=self.training)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01, training=self.training)

        target_embed = self.target_linear1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01, training=self.training)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01, training=self.training)

        self.tf_output = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:, 0].long()]
        train_target = target_embed[train_sample[:, 1].long()]

        pred = self.decode(train_tf, train_target)

        return pred

    def get_embedding(self):
        return self.tf_output, self.target_output



class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True, add_residual=True):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))

        self.add_residual = add_residual
        if add_residual and input_dim != output_dim:
            self.residual_linear = nn.Linear(input_dim, output_dim)
        else:
            self.residual_linear = None

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj):
        h = torch.matmul(x, self.weight)
        h = F.leaky_relu(h, self.alpha)


        e = self._prepare_attentional_mechanism_input(h)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=0.5, training=self.training)

        h_prime = torch.matmul(attention, h)

        h_prime = F.normalize(h_prime, p=2, dim=1)

        if self.bias is not None:
            h_prime += self.bias

        if self.add_residual:
            if self.residual_linear is not None:
                residual = self.residual_linear(x)
            else:
                residual = x
            h_prime += residual

        return h_prime


    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T, negative_slope=self.alpha)
        return e



