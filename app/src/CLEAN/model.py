import torch
import torch.nn as nn


class VanillaNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype):
        super(VanillaNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1, esm_model_dim=1280):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(esm_model_dim, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)


    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
 


class BatchNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(BatchNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.bn1 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.bn2 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.bn1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.bn2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
    
 
class InstanceNorm(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(InstanceNorm, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.in1 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.in2 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.in1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.in2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
 

class MoCoEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1, esm_model_dim=1280):
        super(MoCoEncoder, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(esm_model_dim, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class MoCo(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1, esm_model_dim=1280):
        super(MoCo, self).__init__()
        self.K = 1000
        self.m = 0.999
        self.T = 0.07

        self.encoder_q = MoCoEncoder(hidden_dim, out_dim, device, dtype, drop_out=0.1, esm_model_dim=esm_model_dim)
        self.encoder_k = MoCoEncoder(hidden_dim, out_dim, device, dtype, drop_out=0.1,esm_model_dim=esm_model_dim)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(out_dim, self.K))

        self.ec_number_labels = [None] * self.K
        self.queue.cuda()
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.apply(self._init_weights)

    def _init_weights(self, module):
            """Initialize the weights"""
            if isinstance(module, nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
            
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, ec_numbers=None):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        
        if self.K % batch_size != 0:
            self.queue_ptr[0] = 0
            return
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.queue.shape[1]:
            self.queue[:, ptr : ptr + batch_size] = (keys.T)[:, :(self.queue.shape[1] - ptr)]
            self.queue[:, :ptr + batch_size - self.queue.shape[1]] = (keys.T)[:, (self.queue.shape[1] - ptr):]
            if ec_numbers is not None:
                self.ec_number_labels[ptr : ptr + batch_size] = ec_numbers[:(self.queue.shape[1] - ptr)]
                self.ec_number_labels[:ptr + batch_size - self.queue.shape[1]] = ec_numbers[(self.queue.shape[1] - ptr):]
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            if ec_numbers is not None:
                self.ec_number_labels[ptr : ptr + batch_size] = ec_numbers
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, ec_numbers=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().cuda()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, ec_numbers)

        if ec_numbers is None:
            return logits, labels
        else:
            return logits, labels, self.ec_number_labels

class WeightedMeanFeatureFuser(nn.Module):
    def __init__(self, n_modality,l_dim):
        super().__init__()
        self.weights = nn.Parameter(
            torch.full((l_dim,n_modality),1/n_modality), requires_grad=True
        )

    def forward(self, latents):
        weights = nn.functional.softmax(self.weights, dim=-1)
        weighted_latents = torch.sum(weights[None, :] * torch.stack(latents, dim=-1), dim=-1)
        return weighted_latents

class Fuser(nn.Module):

    def __init__(self, esm_model_dim, smile_embedding_size, fuse_mode):
        super().__init__()
        self.esm_model_dim = esm_model_dim
        self.smile_embedding_size = smile_embedding_size
        self.fuse_mode = fuse_mode

        if self.fuse_mode == 'weighted_mean':
            self.fuser = WeightedMeanFeatureFuser(2, esm_model_dim)
            self.proj = nn.Linear(smile_embedding_size, esm_model_dim)
        elif self.fuse_mode == 'mlp':
            self.fuser = nn.Linear(smile_embedding_size + esm_model_dim, esm_model_dim)
        elif self.fuse_mode == 'no_fuse':
            self.fuser = None

    
    def forward(self, esm_embed, smile_embed):

        if self.fuse_mode == 'no_fuse':
            return esm_embed
        

        if self.fuse_mode == 'weighted_mean':
            projected_smile_embed = self.proj(smile_embed)
            fused_feature = self.fuser([esm_embed, projected_smile_embed])
        elif self.fuse_mode == 'mlp':
            fused_feature = self.fuser(torch.cat([esm_embed, smile_embed], -1))
        else:
            raise NotImplementedError
        return fused_feature
            
        

class MoCo_with_SMILE(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1, esm_model_dim=1280, 
                 use_negative_smile=False, smile_embedding_size = 384, fuse_mode = 'weighted_mean', queue_size=1000):
        super(MoCo_with_SMILE, self).__init__()
        self.K = queue_size
        self.m = 0.999
        self.T = 0.07

        self.fuser = Fuser(esm_model_dim, smile_embedding_size, fuse_mode)
        
        self.encoder_q = MoCoEncoder(hidden_dim, out_dim, device, dtype, drop_out=0.1, esm_model_dim=esm_model_dim)
        self.encoder_k = MoCoEncoder(hidden_dim, out_dim, device, dtype, drop_out=0.1,esm_model_dim=esm_model_dim)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(out_dim, self.K))

        self.ec_number_labels = [None] * self.K
        self.queue.cuda()
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.use_negative_smile = use_negative_smile

        if use_negative_smile:
            self.negative_classifier = nn.Sequential(
                nn.Linear(esm_model_dim, 64), 
                nn.ReLU(),
                nn.Linear(64, 2)
            )

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
            """Initialize the weights"""
            if isinstance(module, nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, ec_numbers=None):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        
        if self.K % batch_size != 0:
            self.queue_ptr[0] = 0
            return
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.queue.shape[1]:
            self.queue[:, ptr : ptr + batch_size] = (keys.T)[:, :(self.queue.shape[1] - ptr)]
            self.queue[:, :ptr + batch_size - self.queue.shape[1]] = (keys.T)[:, (self.queue.shape[1] - ptr):]
            if ec_numbers is not None:
                self.ec_number_labels[ptr : ptr + batch_size] = ec_numbers[:(self.queue.shape[1] - ptr)]
                self.ec_number_labels[:ptr + batch_size - self.queue.shape[1]] = ec_numbers[(self.queue.shape[1] - ptr):]
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            if ec_numbers is not None:
                self.ec_number_labels[ptr : ptr + batch_size] = ec_numbers
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, smile, negative_smile, ec_numbers=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features

        fused_im_q = self.fuser(im_q, smile)
        fused_im_k = self.fuser(im_k, smile)
        metrics = {}

        if self.use_negative_smile:
            metrics['negative_smile_mean'] = negative_smile.mean()
            metrics['smile_mean'] = smile.mean()
            negative_im_q = self.fuser(im_q, negative_smile)

            concat_labels = torch.cat([torch.ones(negative_im_q.shape[0]), torch.zeros(im_q.shape[0])]).to(negative_im_q.device).long()
            concat_im_q = torch.cat([fused_im_q, negative_im_q], 0)
            concat_logits = self.negative_classifier(concat_im_q)
            aux_loss = torch.nn.functional.cross_entropy(concat_logits, concat_labels)
            metrics['negative_classifier_accuracy'] = (torch.argmax(concat_logits, 1) == concat_labels).float().mean(0)
            metrics['aux_loss'] = aux_loss
        else:
            aux_loss = 0
            metrics['negative_classifier_accuracy'] = 0
            metrics['aux_loss'] = aux_loss

        q = self.encoder_q(fused_im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(fused_im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().cuda()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, ec_numbers)
        if ec_numbers is None:
            return logits, labels, aux_loss, metrics, q
        else:
            return logits, labels, aux_loss, metrics, self.ec_number_labels, q
