---
layout: post
title: Triplet loss and quadruplet loss via tensor masking
share: false
tags: [triplet loss, quadruplet loss, hard mining, PyTorch, masking, tensor, representation learning, metric learning, broadcasting, image embeddings]
---

In this blog post, I show how to implement triplet loss and quadruplet loss in PyTorch via tensor masking. The idea of triplet loss is to learn meaningful representations of inputs (e.g. images) given a partition of the dataset (e.g. labels) by requiring that the distance from an anchor input to an positive input (belonging to the same class) is minimised and the distance from an anchor input to a negative input (belonging to a different class) is maximized. I'll excuse myself from explaining it in more detail here as there are [some](https://www.coursera.org/lecture/convolutional-neural-networks/triplet-loss-HuUtN) [great](https://omoindrot.github.io/triplet-loss) [sources](https://gombru.github.io/2019/04/03/ranking_loss/) elsewhere. 

[Quadruplet loss](https://arxiv.org/pdf/1704.01719.pdf) is a simple generalisation of triplet loss with a constraint involving four input: an anchor, a positive input and two negative inputs. Quadruplet loss is supposed to ensure a smaller intra-class variation and a larger inter-class variation in the embedding space, which leads to better performance in downstream tasks (e.g. person re-identification).

Both triplet loss and quadruplet loss require an efficient way of selecting triplets and quadruplets of inputs. A popular solution is *batch hard mining*: selecting the hardest triplets (quadruplets) that can be constructed from a batch of inputs. Hardness is measured in terms of (usually Euclidean) distance we are mostly interested in a (anchor, positive) pair with the highest distance between the anchor and the positive and a (anchor, negative) pair with the lowest distance between the anchor and the negative. These hardest pair will end up producing largest gradients. Gradients from easy pairs will be negligible after a few batches of training, so we would prefer to discard them to save computation time.

For the purpose of this blog post, we will assume a simple setup: we use a ResNet-18 architecture to generate embeddings for images from CIFAR10 and use triplet (quadruplet) loss to force them to capture image content. To that end we use class labels: two images from the same class (e.g. planes) are a positive class, two images from different classes (e.g. a cat and a plane) are a negative pair.

# Distance matrix and masks

Batch hard triplet (quadruplet) mining can be elegantly implemented in terms of tensor masking. While a naive implementation would involve a three (four) nested four loops over a batch of inputs, we can implement a fully tensorised solution. First, we compute a distance matrix using [two tricks](https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065):
1. Quadratic expansion of Euclidean distance: $$\vert\vert a - b\vert\vert ^2 = \vert\vert a\vert\vert ^2 - 2ab + \vert\vert b\vert\vert ^2$$
2. $$\text{diag}(XX^{\text{T}})$$ is $$\vert\vert x\vert\vert ^2$$ for each row $$x$$ in $$X$$.

```python
def get_distance_matrix(
        embeddings: torch.Tensor,  #  [B, E]
    ):
    B = embeddings.size(0)
    dot_product = embeddings @ embeddings.T  # [B, B]
    squared_norm = torch.diag(dot_product) # [B]
    distances = squared_norm.view(1, B) - 2.0 * dot_product + squared_norm.view(B, 1)  # [B, B]
    return torch.sqrt(nn.functional.relu(distances) + 1e-16)  # [B, B]
```

Numbers in square brackets indicate tensor dimensions. We will represent the hardness of each triplet (quadruplet) that can be constructed from a batch in terms of elements of a 3-dimensional (4-dimensional) tensor. To do that, we need to select legal triplets (quadruplet). I'll show how to do that explicily (mostly for debugging purposes) and then how to implement a triplet mask.

To select positive pairs, we construct a 2d mask for pairs `(i, j)` such that<br/>
`labels[i] == labels[j] and i != j`.

```python
def get_positive_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):
    B = labels.size(0)
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B]
    return labels_equal & ~indices_equal  # [B, B]
```
This function returns a `[B, B]` binary tensor indicating positions of valid positive pairs (i.e. pairs sharing a `label`) for a given batch of images.

To select negative pairs, we construct a 2d mask for pairs `(i, j)` such that<br/>
`labels[i] != labels[j] and i != j`.

```python
def get_negative_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):
    B = labels.size(0)
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B]
    return ~labels_equal & ~indices_equal  # [B, B]
```

Finally, a triplet mask is a 3d mask for triplets ``(i, j, k)`` such that<br/>
`labels[i] == labels[j] and labels[i] != labels[k] and i != j != k`.

```python
def get_triplet_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):

    B = labels.size(0)

    # Make sure that i != j != k
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B]
    indices_not_equal = ~indices_equal  # [B, B]
    i_not_equal_j = indices_not_equal.view(B, B, 1)  # [B, B, 1]
    i_not_equal_k = indices_not_equal.view(B, 1, B)  # [B, 1, B]
    j_not_equal_k = indices_not_equal.view(1, B, B)  # [1, B, B]
    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k  # [B, B, B]

    # Make sure that labels[i] == labels[j] but labels[i] != labels[k]
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    i_equal_j = labels_equal.view(B, B, 1)  # [B, B, 1]
    i_equal_k = labels_equal.view(B, 1, B)  # [B, 1, B]
    valid_labels = i_equal_j & ~i_equal_k  # [B, B, B]

    return distinct_indices & valid_labels  # [B, B, B]
```

To make sure these implementations are correct, I've written a few unit tests.

```python
def test_get_distance_matrix(device_for_tests):
    embeddings = torch.FloatTensor(
        [[1, 1], 
        [7, 7], 
        [1, 1]], 
    ).to(device=device_for_tests)
    distance_matrix = get_distance_matrix(embeddings)
    assert torch.allclose(
        torch.diag(distance_matrix), 
        torch.zeros(3, device=device_for_tests)
    )
    assert torch.allclose(distance_matrix, distance_matrix.T)
    assert distance_matrix[0, 2] < distance_matrix[0, 1]


def test_get_positive_mask(device_for_tests):
    labels = torch.LongTensor([1, 2, 3, 1])
    pos_mask = get_positive_mask(labels, device_for_tests)
    assert pos_mask[0, 3]
    assert not pos_mask[0, 1]
    assert not pos_mask[0, 0] and not pos_mask[1, 1]


def test_get_negative_mask(device_for_tests):
    labels = torch.LongTensor([1, 2, 3, 1])
    neg_mask = get_negative_mask(labels, device_for_tests)
    assert not neg_mask[0, 3]
    assert neg_mask[0, 1]
    assert not neg_mask[0, 0] and not neg_mask[1, 1]


def test_get_triplet_mask(device_for_tests):
    labels = torch.LongTensor([1, 2, 3, 1, 3])
    mask = get_triplet_mask(labels, device_for_tests)
    assert mask[0, 3, 2]
    assert mask[2, 4, 1]
    assert mask[4, 2, 0]
    assert not mask[0, 0, 0]
    assert not mask[0, 3, 3]
    assert not mask[0, 0, 4]
```

# Triplet loss


Now we can make use of our distance matrix and helper functions for generating masks while implementing the main model. We can compute triplet loss for each triplet by a simple tensor operation (making use of [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html)): `distance_matrix.view(B, B, 1) - distance_matrix.view(B, 1, B)`. The output is a 3-dimensional tensor, `triplet_loss_unmasked`, encoding hardness of each triplet `(i, j, k)` under `triplet_loss_unmasked[i, j, k]`. Similarily, we can gather valid triplets by indexing this tensor using our mask, `triplet_mask`.

```python
class TripletLossModel(nn.Module):
    
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet
        self.resnet.fc = nn.Identity()
        self.embeddings = nn.Linear(512, 128)
        
    def forward(
            self, 
            inputs: torch.Tensor,  # [B, C, H, W]
            labels: torch.Tensor  # [B]
        ):
        B = labels.size(0)
        embeddings = self.embeddings(self.resnet(inputs))  # [B, E]
        distance_matrix = get_distance_matrix(embeddings)  # [B, B]
        with torch.no_grad():
            mask_pos = get_positive_mask(labels, device)  # [B, B]
            mask_neg = get_negative_mask(labels, device)  # [B, B]
            triplet_mask = get_triplet_mask(labels, device)  # [B, B, B]
            unmasked_triplets = torch.sum(triplet_mask)  # [1]
            mu_pos = torch.mean(distance_matrix[mask_pos])  # [1]
            mu_neg = torch.mean(distance_matrix[mask_neg])  # [1]
            mu = mu_neg - mu_pos  # [1]
        
        distance_i_j = distance_matrix.view(B, B, 1)  # [B, B, 1]
        distance_i_k = distance_matrix.view(B, 1, B)  # [B, 1, B]
        triplet_loss_unmasked = distance_i_k - distance_i_j   # [B, B, B]
        triplet_loss_unmasked = triplet_loss_unmasked[triplet_mask] # [valid_triplets]
        hardest_triplets = triplet_loss_unmasked < max(mu, 0)  # [valid_triplets]
        triplet_loss = triplet_loss_unmasked[hardest_triplets]  # [valid_triplets_after_mask]
        triplet_loss = nn.functional.relu(triplet_loss)  # [valid_triplets_after_mask]

        loss = triplet_loss.mean()
        logs = {
            'positive_pairs': torch.sum(mask_pos).cpu().detach().item(),
            'negative_pairs': torch.sum(mask_neg).cpu().detach().item(),
            'mu_neg': mu_neg.cpu().detach().item(),
            'mu_pos': mu_pos.cpu().detach().item(),
            'valid_triplets': unmasked_triplets.cpu().detach().item(),
            'valid_triplets_after_mask': triplet_loss.size(0),
            'triplet_loss': triplet_loss.mean().cpu().detach().item()
        }
        return loss, logs
```

An example training loop (removing as much boilerplate as possible, including validation) could look like that:

```python
device = torch.device('cuda')
resnet18 = models.resnet18(pretrained=False)
model = TripletLossModel(resnet=resnet18)
model = model.to(device)
opt = SGD(model.parameters(), lr=0.001)
ds_train = CIFAR10('.', transform=to_tensor, download=True)
dataloader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
for e in range(10):
    for batch_idx, (input_, labels) in enumerate(dataloader_train):
        input_, labels = input_.to(device), labels.to(device)
        loss, logs = model(input_, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
```

# Quadruplet loss

Implementing quadruplet loss requires one more masking function, this time for a 4-dimensional tensor. We are looking for quadruples `(i, j, k, l)` of inputs satisfying four constraints:<br/>
1. `labels[i] == labels[j]`,
2. `labels[j] != labels[k]`,
3. `labels[k] != labels[l]`,
4. `i != j != k != l`.

```python
def get_quadruplet_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):
    B = labels.size(0)

    # Make sure that i != j != k != l
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B] 
    indices_not_equal = ~indices_equal  # [B, B] 
    i_not_equal_j = indices_not_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_not_equal_k = indices_not_equal.view(1, B, B, 1)  # [B, 1, 1, B] 
    k_not_equal_l = indices_not_equal.view(1, 1, B, B)  # [1, 1, B, B] 
    distinct_indices = i_not_equal_j & j_not_equal_k & k_not_equal_l  # [B, B, B, B] 

    # Make sure that labels[i] == labels[j] 
    #            and labels[j] != labels[k] 
    #            and labels[k] != labels[l]
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    i_equal_j = labels_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_equal_k = labels_equal.view(1, B, B, 1)  # [1, B, B, 1]
    k_equal_l = labels_equal.view(1, 1, B, B)  # [1, 1, B, B]
    
    return (i_equal_j & ~j_equal_k & ~k_equal_l) & distinct_indices  # [B, B, B, B] 
```

Again, a sanity check:
```python
def test_get_quadruplet_mask(device_for_tests):
    labels = torch.LongTensor([1, 2, 3, 1, 3])
    mask = get_quadruplet_mask(labels, device_for_tests)
    assert mask[0, 3, 1, 2]
    assert mask[2, 4, 0, 1]
    assert mask[4, 2, 1, 0]
    assert not mask[0, 0, 0, 0]
    assert not mask[0, 0, 1, 2]
    assert not mask[0, 3, 4, 4]
    assert not mask[0, 3, 2, 4]
```

A complete model using quadruplet loss shares a lot of computation with the triplet loss model as computing quadriplet loss involves computing triplet loss. To keep things simple, we don't make use of that and implement quadruplet provide an independent class, `QuadrupletLossModel`.

```python
class QuadrupletLossModel(nn.Module):
    
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet
        self.resnet.fc = nn.Identity()
        self.embeddings = nn.Linear(512, 128)
        
    def forward(
            self, 
            inputs: torch.Tensor,  # [B, C, H, W]
            labels: torch.Tensor  # [B]
        ):
        B = labels.size(0)
        embeddings = self.embeddings(self.resnet(inputs))  # [B, E]
        distance_matrix = get_distance_matrix(embeddings)  # [B, B]
        with torch.no_grad():
            mask_pos = get_positive_mask(labels, device)  # [B, B]
            mask_neg = get_negative_mask(labels, device)  # [B, B]
            triplet_mask = get_triplet_mask(labels, device)  # [B, B, B]
            quadruplet_mask = get_quadruplet_mask(labels, device)  # [B, B, B, B]
            unmasked_triplets = torch.sum(triplet_mask)  # [1]
            unmasked_quadruplets = torch.sum(quadruplet_mask)  # [1]
            mu_pos = torch.mean(distance_matrix[mask_pos])  # [1]
            mu_neg = torch.mean(distance_matrix[mask_neg])  # [1]
            mu = mu_neg - mu_pos  # [1]
        
        distance_i_j = distance_matrix.view(B, B, 1)  # [B, B, 1]
        distance_i_k = distance_matrix.view(B, 1, B)  # [B, 1, B]
        triplet_loss_unmasked = distance_i_k - distance_i_j   # [B, B, B]
        triplet_loss_unmasked = triplet_loss_unmasked[triplet_mask] # [valid_triplets]
        hardest_triplets = triplet_loss_unmasked < max(mu, 0)  # [valid_triplets]
        triplet_loss = triplet_loss_unmasked[hardest_triplets]  # [valid_triplets_after_mask]
        triplet_loss = nn.functional.relu(triplet_loss)  # [valid_triplets_after_mask]

        distance_i_j = distance_matrix.view(B, B, 1, 1)  # [B, B, 1, 1]
        distance_k_l = distance_matrix.view(1, 1, B, B)  # [1, 1, B, B]
        auxilary_loss_unmasked = distance_k_l - distance_i_j  # [B, B, B, B]
        auxilary_loss_unmasked = auxilary_loss_unmasked[quadruplet_mask]  # [valid_quadruplets]
        hardest_quadruples = auxilary_loss_unmasked < max(mu, 0)/2  # [valid_quadruplets_after_mask]
        auxilary_loss = auxilary_loss_unmasked[hardest_quadruples]  # [valid_quadruplets_after_mask]
        auxilary_loss = nn.functional.relu(auxilary_loss)  # [valid_triplets_after_mask]

        quadruplet_loss = triplet_loss.mean() + auxilary_loss.mean()
        logs = {
            'positive_pairs': torch.sum(mask_pos).cpu().detach().item(),
            'negative_pairs': torch.sum(mask_neg).cpu().detach().item(),
            'mu_neg': mu_neg.cpu().detach().item(),
            'mu_pos': mu_pos.cpu().detach().item(),
            'valid_triplets': unmasked_triplets.cpu().detach().item(),
            'valid_triplets_after_mask': triplet_loss.size(0),
            'valid_quadruplets': unmasked_quadruplets.cpu().detach().item(),
            'valid_quadruplets_after_mask': auxilary_loss.size(0),
            'auxilary_loss': auxilary_loss.mean().cpu().detach().item(),
            'triplet_loss': triplet_loss.mean().cpu().detach().item()
        }
        return quadruplet_loss, logs
```

`QuadrupletLossModel` can be plugged into the same training loop as in the snippet for training with triplet loss.

*[A Jupyter notebook](https://gist.github.com/tomekkorbak/bdea3fb841fcd390b58f2643eaaf365b) accompanying this blog post is available on GitHub.*