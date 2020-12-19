# Friday Initial Remarks

- Deep Recommender System overview
- Infrastructure
- Initial timeline and administration


# Recommender Systems

## Basic Recommender Tasks

* point-of-interest recommendations (recommend a thing)
* Top N recommendation (ranking based on implicit feedback)
	* implicit feedback = did someone watch this movie
	* explicit feedback = thumbs up thumbs down (users may be reluctant to rate)
* If timestamps are available, sequence aware recommendation
* Recommend new items or new users

## Collaborative Filtering

* Filter information using the help of other users/agents/data sources
* Memory Based
	* Nearest Neighbor like
	* Trouble with big data
* Model based
	* Matrix factorization for example
	* Can be extended to deep learning and friends
* CF uses user interactions


## Matrix Factorization (simplest)

![alt text](https://d2l.ai/_images/rec-mf.svg)


## Matrix Factorization (simplest)

```python
import torch.nn as nn
import torch
class MatrixFactorization(nn.Module):
	def __init__(self, num_factors, num_users, num_items, **kwargs):
		super().__init__(**kwargs)
		self.P = nn.Embedding(
			num_embeddings=num_users,
			embedding_dim=num_factors
			)
		self.Q = nn.Embedding(
			num_embeddings=num_items,
			embedding_dim=num_factors
			)
		self.p_bias = nn.Embedding(num_users, 1)
		self.q_bias = nn.Embedding(num_items, 1)
	def forward(self, user_id, item_id):
		P_u = self.P(user_id)
		Q_i = self.Q(item_id)
		bias_u = self.p_bias(user_id)
		bias_i = self.q_bias(user_id)
		# double check sum
		result = (P_u * Q_i).sum(1) + torch.squeeze(bias_u) + torch.squeeze(bias_i)
		return result.flatten()
```

## Using Autoencoders

Nonlinear matrix completion!

* No longer explicitly state anything, the model is supposed to learn what it is doing and repeat however many times it needs
* [See paper](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)
* same concept, matrix with some masked ratings, try to predict


## Basic Autoencoder for matrix completion

```python
class autorec(nn.Module):
	def __init__self(num_hidden, num_users, dropout=0.05):
		self.encoder = nn.Linear(num_users, num_hidden, bias=True)
		self.decoder = nn.Linear(num_hidden, num_users, bias=True)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		hidden = self.encoder(x)
		hidden = torch.sigmoid(x)
		hidden = self.dropout(x)
		output = self.decoder(x)
		if self.training:
			return output * torch.sign(x) # makes things zero where they should be
		else:
			return output
```

## More complicated ideas: Neural Matrix Factorization

* [Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)
* Need to implement negative sampling (unobserved entries)
![](https://d2l.ai/_images/rec-neumf.svg)

## More complicated ideas: Sequence Aware
![](https://d2l.ai/_images/rec-caser.svg)

# Infrastructure

- Modeling, Logging, and Running Code
    - Lightning Demo, two different ways to do it
    - wandb yaml demo
- Saving for production
	- ONNX


## Logging

wandb[](https://app.wandb.ai)

- Email:
	- josephs.david11@gmail.com
- Password
	- nyx_projct

# Admin & Timeline


## Tonight

- Big programming push
- Get dataloaders, training and evaluation pipeline running
- Write model code, get it running overnight
- Extensive David-based assistance

## Tomorrow

- In morning, evaluate if we need to implement more complex model
- relatively quiet day, code should be running, y'all should be smoothly moving
- Start aggregating results
- Figure out how to visualize with UMAP (get in touch with noelle)
- Log everything!!
- Troubleshoot surprises
- Start reading about how to bottle up with ONNX and coordinate with partee

## Sunday

- Aggregate results
