Download glove.42B.300d.zip from [here](https://nlp.stanford.edu/projects/glove/) and unzip. Then
run embeddings.ipynb and pca_project.ipynb.

You can use something like the below code to set the weights of a nn.Embedding layer where 'glove'
is the PCA-projected, reduced vocabulary word embedding matrix.

```python
def set_embedding_weights(self, glove):
    for i, word in enumerate(self.vocab.word2idx):
        weight = glove[word]
        self.embed.weight[i].data = torch.from_numpy(weight)
```
