---
layout: single
title:  Pooling in Graph Neural Networks (3/3)
nav_order: 3
---

By [Filippo Maria Bianchi](https://sites.google.com/view/filippombianchi/home)

This post is divided in three parts:

<p class="btn-group">
  <a href="{{ '/gnn-pool-1/' | relative_url }}"
     class="btn{% if page.url == '/gnn-pool-1/' %} btn--primary{% else %} btn--light{% endif %}"
     style="margin-right:0.6rem;{% unless page.url == '/gnn-pool-1/' %}border:1px solid #000;{% endunless %}">
     🎱 Part&nbsp;1
  </a>

  <a href="{{ '/gnn-pool-2/' | relative_url }}"
     class="btn{% if page.url == '/gnn-pool-2/' %} btn--primary{% else %} btn--light{% endif %}"
     style="margin-right:0.6rem;{% unless page.url == '/gnn-pool-2/' %}border:1px solid #000;{% endunless %}">
     🎱 Part&nbsp;2
  </a>

  <a href="{{ '/gnn-pool-3/' | relative_url }}"
     class="btn{% if page.url == '/gnn-pool-3/' %} btn--primary{% else %} btn--light{% endif %}"
     style="{% unless page.url == '/gnn-pool-3/' %}border:1px solid #000;{% endunless %}">
     🎱 Part&nbsp;3
  </a>
</p>

---

In the previous part, I introduced the different families of graph pooling operators, highlighting their respective strengths and weaknesses. In this part, I’ll discuss different approaches to evaluate their performance.

# Part 3

## ❼ Evaluation procedures

At this point, we saw some of the main approaches to perform graph pooling and noticed that they are quite different in how they work and in the result they produce. 

So, the next big question is: 

> Which pooling method should we use?

Answering this question is not trivial. In some cases, the nature of the data and the downstream task at hand can help us to take a more informed decision about which approach should be used. In other cases, however, the only way to find out is to give it a shot and see how a given method performs empirically.  Which brings us to the next fundamental question:

> How to measure the performance of a pooling operator?

Once again, this is not an easy question to answer. There exist different methods, but they measure different quantities and types of performance. Understanding them is important to realize what is going on and to make informed decisions.

Let’s see these approaches in detail.

> **Note**  
> Which pooler to choose usually depends on the data, the resources, and the task at hand.
{: .notice--primary}

### Raw performance on the downstream task

Clearly, the most straightforward approach is to equip a GNN with different pooling methods and see how it performs on a downstream task such as graph classification, graph regression, node classification, and so on. By looking, for example, at the different classification accuracies achieved, one could rank and select the pooling operators. Since this is intuitive and straightforward, this type of evaluation is the most popular in the literature. Thanks to their flexibility and capability to adapt well to the data and the task at hand, soft clustering methods usually achieve the best performance on several downstream task[^bench]. So, if memory and computational constraints are not an issues, equipping a GNN with these pooling methods is usually a good idea.

However, evaluating a pooling operator directly on the downstream task is very empirical and, most importantly, indirect. In some cases it is difficult to disentangle the effect of the pooling operator from the other GNN components, its capacity, the training scheme, and so on. In fact, it would be perfectly reasonable to ask ourselves if the same pooling operator would have performed better if inserted within a different GNN architecture or used in a different downstream task or with different data. Also, if one method is performing good or bad we might want to understand better what is going on and how to improve the design of our model.

To answer these questions, let us consider two additional performance measures presented in the original SRC paper[^src].

### Preserving node information

The first approach focuses on evaluating how well the information contained in the vertices of the original graph is preserved in the pooled graph. Ideally, we would like a pooling operator to embed as much as possible of the original information in the pooled graph. If that is the case, most of the original information can be restored from it.

To measure this property, we can use an AutoEncoder architecture that tries to reconstruct the original node features $\mathbf{X}$ from those of the pooled graph $\mathbf{X}'$. Let $\tilde{\mathbf{X}}$ be the reconstructed features, we train the graph Autoencoder by minimizing the following reconstruction loss

$$
\mathcal{L}_\text{rec} = \| \mathbf{X} - \tilde{\mathbf{X}} \|^2
$$

<figure class="align-center" style="max-width:1500px; width:80%; margin:0 auto;">
  <img src="{{ '/assets/figs/pooling/3/preserve_nodes.png' | relative_url }}"
       style="width:100%; height:auto;">
  <figcaption>GNN architecture to evaluate the ability of a pooling method to preserve information from the original graph. The features of the original and reconstructed graph should be as similar as possible.</figcaption>
</figure>

To see how this work in practice with different pooling operators let’s consider a set of point cloud graphs, where the node features are the coordinates of each point. If a pooling method manages to preserve most of the information in the pooled graph, the reconstructed point cloud should look similar to the original one.

<figure class="align-center" style="max-width:1500px; width:100%; margin:0 auto;">
  <img src="{{ '/assets/figs/pooling/3/ae.png' | relative_url }}"
       style="width:100%; height:auto;">
  <figcaption>Original and reconstructed point cloud graphs when using six different pooling operators.</figcaption>
</figure>

From this example we see that one-over-*K* methods such as NDP[^ndp] performs particularly well. On the other hand, we clearly see the main limitation of score-based method such as the vanilla top-*K* pooler[^unet] and SAGPool[^sag] that, by keeping only nodes from the same part of the graph, they fail to reconstruct the other sections.

### Preserving topology

Next, we might be interested in seeing how much the topology of the pooled graph resembles the topology of the original graph. Since the two graphs have different sizes, comparing directly the elements of the adjacency matrices $\mathbf{A}$ and $\mathbf{A}'$ is not possible. However, it is possible to compare the spectrum (*i.e.*, the eigenvalues) of their Laplacians, which should be as similar as possible. To encourage the spectra of the original and pooled graph to become similar, we can train a GNN to maximize the *spectral similarity* between the original and the pooled Laplacians[^similarity]. The spectral similarity is a measure that compares the first $D$ eigenvalues of the two Laplacians $\mathbf{L}$ and  $\mathbf{L}'$ and a loss that maximizes it can be defined as follows:

$$
\mathcal{L}_\text{struct} = \sum_{i=0}^D \|\mathbf{X}_{:,i}^\top   \mathbf{L} \mathbf{X}_{:,i} -  \mathbf{X}_{:,i}^{'\top} \mathbf{L}'  \mathbf{X}'_{:,i}\|^2
$$

where $\mathbf{X}_ {:,i}$ and  $\mathbf{X}'_{:,i}$ is the $i$-th eigenvector of $\mathbf{L}$ and $\mathbf{L}'$, respectively. The GNN architecture used in this task is very simple and consists just of a stack of MP layers followed by a pooling layer.

<figure class="align-center" style="max-width:1500px; width:60%; margin:0 auto;">
  <img src="{{ '/assets/figs/pooling/3/struct_preserve.png' | relative_url }}"
       style="width:100%; height:auto;">
  <figcaption>GNN architecture used to evaluate the ability of a pooling method to preserve in the pooled graph the same structure of the original graph.</figcaption>
</figure>

Once the GNN is trained trained, we can compare the original and the pooled graphs and the adjacency matrices $\mathbf{A}$ and $\mathbf{A}'$ to see how similar they are.

<figure class="align-center" style="max-width:1500px; width:100%; margin:0 auto;">
  <img src="{{ '/assets/figs/pooling/3/spectral.png' | relative_url }}"
       style="width:100%; height:auto;">
  <figcaption>Adjacency matrix and structure of the original and pooled graph when using different pooling operators.</figcaption>
</figure>

The examples in the figure show that soft clustering methods produce pooled graphs that are very dense and, even if informative, they have a structure that do not resemble at all the one of the original graph.  On the other hand, since one-over-*K* methods perform a regular subsampling of the graph topology the structure of the pooled graph closely resemble that of the original graph.  Finally, a regular structure like the grid is very similar to any of its subparts, meaning that the pooled graph obtained by score-based methods has a spectrum very similar to the original graph. Note that this is generally not true in graphs with less regular structures, where the overall spectrum is different from the one of the subgraphs.

### Expressiveness

All the performance measures we saw so far are rather empirical and provide a quantitative result. There is, however, a theoretical result that allows us to measure the performance of a pooling method from a qualitative perspective in terms of its expressive power [^expr]. The result can be summarized as follows.

> Given two graphs $\mathcal{G}_ 1$ and $\mathcal{G}_ 2$ that are different and that can be distinguished by a test for isomorphism called “*WL test*”, a pooling operator is expressive if the pooled graphs $\mathcal{G}_ {1_ P}$ and $\mathcal{G}_ {2_ P}$ can still be distinguished.

The conditions for expressiveness are three and are relatively easy to check.

1. The GNN part before the pooling layer must be expressive itself. To be sure that this is the case, it is enough to use expressive MP layer such as GIN[^gin]. This ensures that the node features $\mathbf{X}^L$ and $\mathbf{Y}^L$ associated with the two graphs $\mathcal{G}_1^L$ and $\mathcal{G}_2^L$ after applying $L$ MP layers are different, *i.e.*, $\sum_i^N \mathbf{x}_i^L \neq \sum_i^M \mathbf{y}_i^L$ , where $N$ and $M$ are the number of vertices in the two graphs.
2.  The $\texttt{SEL}$ operation must be such that all nodes in the original graph are included in at least one supernode of the pooled graph. This condition can be conveniently checked by considering an association matrix $\mathbf{S} \in \mathbb{R}^{N \times K}$, analog to the one we saw in the soft clustering methods, that maps each node of the original graph to the nodes of the pooled graph. If the elements in each row of $\mathbf{S}$  sum to a constant value, *i.e.*, if $\sum_{j=1}^K s_{ij} = \lambda$, then each node from the original graph is represented in the pooled graph. 
3. The $\texttt{RED}$ operation must ensure that the features of the pooled nodes  $\mathbf{X}_P$ (or $\mathbf{X}'$, as we called them before), are a weighted combination of the original features and the weights should be given by the membership values in $\mathbf{S}$ . This happens when $\mathbf{X}' = \mathbf{S}^\top \mathbf{X}$.

<figure class="align-center" style="max-width:1500px; width:100%; margin:0 auto;">
  <img src="{{ '/assets/figs/pooling/3/expr.png' | relative_url }}"
       style="width:100%; height:auto;">
  <figcaption>A GNN with expressive MP layers (condition 1) computes different features for graphs that are WL-distinguishable. A pooling layer satisfying the conditions 2 and 3 generates coarsened graphs  that are still distinguishable.</figcaption>
</figure>

If we look back at the definition of soft clustering method, we can easily see that they all satisfy condition 2 and 3 meaning that they are expressive. On the other hand, in score-based methods the rows of $\mathbf{S}$  associated with the dropped nodes are zero, because they are not assigned to any supernode. As such, score-based poolers are not expressive, meaning that it might be no longer possible to distinguish their pooled graphs even if the original graphs were different. For one-over-*K* methods the situation is a bit more complicated. Some methods like NDP are not expressive because they drop the nodes of one side of the MAXCUT partition. Other methods like *k*-MIS are expressive because they assign all nodes to those in the maximal independent set.

An important remark is that the three conditions above are *sufficient but not necessary.* This means that an expressive pooling operator always creates pooled graph that are distinguishable if the original ones were distinguishable. However, there could also be a non-expressive pooling operator that produces distinguishable pooled graphs.


## ❽ Concluding remarks

Let’s recap what we've covered in these posts.

[In Part 1]({{ '/gnn-pool-1/' | relative_url }}), we introduced Graph Neural Networks (GNNs) and graph pooling through an analogy with the more intuitive—and arguably simpler—Convolutional Neural Networks (CNNs) used in computer vision. Then, we presented a general framework called **Select-Reduce-Connect (SRC)**, which provides a formal definition of graph pooling and unifies many different pooling approaches under one umbrella.

[In Part 2]({{ '/gnn-pool-2/' | relative_url }}), we explored three different families of graph pooling methods: **soft clustering**, **one-over-*K***, and **score-based approaches**. We reviewed key representatives of each family and demonstrated how they can be expressed within the SRC framework. For each family, we highlighted their main pros and cons.

[In Part 3]({{ '/gnn-pool-3/' | relative_url }}), we examined how to evaluate a pooling operator using three quantitative measures: its **performance on downstream tasks**, its ability to **preserve information** from the original graph, and the **similarity between the spectra** of the original and pooled graphs. We also discussed a theoretical measure that characterizes a pooling operator qualitatively based on its **expressiveness**—that is, its ability to preserve the distinguishability of the original graphs in the pooled graphs.

### Graph pooling software library

While libraries such as [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and [Spektral](https://graphneural.network/) come with several pooling operators already implemented, they lack more recent approaches and do not provide a unified API for the different pooling operators.

Here enters [Torch Geometric Pool](https://torch-geometric-pool.readthedocs.io/en/latest/) (🎱 tgp), the first library explicitly made for pooling in GNNs.

<figure class="align-center" style="max-width:1500px; width:45%; margin:0 auto;">
  <img src="https://raw.githubusercontent.com/tgp-team/torch-geometric-pool/main/docs/source/_static/img/tgp-logo-bar.svg"
       style="width:100%; height:auto;">
</figure>

Explicitly designed for PyTorch Geometric, 🎱 tgp implements every pooling operator according to the SRC framework. 
Each pooling layer in 🎱 tgp can be just dropped into a GNN architecture implemented in PyTorch Geometric, without making any major change to the original architecture or to the training procedure, making it very easy to build a GNN with hierarchical pooling. Check out the [tutorials page](https://torch-geometric-pool.readthedocs.io/en/latest/tutorials/index.html) to learn how to use 🎱 tgp at its best, how to quickly deploy existing pooling operators, and how to design new ones!


### Open challenges

I hope this introduction has shed some light on this fascinating research area. There are many other graph pooling approaches that I did not cover in this post, and they are out there for you to discover. Given the variety and diversity of existing methods and the overall complexity of graph pooling, there are still many open possibilities for crafting new and more efficient pooling operators that can overcome some limitations of the current ones.

Key open research areas include improving the scalability of soft clustering methods and enabling them to learn pooled graphs of varying sizes. Additionally, there's the challenge of eliminating the top-*K* selection in score-based methods—which is not differentiable—while preserving their scalability, and selecting supernodes in a more uniform manner, similar to one-over-*K* methods. Interestingly, the different families of pooling operators are complementary in terms of strengths and weaknesses, suggesting that the next generation of pooling operators could combine their strengths to harness the best of all worlds.

I hope you enjoyed this post and that I have piqued your interest in graph pooling.

---

**📝 Citation**

If you found this useful and want to cite it in your research, you can use the following bibtex.

```bibtex
@misc{bianchi2024intropool
  author = {Filippo Maria Bianchi},
  title = {An introduction to pooling in GNNs},
  year = {2025},
  howpublished = {\url{https://filippomb.github.io/blogs/gnn-pool-1/}}
}
```

## 📚 References

[^unet]:[Gao H. & Jin S., “Graph U-Nets”, 2019.](https://arxiv.org/abs/1905.05178)

[^imgunet]:[Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015](https://arxiv.org/abs/1505.04597)

[^diffusion]:[Ho J. et al., "Denoising Diffusion Probabilistic Models", 2020](https://arxiv.org/abs/2006.11239)

[^banks]:[Tremblay N. et al., "Design of graph filters and filterbanks", 2017](https://arxiv.org/abs/1711.02046)

[^fourier]:[Ricaud B. et al., "Fourier could be a data scientist: From graph Fourier transform to signal processing on graphs", 2019](https://comptes-rendus.academie-sciences.fr/physique/articles/10.1016/j.crhy.2019.08.003/)

[^coarsening]:[Jin Y. et al., "Graph Coarsening with Preserved Spectral Properties", 2020](https://proceedings.mlr.press/v108/jin20a/jin20a.pdf)

[^src]:[Grattarola D., et al., “Understanding Pooling in Graph Neural Networks”, 2022](https://arxiv.org/abs/2110.05292).

[^diffpool]: [Ying, Z., et al., “Hierarchical graph representation learning with differentiable pooling”, 2018.](https://arxiv.org/abs/1806.08804)

[^mincut]: [Bianchi F. M., et al., “Spectral clustering with Graph Neural Networks for Graph Pooling”, 2020.](https://arxiv.org/abs/1907.00481)

[^dmon]: [Tsitsulin A., et al. “Graph clustering with graph neural networks”, 2023.](https://arxiv.org/abs/2006.16904)

[^tvgnn]: [Hansen J. B. & F. M. Bianchi, “Total Variation Graph Neural Networks”, 2023.](https://arxiv.org/abs/2211.06218)

[^graclus]: [Dhillon I. S., et al., “Weighted graph cuts without eigenvectors a multilevel approach”, 2018.](https://ieeexplore.ieee.org/abstract/document/4302760?casa_token=PNPdlsxCZ0kAAAAA:oXfXh72pXBlxVfeBqsKfqbKnDFcyG74CswZZQ5peFEli3djsjrEuyE6SdX3tPXZMLlbzKjb6bmc)

[^ndp]: [Bianchi F. M., et al., “Hierarchical representation learning in graph neural networks with node decimation pooling”, 2020.](https://arxiv.org/abs/1910.11436)

[^kron]: [Dorfler F. & Bullo F., “Kron Reduction of Graphs with Applications to Electrical Networks”, 2011.](https://arxiv.org/abs/1102.2950)

[^kmis]:[Bacciu D., et al, “Generalizing Downsampling from Regular Data to Graphs”, 2022.](https://arxiv.org/abs/2208.03523)

[^mis]: [Wu C., et al., “From Maximum Cut to Maximum Independent Set”, 2024.](https://arxiv.org/abs/2408.06758)

[^bench]:[Wang P., et al. “A Comprehensive Graph Pooling Benchmark: Effectiveness, Robustness and Generalizability”. 2024.](https://arxiv.org/abs/2406.09031)

[^sag]:[Lee J., et al., “Self-Attention Graph Pooling”, 2019.](https://arxiv.org/abs/1904.08082)

[^similarity]:[Loukas A., “Graph reduction with spectral and cut guarantees”, 2019.](https://www.jmlr.org/papers/v20/18-680.html)

[^expr]:[Bianchi F. M. & Lachi V., “The expressive power of pooling in Graph Neural Networks”, 2023.](https://arxiv.org/abs/2304.01575)

[^gin]:[Xu K. et al., “How Powerful are Graph Neural Networks?”, 2019.](https://arxiv.org/abs/1810.00826)