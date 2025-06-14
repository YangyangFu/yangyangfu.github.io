---
layout: inner
title: "Variational Inference"
date: 2025-06-14
categories: [machine-learning]
---

# Variational Inference

The goal of generative models is to learn a distribution $p(x)$ (or known as **Evidence**) over the data $x$. Variational inference is a method to approximate complex distributions by transforming them into simpler ones.


## Motivation

For training a generative mode, the goal is to maximize the likelihood of the data, which can be expressed as:

$$\log p(x)$$

Directly estimating $p(x)$ requires we have all the data in the world, which is not feasible. Based on chain rule, we can rewrite the log likelihood as:

$$\log p(x) = \log \int p(x, z) dz = \log \int p(x|z) p(z) dz$$


However, this marginal likelihood is intractable, as it requires integrating over all possible latent variables $z$.
Even if the $p(z), p(x\|z)$ are simple distributions (e.g., Gaussian), the product is often non-Gaussian and high-dimensional, making the integral difficult to compute.
The $p(x|z)$ as the decoder is usually a complex approximator, which makes the integral even harder to compute. 

To address this, we can propose a simpler parameterized distribution $q(z\|x)$ (known as **Variational Distribution**) to approximate the posterior distribution $p(z\|x)$. 
One of the goal is to find the parameters of $q(z|x)$ that minimize the difference between $q(z|x)$ and $p(z|x)$. 

## Variational Trick

Now we introduce a **variational distribution** $q(z\|x)$ to approximate the true posterior $p(z\|x)$, an easier distribution to sample from. 
We usually reparameterize $q(z\|x)$ with an encoder network.

We can then rewrite the log likelihood as:

$$\log p(x) = E_{q(z|x)}[\log p(x)]$$

This holds because: $\log p(x)$ is a constant with respect to $z$, independent of the integration variable. 

$$E_{q(z|x)}[\log p(x)] = \sum_z q(z|x) \log p(x) = \log p(x) \sum_z q(z|x) = \log p(x) \times 1 = \log p(x)$$

Now add and subtract $\log q(z\|x)$ inside the expectation:

$$\log p(x) = E_{q(z|x)}[\log \frac{p(x, z)}{q(z|x)} + \log \frac{q(z|x)}{p(z|x)}]$$

This gives:

$$ \log p(x) = E_{q(z|x)}[\log \frac{p(x, z)}{q(z|x)}] + E_{q(z|x)}[\log \frac{q(z|x)}{p(z|x)}] $$


The first term is the **variational evidence lower bound** (ELBO), 

```math
\begin{aligned}
    \mathcal{L}(q) &= E_{q(z|x)}[\log \frac{p(x, z)}{q(z|x)}]\\
    &= E_{q(z|x)}[\log \frac{p(x|z)p(z)}{q(z|x)}] \\
    &= E_{q(z|x)}[\log p(x|z)] + E_{q(z|x)}[\log \frac{p(z)}{q(z|x)}] \\
    &= E_{q(z|x)}[\log p(x|z)] - E_{q(z|x)}[\log \frac{q(z|x)}{p(z)}] \\
    &= E_{q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z))
\end{aligned}
```


The second term is the **KL divergence** between the estimated variational distribution and the true posterior distribution, which is unknown because we cannot compute $p(z\|x)$ directly.

$$ KL(q(z|x) || p(z|x)) = E_{q(z|x)}[\log \frac{q(z|x)}{p(z|x)}] $$


**Note that KL divergence is always non-negative**:

Consider the function $f(x) = -\log x$, which is convex, 
From Jensen's inequality, for a random variable $X$ and the convex function $f$, we have:
$$ E[f(X)] \geq f(E[X]) $$

Apply $X = \frac{p(z)}{q(z)}$:

$$ E_{q(z)} [\log(\frac{p(z)}{q(z)})] \leq \log E_{q(z)}[\frac{p(z)}{q(z)}] = \log \int_z q(z) \frac{p(z)}{q(z)} dz = \log 1 = 0 $$

Thus, KL divergence is always non-negative:
$$ KL(q(z) | p(z)) = E_{q(z)}[\log \frac{q(z)}{p(z)}] = - E_{q(z)} [\log \frac{p(z)}{q(z)}] \geq 0 $$

This means that **maximizing the likelihood $\log p(x)$ is equivalent to maximizing the ELBO** $\mathcal{L}(q)$, which is a lower bound of the log likelihood.
This is the **variational trick**: we can find the variational distribution $q(z|x)$ to maximize the ELBO, which in turn approximates the true posterior distribution $p(z|x)$.

$$\log(p(x)) \ge E_{q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z))$$



### How to evaluate the ELBO?

The first term in ELBO is the expected reconstruction log likelihood:
$$E_{q(z|x)}[\log p(x|z)]$$

This is an expectation over the latent variable $z \sim q(z|x)$. 
- $q(z\|x)$ is often parameterized by an encoder network, which outputs the parameters of the distribution (e.g., mean and variance for Gaussian). $p(x\|z)$ is a decoder distribution (e.g., Gaussian)
- no closed-form solution for this expectation, so we need to use sampling methods to estimate it.

(1) **Reparameterization trick**:
We can use the reparameterization trick to sample from $q(z|x)$, which allows us to backpropagate through the sampling process. 
For example, if $q(z|x)$ is a Gaussian distribution with mean $\mu(x)$ and variance $\sigma^2(x)$, we can sample $z$ as:
$$z = \mu(x) + \sigma(x) \odot \epsilon$$

where $\epsilon \sim N(0, I)$ is a standard Gaussian noise. This allows us to compute gradients with respect to the parameters of the encoder network.

(2) **Monte Carlo Approximation**:

We can approximate the expectation by sampling $N$ times from $q(z|x)$:
$$E_{q(z|x)}[\log p(x|z)] \approx \frac{1}{N} \sum_{i=1}^{N} \log p(x|z_i)$$

where $z_i \sim q(z|x)$ for $i = 1, 2, \ldots, N$.


### How to evaluate the KL divergence?
The second term in ELBO is the KL divergence:
$$KL(q(z|x) || p(z)) = E_{q(z|x)}[\log \frac{q(z|x)}{p(z)}]$$

This term can often be computed in closed form, depending on the choice of $q(z|x)$ and $p(z)$.

## Variational Distribution

The **variational distribution** or **encoder** $q(z|x)$ is a simpler distribution that we can sample from. The choice of its form affects how well it approximates the true posterior $p(z|x)$.

### Diagonal Gaussian
A common choice is a **diagonal Gaussian**:
$$ q(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x)) $$

where $\mu(x)$ and $\sigma^2(x)$ are functions (often neural networks) that output the mean and variance for each latent variable $z$ given the input $x$. 

This distribution leads to analytical KL with a closed-form solution:
$$ KL(\mathcal{N}(\mu, \delta^2) || \mathcal{N}(0, 1)) = \frac{1}{2} \sum_i (\mu_i^2 + \sigma_i^2 - \log(\sigma_i^2) - 1) $$

where $i$ indexes the latent dimensions.

This choice allows sampling via the reparameterization trick:
$$ z = \mu(x) + \sigma(x) \odot \epsilon $$

where $\epsilon \sim \mathcal{N}(0, I)$ is a standard normal noise vector, and $\odot$ denotes element-wise multiplication. 
This allows gradients to flow through the sampling process, enabling backpropagation during training.

### Mixture of Gaussians
Another choice is a **mixture of Gaussians**:
$$ q(z|x) = \sum_{k=1}^K \pi_k(x) \mathcal{N}(z; \mu_k(x), \sigma_k^2(x)) $$

where $\pi_k(x)$ are the mixing coefficients, $\mu_k(x)$ and $\sigma_k^2(x)$ are the means and variances for each component, and $K$ is the number of components.
This can model multimodal posteriors better than a single Gaussian.

But it has no analytical KL divergence, so we need to use Monte Carlo methods to estimate the ELBO.

### Normalizing Flows
A more flexible approach is to use **normalizing flows**, which transform a simple distribution (like a Gaussian) into a more complex one through a series of invertible transformations.

$z = f_K f_{K-1} \dots f_1(\epsilon)$, 

where $f$ is a sequence of transformations parameterized by $\theta$. 

The ELBO can be computed using the change of variables formula:
$$ \log p(x) = E_{q(z|x)}[\log p(x|z) + \log \det \frac{\partial f^{-1}}{\partial z}] $$
This allows for complex posteriors while still being able to compute the ELBO. 

## Autoencoder 

An **autoencoder** is a neural network architecture that learns to encode input data into a lower-dimensional latent space and then decode it back to reconstruct the original data.

It consists of two main components:
1. **Encoder**: Maps the input data $x$ to a latent representation $z$ 
2. **Decoder**: Maps the latent representation $z$ back to the data space to reconstruct $x$.

If the encoder is not probabilistic, it simply outputs a deterministic $z = f(x)$, where $f$ is a neural network. The decoder then reconstructs the input as $\hat{x} = g(z)$, where $g$ is another neural network.
The training objective is to minimize the reconstruction error, often using mean squared error (MSE) or cross-entropy loss:
$$ \mathcal{L}_{\text{recon}}(x, \hat{x}) = || x - \hat{x} ||^2 $$

Thus this traditional autoencoders
- are not generative models, as they do not model the distribution of the data but rather learn a compressed representation.
- have no mechanism to sample from the latent space, as it does not define a distribution over $z$.
- latent space may be irrelevant or not sparse -> a random z often leads to garbage output.



## Variational Autoencoder (VAE)

To make the autoencoder generative, we introduce a probabilistic encoder that outputs a distribution over the latent space instead of a single point.
The encoder outputs parameters of a variational distribution $q(z|x)$, typically a diagonal Gaussian with mean $\mu(x)$ and variance $\sigma^2(x)$.
The decoder then samples from this distribution to generate the latent variable $z$:
$$ z \sim q(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x)) $$

The decoder then reconstructs the input from the sampled $z$:
$$ \hat{x} = g(z) $$
The training objective is to maximize the ELBO, based on the variational trick.

In practice, during training, to estimate the first term of ELBO, instead of sampling $N$ samples from $q(z|x)$, we can use the reparameterization trick to sample one single $z$:

$$ E_{q(z|x)}[\log p(x|z)] \approx \log p(x|z_1), z_1 \sim q(z|x) $$

Only one sample works because this is training. Training with one sample is like a stochastic gradient descent, which is a common practice in training neural networks.


With the assumption that $x \in \mathcal{R}^D$, the decoder $p(x|z)$ is a Gaussian distribution, the likelihood becomes:
$$ \log p(x|z) = -\frac{1}{2\delta^2} || x - g(z) ||^2 - \frac{D}{2} \log(2\pi \delta^2) $$

where $\delta$ is a hyperparameter controlling the noise level in the reconstruction, usually set to 1. 
Thus this term reduces to minimizing the reconstruction error between the input $x$ and the reconstructed output $\hat{x} = g(z)$.

### How to use VAE?
This generative model can be used in several ways:
1. **Reconstruction**: Given an input $x$, the encoder maps it to a latent representation $z$ and the decoder reconstructs $\hat{x}$. The model is trained to minimize the reconstruction error while also ensuring that the latent space follows a prior distribution (e.g., standard normal).
2. **Sampling**: We can sample from the latent space by sampling $z$ from the prior $p(z)$ (often a standard normal $\mathcal{N}(0, I)$) and then passing it through the decoder to generate new data points. This process is random but constrained by the learned latent space structure. To maxmize the ELBO is to minimize the KL divergence between the variational distribution and the prior, which push the latent space to be close to the prior distribution (e.g., normal distribution). During inference, sampling from the prior leads to meaningful samples that resemble the training data.


For generative tasks, **how to control the sampling process?**
- **Interpolation**: By sampling two points in the latent space and interpolating between them, we can generate smooth transitions between different data points.

$$ z_{\text{interp}} = \alpha z_1 + (1 - \alpha) z_2 $$

where $\alpha$ is a parameter that controls the interpolation.
- **Conditional Generation**: By conditioning the encoder on additional information (e.g., class labels), we can generate samples that belong to specific categories. This is done by modifying the encoder to take the additional information as input, allowing it to learn a conditional distribution over the latent space.
- **VQ-VAE + Transformer**: For text generation, we can use a VQ-VAE to encode text into discrete latent codes and then use a transformer to model the relationships between these codes, allowing for coherent text generation.





## Vector Quantized Variational Autoencoder (VQ-VAE)

Instead of using a continuous latent space, VQ-VAE uses a discrete latent space by quantizing the latent representations into a finite set of vectors (codebook). The architecure consists of:
1. **Encoder**: Maps the input data $x$ to a continuous latent representation $z_e = f_e(x)$.
2. **Codebook**: A set of discrete vectors (codebook), e.g., $K$, that the continuous latent representation is quantized to. The encoder output is mapped to the nearest codebook vector:
$$ z_q = \text{argmin}_{e_k \in \text{codebook}} || z_e - e_k ||^2 $$
3. **Decoder**: Maps the quantized latent representation $z_q$ back to the data space to reconstruct $x$:
$$ \hat{x} = g(z_q) $$


### Varitional Distribution
The variational distribution $q(z|x)$ is not a continuous distribution but rather a discrete set of vectors from the codebook. The encoder outputs a continuous latent representation, which is then quantized to the nearest codebook vector.
Thus the posterior approximation $q(z_q|x)$  are defined as one-hot as follows:
$$ q(z_q = e_k | x) = \begin{cases}
1 & \text{if } e_k \text{ is the nearest codebook vector for } z_e \\
0 & \text{otherwise}
\end{cases} $$
This means that the encoder does not output parameters of a distribution but rather a discrete choice of which codebook vector to use. 
We can use this distribution to bound ELBO.

The prior distribution $p(z_q)$ is a uniform distribution over the codebook vectors, as each vector is equally likely to be chosen.

### Training Objective
The training involves learning the encoder, the codebook and the decoder, which can be formulated as follows:

$$ \mathcal{L} = \mathcal{L}_{ELBO} + \mathcal{L}_{codebook} $$

where the first term is the ELBO for VAE, and the second term is a commitment loss that encourages the encoder outputs and the codebook vectors to stay close to each other.



**ELBO loss**

Because of the quantization process, the latent variable after the encoder is no longer continuous.

$$\log p(x) = \int \log p(x|z_e) p(z_e) dz_e$$

Because the decoder $p(x|z)$ is trained with $z = z_q(x)$, the decoder shouldn't allocate any probability mass to p(x|z_e)$ when $z_e$ is not in the codebook once it's converged.
Thus we can write:
$$\log p(x) \approx \sum_{z_q \in \text{codebook}} \log p(x|z_q) p(z_q) $$

To maximize the above likelihodd is to maximize the ELBo. From the derivation of ELBO, we have:
$$ \mathcal{L}_{ELBO} = E_{q(z_q|x)}[\log p(x|z_q)] - KL(q(z_q|x) || p(z_q)) $$

The first term as a decoder loss, is equivalent to minimizing the reconstruction error $||x - \hat x||_2^2$.
The second term is the KL divergence between the variational distribution and the prior distribution, which encourages the quantized latent representations to be close to the codebook vectors.
The variation distribution $q(z_q|x)$ is a categorical distribution over the codebook vectors, and the prior distribution $p(z_q)$ is a uniform distribution over the codebook vectors.
Thus the KL divergence can be computed as:
$$ KL(q(z_q|x) || p(z_q)) = \log q(z_q=k|x) - \log p(z_q=k) = \log 1 - \log\frac{1}{K} = \log K$$

which is constant. 

Then to minimize ELBO loss is simplified to minimize:
$$\mathcal{L}_{ELBO} = -E_{q(z_q|x)}[\log p(x|z_q)]$$ 


### Codebook Loss
The codebook loss encourages the encoder outputs to be close to the codebook vectors. It is defined as:
$$ \mathcal{L}_{codebook} = || z_e - e_k ||^2 $$

However, due to the moving target problem while jointly training the encoder and the codebook, this loss is broken down into two parts:

$$ \mathcal{L}_{codebook} = || sg(z_e) - e_k ||^2 + || z_e - sg(e_k) ||^2 $$

where $e_k$ is the codebook vector that is closest to $z_e$, and $sg$ is the stop gradient operation that prevents gradients from flowing through the codebook vectors.



## Importance Weighted Autoencoder (IWAE)

The above variational autoencoders (VAEs) use a single sample from the variational distribution to estimate the ELBO. However, this can lead to high variance in the gradient estimates, making training unstable and slow.
To address this, the **Importance Weighted Autoencoder (IWAE)** uses multiple samples from the variational distribution to estimate the ELBO more accurately, which also leads a tighter bound on the log likelihood.

In VAEs, we maximize the ELBO:
$$ \log p(x) \geq E_{q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z)) $$
This is a loose bound, especially when the variational distribution $q(z|x)$ is not a good approximation of the true posterior $p(z|x)$.

IWAE improves this by drawing $K$ samples for the variational distribution $q(z|x)$ and constructing a tighter bound:

$$ \log p(x) \geq E_{q(z_1, z_2, \ldots, z_K | x)}[\log \frac{p(x, z_1, z_2, \ldots, z_K)}{q(z_1, z_2, \ldots, z_K | x)}] = E_{q(z_1, z_2, \ldots, z_K | x)}[\log \frac{1}{K} \sum_{i=1}^{K} \frac{p(x, z_i)}{q(z_i | x)}] $$

where $z_i \sim q(z|x)$ for $i = 1, 2, \ldots, K$.

