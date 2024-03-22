# Conditional Variational Autoencoders for MNIST Digit Generation
The goal of this project is to explore the key differences between autoencoders
and variational autoencoders (VAEs), and how these differences impact image
generation, even in simple applications like handwritten digit generation. A
step-by-step notebook can be found here, if you would like to follow along or
reproduce the figures and results.

## Introduction
Suppose we have a set of images $X = \\{x_1,...,x_n\\}$ (e.g. handwritten digits) that were drawn 
from some underlying probability distribution $p(x)$. Without knowing exactly
how samples are generated by this unknown real-world process, can we somehow
model such a process by learning some parameters $θ$ of a neural network? Yes. In fact,
generative models of the form $p\_{θ}(x)$ are designed exactly for the purpose of 
generating new data points that resemble those from $X$.

### What is an Autoencoder?
One such generative model that can learn $p_{θ}(x)$ is an _autoencoder_. An autoencoder
is composed of two separate neural networks: an encoder $f$ and decoder $g$, which
are typically symmetric and chained together (Fig 1) for end-to-end generation $x' = f(g(x))$.

<p align="middle" float="left">
  <img src="assets/autoencoder.jpg" width="65%" />
</p>
<p align="center">
    <i> Figure 1. Simplified diagram of an autoencoder. </i>
</p>

#### Encoder
An encoder is a non-linear mapping $f: X \mapsto Z$, where $Z$ is called
the latent representation of $X$. The primary function of the encoder is to
"encode" data into a compact, lower dimensional form (aka dimensionality reduction).
A latent representation is like an embedding, whereby similar representations
should cluster together and dissimilar ones should be far away from each other.

#### Decoder
A decoder can be thought of as the reverse process, $g: Z \mapsto X$, where
its primary function is to "decode" the latent representation and reconstruct 
it back to $X$. Again, similar latent representations should yield similar 
reconstructions, and vice versa.

Now you may ask, if the encoder and decoder are supposed to be inverses of each other,
then aren't we just learning the identity function. Well, in some sense yes, but 
this is not merely a trivial task. If we choose the dimensionality of $Z$ to be relatively 
small compared to $X$, then we will be imposing a strict bottleneck on the model
that encourages the encoder to learn the most salient features pertaining to $X$,
so that the decoder can accurately reconstruct $X$. However, this loss of information
means that the reconstruction will not be perfect ($X' \neq X$). 

#### Reconstruction Error
How do we ensure that $X' \approx X$? Two common loss functions are used for 
measuring reconstruction error: __Binary Cross Entropy (BCE)__ and __Mean Squared Error (MSE)__. 
While both are commonly used, it may make more intuitive sense to use MSE, for two
reasons: 
1. BCE is asymmetric []. While this property is useful for classification, 
where you may want to penalize false positives more than false negatives, it 
does not make much sense to penalize a pixel value of 0.6 more than 0.4 
(supposing the true value is 0.5). 
2. BCE is designed for outputs that model probabilities, while MSE is typically
used for regression. Explain more...
In this implementation, we use MSE as our reconstruction loss, which is defined as:

$$ MSE = \frac{1}{N}||X - X'||_{2}^2$$

where N is the number of samples.
### Variational Autoencoders
Variational Autoencoders (VAEs) represent a powerful class of probabilistic 
generative models that aim to model an underlying distribution of real-world 
data. Coupled with their ability to learn deep representations of such data, 
VAEs are capable of generating entirely new data points using variational 
inference. Unlike traditional autoencoders, which deterministically reconstruct 
outputs from discrete latent representations, VAEs probabilistically generate 
outputs, by sampling from a continuous latent space. For any given input x, 
the encoder of a VAE attempts to learn a mapping of x to a probability 
distribution, which is assumed to be Gaussian (i.e. roughly standard normal). 
This property, sometimes referred to as latent space regularization, is 
achieved by incorporating Kullback-Leibler (KL) Divergence, which measures 
the distance (or dissimilarity) between two probabilities distributions, and 
effectively encourages the modeled latent distributions to be close to standard 
normal.

<p align="middle" float="left">
  <img src="assets/VAE_Basic.png" width="80%" />
</p>

## Implementation
Model design and training were implemented in PyTorch.

### Architecture
The encoder and decoders use only linear layers and ReLU activations. To enable
differentiability, we use the "reparameterization trick," which allows sampling
from the latent distribution z ~ p_theta(z|x). The output is normalized between 
[0, 1.0] using the sigmoid function. Note that since the network is shallow,
additional normalization (e.g. batch normalization) is not necessary.

### Reparameterization Trick

<p align="middle" float="left">
  <img src="assets/Reparameterized_Variational_Autoencoder.png" width="80%" />
</p>

### Loss
* __Reconstruction Loss__: To compare generated images with ground truth (original images),
we measure the reconstruction error. Common reconstruction losses include __Binary Cross Entropy (BCE)__
and __Mean Squared Error (MSE)__. While both are acceptable, in this implementation we use MSE loss, primarily for
two reasons: (1) BCE is both asymmetric and biased around p=0.5 [], and (2) BCE is designed for outputs that
model probabilities. Since images are supposed to...
* __KL Divergence__: Kullback-Leibler (KL) Divergence

## Training & Validation
A vanilla autoencoder (baseline) and VAE were trained on the MNIST
dataset, which contains handwritten digits labeled 0 through 9 (totaling 10 classes). 
The original training set consists of 60k images, each as 28 x 28, 8-bit unsigned
gray scale images. From this set, 10% were randomly chosen for validation (6k images)
to guide model selection. Each model was configured with a different latent size 
(i.e. 2, 5, 10, 20, 50, or 100). Training was conducted for 50 epochs on 
a single NVIDIA V100 GPU, utilizing batch sizes of 1024. Note that images were 
first converted to floating point tensors in the range [0, 1.0]. Optimization was 
carried using AdamW [], with a learning rate of 1e-3 and default weight decay parameters.
Training and validation losses were recorded for each epoch.

<p align="middle" float="left">
  <img src="output/Autoencoder/plots/train_MSE.jpg" width="48%" />
  <img src="output/Autoencoder/plots/val_MSE.jpg" width="48%" />
</p>
<p style="text-align: center;"> 
  <i>MSE over 100 epochs for training (left) and validation (right) for the <b>vanilla autoencoder</b>.</i>
</p>
<p align="middle" float="left">
  <img src="output/VAE/plots/train_MSE.jpg" width="48%" />
  <img src="output/VAE/plots/val_MSE.jpg" width="48%" />
</p>
<p style="text-align: center;"> 
  <i>MSE over 100 epochs for training (left) and validation (right) for the <b>VAE</b>.</i>
</p>
<p align="middle" float="left">
  <img src="output/ConditionalVAE/plots/train_MSE.jpg" width="48%" />
  <img src="output/ConditionalVAE/plots/val_MSE.jpg" width="48%" />
</p>
<p style="text-align: center;"> 
  <i>MSE over 100 epochs for training (left) and validation (right) for the <b>Conditional VAE</b>.</i>
</p>

### Reconstruction Error by Digit

<p align="middle" float="left">
  <img src="output/ConditionalVAE/plots/MSE_by_class.jpg" width="65%" />
</p>

<p align="middle" float="left">
  <img src="output/VAE/plots/MSE_by_class.jpg" width="48%" />
  <img src="output/Autoencoder/plots/MSE_by_class.jpg" width="43.6%" />
</p>

* __Some digits are harder.__ Another important observation is that some digits are harder to reconstruction
than others. We see that digits 1 and 7 are easiest to reconstruct, since they
are both composed of straight lines, and digits 2 and 8 are hardest, as they
are more complex (loops and curves).

* Increasing the dimensionality of the latent space expectedly decreases image
reconstruction error across all digits for the vanilla autoencoder. In general,
increasing the degree of information that needs to be compressed by the encoder
makes it harder for the decoder to reconstruct the original image. 

* In contrast, for the VAE we see that higher latent sizes do not meaningfully
reduce reconstruction error. In fact, the MSE worsens slightly for nearly all 
digits, which perhaps reveals that either (1) the model is underfitting the data as
it struggles to generate continuous latent space representations, or (2) the model
is overfitting to the training data because of the increased capacity. Regardless,
it seems that most of the semantic information of MNIST digits can be compressed 
in a relatively small latent space.

### Model Selection: Choosing Optimal Latent Space Dimensionality 
Based on the reconstruction errors of the VAE model on the validation
split, a latent size of 20 was chosen. 


## Generating Handwritten Digits
There are two ways to generate digits. The first method uses both the encoder and
decoder to reconstruct an image. Although for this task this may seem trivial, 
many unsupervised anomaly detection methods essentially rely on the technique of comparing
an image x from its reconstruction x', whereby large reconstruction errors 
suggest potentially anomalous data. The second method, which only uses the decoder,
allows us to generate new digits "from scratch." We will see the limitations of 
vanilla autoencoders for decoder-only generation, and the benefit of latent space
regularization imposed on VAE models.

### Method 1: Encoder-Decoder Generation
#### Process for Autoencoders
1. Compress image x into its latent representation z, i.e. z = enc(x).
2. Reconstruct image x' by feeding z into the decoder, i.e. x' = dec(z).

#### Process for VAEs
1. Map image x to its latent posterior distribution p_theta(z|mu, sigma). 
2. Sample from the latent probability distribution using the reparameterization 
trick, z = sigma * eps + mu, where eps ~ N(0, 1).
3. Reconstruct image x' by feeding z into the decoder, i.e. x' = dec(z).

#### Process for Conditional VAEs
1. Concatenate image x with its one-hot encoded label y, i.e. concat([x,y]), to its
latent posterior distribution p_theta(z|mu, sigma). 
2. Sample from the latent probability distribution using the reparameterization trick, 
z = sigma * eps + mu, where eps ~ N(0, 1).
3. Concatenate z again with y, i.e. concat([z,y]), and reconstruct image x' by feeding 
it into the decoder, i.e. x' = dec(concat([z, y])).

#### Visualizing Reconstructions
Below, we randomly sample 10 unseen images from the MNIST test split, and visualize
the encoder-decoder reconstructions for each model:
<p align="middle" float="left">
  <img src="output/Autoencoder/plots/reconstructed_digits_color.jpg" width="80%" />
  <img src="output/VAE/plots/reconstructed_digits_color.jpg" width="80%" />
  <img src="output/ConditionalVAE/plots/reconstructed_digits_color.jpg" width="80%" />
</p>
<p style="text-align: center;"> 
  <i>Original handwritten digits and their reconstructions for the vanilla autoencoder (top),
  VAE (middle), and Conditional VAE (bottom).
  </i>
</p>

* __Fuzziness__: we see that the digits appear "fuzzy," which is expected because
of the probabilistic nature of the VAE.
### Method 2: Decoder-Only Generation

#### Process
The process for decoder-only generation is exactly the same as that in Method 1,
except for the fact that instead of compressing x into its latent representation z,
we generate a noise vector z ~ N(0, 1), which simulates sampling from the latent 
space (under the assumption that it is roughly standard normal). In fact, we
don't even have an x to work with, so there is nothing to feed the encoder!

How do we specify which digit we want to generate? While we can only do this
with Conditional VAEs, we simply feed the one-hot label corresponding to the
digit of our choice, along with the noise vector to the decoder.

#### Visualizing Generated Digits
Below, we generate new MNIST-like handwritten digits from random noise vectors:
<p align="middle" float="left">
  <img src="output/Autoencoder/plots/generated_digits_color.jpg" width="80%" />
  <img src="output/VAE/plots/generated_digits_color.jpg" width="80%" />
  <img src="output/ConditionalVAE/plots/generated_digits_color.jpg" width="82%" />
</p>
<p style="text-align: center;"> 
  <i>Generated digits from random noise vectors for the autoencoder (top),
  VAE (middle), and Conditional VAE (bottom).
  </i>
</p>

* __Unregularized Latent Space__: Unsurprisingly, we see that the vanilla autoencoder suffers to
generate anything resembling handwritten digits from the training set. This is
undeniably due to the lack of latent space regularization, hence why VAEs are more powerful. 
Because autoencoders deterministically maps a data point x to its latent representation z,
it will fail to generalize for an unseen z.

* Both the VAE and Conditional VAE produce recognizable digits. However, unlike
the Conditional VAE, which can generate specific digits on command, the 
unconditional VAE generates digits randomly, since it s only responsible 
for reconstructing x' from the sampled noise vector z, which encodes no
information about the digit class. Additionally, we see that the unconditional
VAE produces digits that are more ambiguous, while the Conditional VAE has seemingly
learned to "disentangle" the digit classes.


## Visualizing Latent Spaces

<p align="middle" float="left">
  <img src="output/Autoencoder/plots/latent_space_scatter_2d.jpg" width="48%" />
  <img src="output/Autoencoder/plots/latent_space_kde_1d.jpg" width="44%" />
</p>
<p align="middle" float="left">
  <img src="output/VAE/plots/latent_space_scatter_2d.jpg" width="48%" />
  <img src="output/VAE/plots/latent_space_kde_1d.jpg" width="42.2%" />
</p>
<p align="middle" float="left">
  <img src="output/ConditionalVAE/plots/latent_space_scatter_2d.jpg" width="48%" />
  <img src="output/ConditionalVAE/plots/latent_space_kde_1d.jpg" width="45.3%" />
</p>

<p style="text-align: center;"> 
  <i>Figure 7. Latent representations in 2D (left) and density estimates for 1D representations
    (right) using PCA, for the Autoencoder (top), VAE (middle), and Conditional 
    VAE (bottom). 
  </i>
</p>

<p align="middle" float="left">
  <img src="output/Autoencoder/plots/tsne_latent_space.jpg" width="28%" />
  <img src="output/VAE/plots/tsne_latent_space.jpg" width="31%" />
  <img src="output/ConditionalVAE/plots/tsne_latent_space.jpg" width="31%" />
</p>
<p align="middle" float="left">

</p>
<p align="middle" float="left">

</p>

<p align="center"> 
  <i>Figure 8. Latent representations projected onto 2D space using t-SNE
    algorithm for dimensionality reduction, for the Autoencoder (left), 
    VAE (middle), and conditional VAE (right). 
  </i>
</p>

Fig. 8 shows the latent representations projected onto 2D space when 
t-distributed Stochastic Neighbor Embedding (t-SNE) algorithm [] is applied to 
the latent representations. Since t-SNE is a non-linear dimensionality reduction
method, it applies a non-linear transformation that can separate data points
into clusters, unlike PCA, which applies a linear transformation using the
eigenvectors of the covariance matrix. For this reason, T-SNE is useful for
identify features or embeddings that share similarities (hence, are neighbors).

We see that for the autoencoder, the latent representations can be clustered
by their digits. The large gaps between clusters signify points in the reduced
latent space that do not encode meaningful information, so when an image gets mapped
to one of those regions in the original latent space, it will most certainly 
produce a noisy, meaningless image.

So you may be wondering why the conditional VAE's latent distribution looks 
more bivariate normal than the VAE. Since the one-hot encoding $Y$ is provided
to the encoder, the encoder does not need to "encode" any additional information
regarding the digit class into latent representations. Therefore, this allows
the encoder to encode more information about the actual styles and shape of 
digits, which can be shared across many digit classes. Instead, it is then
the job of the decoder to reconstruct the specific digit.

## Meaning of Latent Representations 
Okay, we now know that VAEs are particularly good at generating new digits
from latent representations, given their ability to manifest regularized latent
spaces. 

But what do the latent representations actually encode? Let's consider VAEs
with latent representations that live in 2D space. Typically, we sample $z ~ N(0, 1)$,
because the latent space is supposed to be approximately standard normal, hence 
why we can feed the decoder a random noise vector. But we don't have to do that.
Instead, we can traverse a subspace of the latent space, say, a 10 x 10 grid centered
around (0, 0). This allows us to see more clearly, the relationship between the 
components $(z_1, z_2)$ and the generated digit:

<p align="middle" float="left">
  <img src="output/VAE/plots/generated_digits_latent_grid.jpg" width="45%" />
  <img src="output/ConditionalVAE/plots/conditional_vae_latent_grid.gif" width="46.5%" />
</p>
<p align="center">
    <i> Figure 9. Generated digits corresponding to a grid of latent vectors 
        centered at (0, 0) for the VAE (left) and conditional VAE (right). </i>
</p>

* For unconditional VAEs, traversing the latent space in different directions 
changes the digit that is generated. We see that every digit class is accounted
for in the grid, and has a "center." Moving away from that center means moving
closer to another digit's center, and you can see digits morphing into other digits
(e.g. 7s morphing into 9s and 2s morphing into 8s). However, it is difficult to
see how the style of the digit is accounted for in the latent space, since the
digit classes are "entangled."

* For conditional VAEs, we see something even more interesting. Since it is the 
job of the decoder to reconstruct the specific digit when given a label, the encoder
can encode more primitive features that are shared among all digit classes, like
curves, rotation, and style. 
