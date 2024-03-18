
note that we don't need normalization since the network is very shallow

here we can use log_var or var, doesn't matter

1. Train autoencoders for 30 epochs, plot train/val, select best latent dim
based on val reconstruction error. Plot line graphs for each digit (MSE on y, latent size on x) on test data.
2. Visualize (on test data) across all latent dims in a grid, against real data.
3. Plot tSNES, compare all latent dims in a grid.
4. Repeat steps for VAE.
5. For each AE VAE pair, compare reconstruction losses.
6. Pick best VAE and generate new samples from noise, and show lack of guided generation.
6. Fix this issue with conditional VAE (repeat steps).


## Training & Validation
A vanilla autoencoder (baseline) and VAE were trained on the MNIST
dataset, which contains handwritten digits labeled 0 through 9 (totaling 10 classes). 
The original training set consists of 60k images, each as 28 x 28, 8-bit unsigned
gray scale images. From this set, 10% were randomly chosen for validation (6k images)
to guide model selection. Each model was configured with a different latent size 
(i.e. 2, 5, 10, 20, 50, or 100). Training was conducted for 50 epochs on 
a single NVIDIA V100 GPU, utilizing batch sizes of 1024. Note that images were 
first converted to floating point tensors in the range (0, 1). Optimization was 
carried using AdamW [], with a learning rate of 1e-3 and default weight decay parameters.
Training and validation losses were recorded for each epoch.

<p align="middle" float="left">
  <img src="output/Autoencoder/validation_MSE.jpg" width="45%" />
  <img src="output/VAE/validation_MSE.jpg" width="45%" />
</p>
<p style="text-align: center;"> <i>Epoch-wise MSE loss recorded on validation data for the vanilla autoencoder 
    (left) and VAE (right) </i>
</p>

### Effect of Latent Space Dimensionality on Digit Reconstruction
<p align="middle" float="left">
  <img src="output/Autoencoder/class_results_MSE.jpg" width="48%" />
  <img src="output/VAE/class_results_MSE.jpg" width="48%" />
</p>

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

* __Some digits are harder.__ Another important observation is that some digits are harder to reconstruction
than others. We see that digits 1 and 7 are easiest to reconstruct, since they
are both composed of straight lines, and digits 2 and 8 are hardest, as they
are more complex (loops and curves).

### Model Selection: Choosing Optimal Latent Space Dimensionality 
Based on the reconstruction errors of the VAE model on the validation
split, a latent size of 20 was chosen. 
<p align="middle" float="left">
  <img src="output/VAE/train_val_latent_20_MSE.jpg" width="45%" />
</p>


[//]: # ()
[//]: # (<div style="display: flex; flex-direction: row; justify-content: center">)

[//]: # (    <p align="center">)

[//]: # (        <img src="output/Autoencoder/validation_MSE.jpg" width=30%/>)

[//]: # (    </p>)

[//]: # (    <p align="center">)

[//]: # (        <img src="output/VAE/validation_MSE.jpg" width=30%/>)

[//]: # (    </p>)

[//]: # (    <p align="center">)

[//]: # (        <img src="output/ConditionalVAE/validation_MSE.jpg" width=30%/>)

[//]: # (    </p>)

[//]: # (    <figcaption>)

[//]: # (        Validation losses for Autoencoders &#40;left&#41;, VAE &#40;middle&#41;, and )

[//]: # (        ConditionalVAE &#40;right&#41;. )

[//]: # (    </figcaption>)

[//]: # (</div>)




## Choosing Optimal Latent Space Dimensionality

[//]: # (![img]&#40;output/Autoencoder/validation_MSE.jpg&#41;)

[//]: # (![img]&#40;output/Autoencoder/class_results_MSE.jpg&#41;)

[//]: # (![img]&#40;output/Autoencoder/reconstruction_grid.jpg&#41;)

## VAE
![img](output/VAE/validation_MSE.jpg)
![img](output/VAE/validation_KL_Divergence.jpg)
![img](output/VAE/class_results_MSE.jpg)
![img](output/VAE/reconstruction_grid.jpg)
![img](output/VAE/decoding_grid.jpg)