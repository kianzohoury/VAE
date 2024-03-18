
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

<p align="middle" float="left">
  <img src="output/Autoencoder/validation_MSE.jpg" width="45%" />
  <img src="output/VAE/validation_MSE.jpg" width="45%" />
</p>

### Effect of Latent Dimensionality on Digit Reconstruction
<p align="middle" float="left">
  <img src="output/Autoencoder/class_results_MSE.jpg" width="45%" />
  <img src="output/VAE/class_results_MSE.jpg" width="45%" />
</p>

* Increasing the dimensionality of the latent space improves the image
reconstruction (as measured by MSE) across all digits for vanilla autoencoder 
models. By not having to compress information too much, the autoencoder can
better model the identity function.

* In contrast, for VAE models we see that past a latent size of about 20, the 
reconstruction task is not meaningfully improved. In fact, the reconstruction 
losses worsen slightly for nearly all digits, which perhaps reveals that most 
of the semantic information of a digit can be compressed in a relatively small 
latent space (say, no smaller than 2 but no larger than 20).



the benefit of increasing the dimensionality
diminishes 


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