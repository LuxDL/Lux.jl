```@raw html
<script setup lang="ts">
import Gallery from "../components/Gallery.vue";

const beginner = [
  {
    href: "beginner/1_Basics",
    src: "https://picsum.photos/350/250?image=444",
    caption: "Julia & Lux for the Uninitiated",
    desc: "How to get started with Julia and Lux for those who have never used Julia before."
  },
  {
    href: "beginner/2_PolynomialFitting",
    src: "../mlp.webp",
    caption: "Fitting a Polynomial using MLP",
    desc: "Learn the Basics of Lux by fitting a Multi-Layer Perceptron to a Polynomial."
  },
  {
    href: "beginner/3_SimpleRNN",
    src: "../lstm-illustrative.webp",
    caption: "Training a Simple LSTM",
    desc: "Learn how to define custom layers and train an RNN on time-series data."
  },
  {
    href: "beginner/4_SimpleChains",
    src: "../blas_optimizations.jpg",
    caption: "Use SimpleChains.jl as a Backend",
    desc: "Learn how to train small neural networks really fast on CPU."
  },
  {
    href: "beginner/5_OptimizationIntegration",
    src: "../optimization_integration.png",
    caption: "Fitting with Optimization.jl",
    desc: "Learn how to use Optimization.jl with Lux (on GPUs)."
  },
  {
    href: "https://luxdl.github.io/Boltz.jl/stable/tutorials/1_GettingStarted",
    src: "https://blog.roboflow.com/content/images/2021/06/image-18.png",
    caption: "Pre-Built Deep Learning Models",
    desc: "Use Boltz.jl to load pre-built DL and SciML models."
  }
];

const intermediate = [
  {
    href: "intermediate/1_NeuralODE",
    src: "../mnist.jpg",
    caption: "MNIST Classification using Neural ODE",
    desc: "Train a Neural Ordinary Differential Equations to classify MNIST Images."
  },
  {
    href: "intermediate/3_HyperNet",
    src: "../hypernet.jpg",
    caption: "Training a HyperNetwork",
    desc: "Train a hypernetwork to work on multiple datasets by predicting NN parameters."
  },
  {
    href: "intermediate/4_PINN2DPDE",
    src: "../pinn_nested_ad.gif",
    caption: "Training a PINN",
    desc: "Train a PINN to solve 2D PDEs (using Nested AD)."
  },
  {
    href: "intermediate/5_ConvolutionalVAE",
    src: "../conditional_vae.png",
    caption: "Convolutional VAE for MNIST",
    desc: "Train a Convolutional VAE to generate images from a latent space."
  },
  {
    href: "intermediate/6_GCN_Cora",
    src: "../gcn_cora.jpg",
    caption: "Graph Convolutional Network on Cora",
    desc: "Train a Graph Convolutional Network on Cora dataset."
  },
  {
    href: "intermediate/7_RealNVP",
    src: "../realnvp.png",
    caption: "Normalizing Flows for Density Estimation",
    desc: "Train a normalizing flow for density estimation on the Moons dataset.",
  },
  {
    href: "intermediate/8_LSTMEncoderDecoder",
    src: "../lstm_encoder_decoder.png",
    caption: "LSTM Encoder-Decoder",
    desc: "Train an LSTM Encoder-Decoder for sequence-to-sequence tasks."
  },
  {
    href: "intermediate/9_CIFAR10_conv_mixer",
    src: "https://datasets.activeloop.ai/wp-content/uploads/2022/09/CIFAR-10-dataset-Activeloop-Platform-visualization-image-1.webp",
    caption: "Conv-Mixer on CIFAR-10",
    desc: "Train Conv-Mixer on CIFAR-10 to 90% accuracy."
  },
  {
    href: "intermediate/10_CIFAR10_simple_cnn",
    src: "https://datasets.activeloop.ai/wp-content/uploads/2022/09/CIFAR-10-dataset-Activeloop-Platform-visualization-image-1.webp",
    caption: "Simple Convolutional Neural Network on CIFAR-10",
    desc: "Train a CNN on CIFAR-10 to 95% accuracy."
  },
  {
    href: "intermediate/11_CIFAR10_resnet20",
    src: "https://datasets.activeloop.ai/wp-content/uploads/2022/09/CIFAR-10-dataset-Activeloop-Platform-visualization-image-1.webp",
    caption: "ResNet20 on CIFAR-10",
    desc: "Train a ResNet20 on CIFAR-10 to 90% accuracy."
  }
];

const advanced = [
  {
    href: "advanced/1_GravitationalWaveForm",
    src: "../gravitational_waveform.png",
    caption: "Neural ODE to Model Gravitational Waveforms",
    desc: "Training a Neural ODE to fit simulated data of gravitational waveforms."
  },
  {
    href: "https://luxdl.github.io/Boltz.jl/stable/tutorials/2_SymbolicOptimalControl",
    src: "../symbolic_optimal_control.png",
    caption: "Optimal Control with Symbolic UDE",
    desc: "Train a UDE and replace a part of it with Symbolic Regression."
  },
  {
    href: "advanced/2_DDIM",
    src: "https://raw.githubusercontent.com/LuxDL/Lux.jl/main/examples/DDIM/assets/flowers_generated.png",
    caption: "Denoising Diffusion Implicit Model (DDIM)",
    desc: "Train a Diffusion Model to generate images from Gaussian noises."
  },
  {
    href: "advanced/3_ImageNet",
    src: "https://datasets.activeloop.ai/wp-content/uploads/2022/09/ImageNet-dataset-main-image.webp",
    caption: "ImageNet Classification",
    desc: "Train Large Image Classifiers using Lux (on Distributed GPUs)."
  },
  {
    href: "advanced/4_Qwen3",
    src: "../qwen3-30a3.jpg",
    caption: "Text Generation with Qwen-3",
    desc: "Building a command-line text generation tool with Lux by loading pre-trained Qwen-3 models."
  }
];

const third_party = [
  {
    href: "https://docs.sciml.ai/Overview/stable/showcase/pinngpu/",
    src: "../pinn.gif",
    caption: "GPU-Accelerated Physics-Informed Neural Networks",
    desc: "Use Machine Learning (PINNs) to solve the Heat Equation PDE on a GPU."
  },
  {
    href: "https://turinglang.org/docs/tutorials/bayesian-neural-networks/",
    src: "https://github.com/TuringLang.png",
    caption: "Bayesian Neural Networks",
    desc: "Figure out how to use Probabilistic Programming Frameworks like Turing with Lux."
  },
  {
    href: "https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode_weather_forecast/",
    src: "../weather-neural-ode.gif",
    caption: "Weather Forecasting with Neural ODEs",
    desc: "Train a neural ODEs to a multidimensional weather dataset and use it for weather forecasting."
  },
  {
    href: "https://docs.sciml.ai/SciMLSensitivity/stable/examples/sde/SDE_control/",
    src: "../neural-sde.png",
    caption: "Controlling Stochastic Differential Equations",
    desc: "Control the time evolution of a continuously monitored qubit described by an SDE with multiplicative scalar noise."
  },
  {
    href: "https://github.com/Dale-Black/ComputerVisionTutorials.jl/",
    src: "https://raw.githubusercontent.com/Dale-Black/ComputerVisionTutorials.jl/main/assets/image-seg-green.jpeg",
    caption: "Medical Image Segmentation",
    desc: "Explore various aspects of deep learning for medical imaging and a comprehensive overview of Julia packages."
  },
  {
    href: "https://github.com/agdestein/NeuralClosureTutorials",
    src: "https://raw.githubusercontent.com/agdestein/NeuralClosureTutorials/main/assets/navier_stokes.gif",
    caption: "Neural PDE closures",
    desc: "Learn an unknown term in a PDE using convolutional neural networks and Fourier neural operators."
  }
];
</script>

# Tutorials

## Beginner Tutorials

<Gallery :images="beginner" />

## Intermediate Tutorials

<Gallery :images="intermediate" />

## Advanced Tutorials

<Gallery :images="advanced" />

## Selected 3rd Party Tutorials

::: warning

These tutorials are developed by the community and may not be up-to-date with the latest
version of `Lux.jl`. Please refer to the official documentation for the most up-to-date
information.

Please open an issue (ideally both at `Lux.jl` and at the downstream linked package) if any
of them are non-functional and we will try to get them updated.

:::

<Gallery :images="third_party" />


::: tip

If you found an amazing tutorial showcasing `Lux.jl` online, or wrote one yourself, please
open an issue or PR to add it to the list!

:::
```
