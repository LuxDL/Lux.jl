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
    src: "https://production-media.paperswithcode.com/datasets/ImageNet-0000000008-f2e87edd_Y0fT5zg.jpg",
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
    href: "intermediate/2_BayesianNN",
    src: "https://github.com/TuringLang.png",
    caption: "Bayesian Neural Networks",
    desc: "Figure out how to use Probabilistic Programming Frameworks like Turing with Lux."
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
  }
];

const large_models = [
  {
    href: "https://github.com/LuxDL/Lux.jl/tree/main/examples/ImageNet",
    src: "https://production-media.paperswithcode.com/datasets/ImageNet-0000000008-f2e87edd_Y0fT5zg.jpg",
    caption: "ImageNet Classification",
    desc: "Train Large Image Classifiers using Lux (on Distributed GPUs)."
  },
  {
    href: "https://github.com/LuxDL/Lux.jl/tree/main/examples/DDIM",
    src: "https://raw.githubusercontent.com/LuxDL/Lux.jl/main/examples/DDIM/assets/flowers_generated.png",
    caption: "Denoising Diffusion Implicit Model (DDIM)",
    desc: "Train a Diffusion Model to generate images from Gaussian noises."
  },
  {
    href: "https://github.com/LuxDL/Lux.jl/tree/main/examples/CIFAR10",
    src: "https://datasets.activeloop.ai/wp-content/uploads/2022/09/CIFAR-10-dataset-Activeloop-Platform-visualization-image-1.webp",
    caption: "Vision Models on CIFAR-10",
    desc: "Train differnt vision models on CIFAR-10 to 90% accuracy within 10 minutes."
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

## Larger Models

::: warning

These models are part of the Lux examples, however, these are larger model that cannot be
run on CI and aren't frequently tested. If you find a bug in one of these models, please
open an issue or PR to fix it.

:::

<Gallery :images="large_models" />

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
