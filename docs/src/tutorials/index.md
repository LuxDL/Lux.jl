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
    desc: "Train a hypernetwork to work on multiple datasets by predicting neural network parameters."
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
    href: "advanced/2_SymbolicOptimalControl",
    src: "../symbolic_optimal_control.png",
    caption: "Optimal Control with Symbolic UDE",
    desc: "Train a UDE and replace a part of it with Symbolic Regression."
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
