```@raw html
<script setup lang="ts">
import Gallery from "../components/Gallery.vue";

const beginner = [
  {
    href: "beginner/1_Basics",
    src: "../assets/tutorials/julia.jpg",
    caption: "Julia & Lux for the Uninitiated",
    desc: "How to get started with Julia and Lux for those who have never used Julia before."
  },
  {
    href: "beginner/2_PolynomialFitting",
    src: "../assets/tutorials/mlp.webp",
    caption: "Fitting a Polynomial using MLP",
    desc: "Learn the Basics of Lux by fitting a Multi-Layer Perceptron to a Polynomial."
  },
  {
    href: "beginner/3_SimpleRNN",
    src: "../assets/tutorials/lstm-illustrative.webp",
    caption: "Training a Simple LSTM",
    desc: "Learn how to define custom layers and train an RNN on time-series data."
  },
  {
    href: "beginner/4_SimpleChains",
    src: "../assets/tutorials/blas_optimizations.jpg",
    caption: "Use SimpleChains.jl as a Backend",
    desc: "Learn how to train small neural networks really fast"
  }
];

const intermediate = [
  {
    href: "intermediate/1_NeuralODE",
    src: "../assets/tutorials/mnist.jpg",
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
    src: "../assets/tutorials/hypernet.jpg",
    caption: "Training a HyperNetwork",
    desc: "Train a hypernetwork to work on multiple datasets by predicting neural network parameters."
  }
];

const advanced = [
  {
    href: "advanced/1_GravitationalWaveForm",
    src: "../assets/tutorials/gravitational_waveform.png",
    caption: "Neural ODE to Model Gravitational Waveforms",
    desc: "Training a Neural ODE to fit simulated data of gravitational waveforms."
  }
];

const aggregated = [

];
</script>

# Tutorials

## Beginner Tutorials

<Gallery :images="beginner" />

## Intermediate Tutorials

<Gallery :images="intermediate" />

## Advanced Tutorials

<Gallery :images="advanced" />

## Aggregated Tutorials

::: warning

These tutorials are developed by the community and may not be up-to-date with the latest
version of `Lux.jl`. Please refer to the official documentation for the most up-to-date
information.

Please open an issue (ideally both at `Lux.jl` and at the downstream linked package) if any
of them are non-functional and we will try to get them updated.

:::

<Gallery :images="aggregated" />
```
