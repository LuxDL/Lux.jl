import { defineConfig } from "vitepress";
import mathjax3 from "markdown-it-mathjax3";
import markdownitsup from "markdown-it-sup";
import markdownitabbr from "markdown-it-abbr";
import markdownittasklists from "markdown-it-task-lists";

const customElements = ["mjx-container"];

const tutorialsSideBar = {
  items: [
    {
      text: "Beginner",
      base: "/generated/tutorials/beginner/",
      items: [
        { text: "Julia & Lux for the Uninitiated", link: "Basics/main" },
        { text: "Fitting a Polynomial", link: "PolynomialFitting/main" },
        { text: "Training a Simple LSTM", link: "SimpleRNN/main" },
      ],
    },
    {
      text: "Intermediate",
      base: "/generated/tutorials/intermediate/",
      items: [
        {
          text: "MNIST Classification using Neural ODE",
          link: "NeuralODE/main",
        },
        { text: "Bayesian Neural Network", link: "BayesianNN/main" },
        { text: "Training a Hyper Network", link: "HyperNet/main" },
      ],
    },
  ],
};

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "LuxDL Docs",
  description: "Elegant Deep Learning in Julia",

  lastUpdated: true,
  cleanUrls: true,

  themeConfig: {
    logo: {
      light: "/lux-logo.svg",
      dark: "/lux-logo-dark.svg",
    },

    nav: [
      { text: "Home", link: "/" },
      { text: "Getting Started", link: "/introduction/installation" },
      { text: "Ecosystem", link: "/ecosystem" },
      { text: "Tutorials", link: "/tutorials/" },
      { text: "Manual", link: "/generated/manual/interface" },
      { text: "API", link: "/generated/api/Lux/layers" },
    ],

    sidebar: {
      "/introduction/": {
        base: "/introduction/",
        items: [
          { text: "Installation", link: "installation" },
          { text: "Additional Resources", link: "resources" },
          { text: "Quickstart", link: "quickstart" },
          { text: "Why Lux?", link: "overview" },
          { text: "Citing Lux", link: "citation" },
        ],
      },
      "/tutorials/": tutorialsSideBar,
      "/generated/tutorials/": tutorialsSideBar,
      "/generated/manual/": {
        base: "/generated/manual/",
        items: [
          { text: "Lux Interface", link: "interface" },
          { text: "Migration Guide from Flux.jl", link: "migrate_from_flux" },
          {
            text: "Freezing Model Parameters",
            link: "freezing_model_parameters",
          },
          { text: "Dispatch on Custom Inputs", link: "dispatch_custom_input" },
          { text: "GPU Management", link: "gpu_management" },
          { text: "Weight Initialization", link: "weight_initializers" },
        ],
      },
      "/generated/api/": {
        base: "/generated/api/",
        items: [
          {
            text: "Lux",
            items: [
              { text: "Layers", link: "Lux/layers" },
              { text: "Flux To Lux", link: "Lux/flux_to_lux" },
              { text: "Misc. Utilities", link: "Lux/utilities" },
              { text: "Experimental Features", link: "Lux/contrib" },
            ],
          },
          {
            text: "Building Blocks for Lux",
            items: [
              { text: "LuxCore", link: "LuxCore/" },
              { text: "LuxLib", link: "LuxLib/" },
              { text: "WeightInitializers", link: "WeightInitializers/" },
            ],
          },
          {
            text: "Building Domain Specific Models",
            items: [{ text: "Boltz", link: "Boltz/" }],
          },
          {
            text: "Accelerator Support",
            items: [
              { text: "LuxDeviceUtils", link: "LuxDeviceUtils/" },
              { text: "LuxCUDA", link: "LuxCUDA/" },
              { text: "LuxAMDGPU", link: "LuxAMDGPU/" },
            ],
          },
          {
            text: "Testing Functionality",
            items: [{ text: "LuxTestUtils", link: "LuxTestUtils/" }],
          },
        ],
      },

      socialLinks: [
        { icon: "github", link: "https://github.com/LuxDL/" },
        { icon: "twitter", link: "https://twitter.com/avikpal1410" },
      ],
    },

    search: {
      provider: "local",
    },

    footer: {
      message:
        "Released under the MIT License. Powered by the <a href='https://www.julialang.org'>Julia Programming Language</a>.",
      copyright: "Copyright © 2022-Present Avik Pal",
    },
  },

  markdown: {
    theme: {
      light: "github-light",
      dark: "github-dark",
    },

    // adjust how header anchors are generated,
    // useful for integrating with tools that use different conventions
    anchor: {
      slugify(str) {
        return encodeURIComponent(str);
      },
    },

    config: (md) => {
      md.use(markdownitsup);
      md.use(markdownitabbr);
      md.use(markdownittasklists);
      md.use(mathjax3);
    },
  },

  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag),
      },
    },
  },
});
