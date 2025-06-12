import { d as defineComponent, o as openBlock, c as createElementBlock, l as createBaseVNode, m as unref, g as withBase, t as toDisplayString, _ as _export_sfc, F as Fragment, E as renderList, b as createBlock, M as mergeProps, I as createVNode, a as createTextVNode } from "./chunks/framework.BksySIuR.js";
const _hoisted_1$2 = { class: "img-box" };
const _hoisted_2$1 = ["href"];
const _hoisted_3$1 = ["src"];
const _hoisted_4$1 = { class: "transparent-box1" };
const _hoisted_5$1 = { class: "caption" };
const _hoisted_6$1 = { class: "transparent-box2" };
const _hoisted_7$1 = { class: "subcaption" };
const _hoisted_8 = { class: "opacity-low" };
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "GalleryImage",
  props: {
    href: {},
    src: {},
    caption: {},
    desc: {}
  },
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$2, [
        createBaseVNode("a", { href: _ctx.href }, [
          createBaseVNode("img", {
            src: unref(withBase)(_ctx.src),
            height: "150px",
            alt: ""
          }, null, 8, _hoisted_3$1),
          createBaseVNode("div", _hoisted_4$1, [
            createBaseVNode("div", _hoisted_5$1, [
              createBaseVNode("h2", null, toDisplayString(_ctx.caption), 1)
            ])
          ]),
          createBaseVNode("div", _hoisted_6$1, [
            createBaseVNode("div", _hoisted_7$1, [
              createBaseVNode("p", _hoisted_8, toDisplayString(_ctx.desc), 1)
            ])
          ])
        ], 8, _hoisted_2$1)
      ]);
    };
  }
});
const GalleryImage = /* @__PURE__ */ _export_sfc(_sfc_main$2, [["__scopeId", "data-v-06a0366f"]]);
const _hoisted_1$1 = { class: "gallery-image" };
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "Gallery",
  props: {
    images: {}
  },
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        (openBlock(true), createElementBlock(Fragment, null, renderList(_ctx.images, (image) => {
          return openBlock(), createBlock(GalleryImage, mergeProps({ ref_for: true }, image), null, 16);
        }), 256))
      ]);
    };
  }
});
const Gallery = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["__scopeId", "data-v-578d61bc"]]);
const _hoisted_1 = /* @__PURE__ */ createBaseVNode("h1", {
  id: "tutorials",
  tabindex: "-1"
}, [
  /* @__PURE__ */ createTextVNode("Tutorials "),
  /* @__PURE__ */ createBaseVNode("a", {
    class: "header-anchor",
    href: "#tutorials",
    "aria-label": 'Permalink to "Tutorials"'
  }, "​")
], -1);
const _hoisted_2 = /* @__PURE__ */ createBaseVNode("h2", {
  id: "beginner-tutorials",
  tabindex: "-1"
}, [
  /* @__PURE__ */ createTextVNode("Beginner Tutorials "),
  /* @__PURE__ */ createBaseVNode("a", {
    class: "header-anchor",
    href: "#beginner-tutorials",
    "aria-label": 'Permalink to "Beginner Tutorials"'
  }, "​")
], -1);
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("h2", {
  id: "intermediate-tutorials",
  tabindex: "-1"
}, [
  /* @__PURE__ */ createTextVNode("Intermediate Tutorials "),
  /* @__PURE__ */ createBaseVNode("a", {
    class: "header-anchor",
    href: "#intermediate-tutorials",
    "aria-label": 'Permalink to "Intermediate Tutorials"'
  }, "​")
], -1);
const _hoisted_4 = /* @__PURE__ */ createBaseVNode("h2", {
  id: "advanced-tutorials",
  tabindex: "-1"
}, [
  /* @__PURE__ */ createTextVNode("Advanced Tutorials "),
  /* @__PURE__ */ createBaseVNode("a", {
    class: "header-anchor",
    href: "#advanced-tutorials",
    "aria-label": 'Permalink to "Advanced Tutorials"'
  }, "​")
], -1);
const _hoisted_5 = /* @__PURE__ */ createBaseVNode("h2", {
  id: "selected-3rd-party-tutorials",
  tabindex: "-1"
}, [
  /* @__PURE__ */ createTextVNode("Selected 3rd Party Tutorials "),
  /* @__PURE__ */ createBaseVNode("a", {
    class: "header-anchor",
    href: "#selected-3rd-party-tutorials",
    "aria-label": 'Permalink to "Selected 3rd Party Tutorials"'
  }, "​")
], -1);
const _hoisted_6 = /* @__PURE__ */ createBaseVNode("div", { class: "warning custom-block" }, [
  /* @__PURE__ */ createBaseVNode("p", { class: "custom-block-title" }, "WARNING"),
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createTextVNode("These tutorials are developed by the community and may not be up-to-date with the latest version of "),
    /* @__PURE__ */ createBaseVNode("code", null, "Lux.jl"),
    /* @__PURE__ */ createTextVNode(". Please refer to the official documentation for the most up-to-date information.")
  ]),
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createTextVNode("Please open an issue (ideally both at "),
    /* @__PURE__ */ createBaseVNode("code", null, "Lux.jl"),
    /* @__PURE__ */ createTextVNode(" and at the downstream linked package) if any of them are non-functional and we will try to get them updated.")
  ])
], -1);
const _hoisted_7 = /* @__PURE__ */ createBaseVNode("div", { class: "tip custom-block" }, [
  /* @__PURE__ */ createBaseVNode("p", { class: "custom-block-title" }, "TIP"),
  /* @__PURE__ */ createBaseVNode("p", null, [
    /* @__PURE__ */ createTextVNode("If you found an amazing tutorial showcasing "),
    /* @__PURE__ */ createBaseVNode("code", null, "Lux.jl"),
    /* @__PURE__ */ createTextVNode(" online, or wrote one yourself, please open an issue or PR to add it to the list!")
  ])
], -1);
const __pageData = JSON.parse('{"title":"Tutorials","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/index.md","filePath":"tutorials/index.md","lastUpdated":null}');
const __default__ = { name: "tutorials/index.md" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  ...__default__,
  setup(__props) {
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
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", null, [
        _hoisted_1,
        _hoisted_2,
        createVNode(Gallery, { images: beginner }),
        _hoisted_3,
        createVNode(Gallery, { images: intermediate }),
        _hoisted_4,
        createVNode(Gallery, { images: advanced }),
        _hoisted_5,
        _hoisted_6,
        createVNode(Gallery, { images: third_party }),
        _hoisted_7
      ]);
    };
  }
});
export {
  __pageData,
  _sfc_main as default
};
