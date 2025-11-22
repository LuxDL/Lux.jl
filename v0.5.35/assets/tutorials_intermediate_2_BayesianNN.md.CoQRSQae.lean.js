import { _ as _export_sfc, c as createElementBlock, l as createBaseVNode, a4 as createStaticVNode, o as openBlock } from "./chunks/framework.CSxcQPlK.js";
const _imports_0 = "/v0.5.35/assets/results.Dao8ZugC.gif";
const __pageData = JSON.parse('{"title":"Bayesian Neural Network","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/2_BayesianNN.md","filePath":"tutorials/intermediate/2_BayesianNN.md","lastUpdated":null}');
const _sfc_main = { name: "tutorials/intermediate/2_BayesianNN.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode("", 32);
const _hoisted_33 = {
  class: "MathJax",
  jax: "SVG",
  display: "true",
  style: { "direction": "ltr", "display": "block", "text-align": "center", "margin": "1em 0", "position": "relative" }
};
const _hoisted_34 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-3.222ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "46.264ex",
  height: "6.301ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -1361 20448.8 2785.1",
  "aria-hidden": "true"
};
const _hoisted_35 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_36 = [
  _hoisted_35
];
const _hoisted_37 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "block",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "overflow": "hidden", "width": "100%" }
}, [
  /* @__PURE__ */ createBaseVNode("math", {
    xmlns: "http://www.w3.org/1998/Math/MathML",
    display: "block"
  }, [
    /* @__PURE__ */ createBaseVNode("mi", null, "p"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
      /* @__PURE__ */ createBaseVNode("mover", null, [
        /* @__PURE__ */ createBaseVNode("mi", null, "x"),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "~")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("mo", {
      "data-mjx-texclass": "ORD",
      stretchy: "false"
    }, "|"),
    /* @__PURE__ */ createBaseVNode("mi", null, "X"),
    /* @__PURE__ */ createBaseVNode("mo", null, ","),
    /* @__PURE__ */ createBaseVNode("mi", null, "α"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")"),
    /* @__PURE__ */ createBaseVNode("mo", null, "="),
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mo", { "data-mjx-texclass": "OP" }, "∫"),
      /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
        /* @__PURE__ */ createBaseVNode("mi", null, "θ")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("mi", null, "p"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
      /* @__PURE__ */ createBaseVNode("mover", null, [
        /* @__PURE__ */ createBaseVNode("mi", null, "x"),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "~")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("mo", {
      "data-mjx-texclass": "ORD",
      stretchy: "false"
    }, "|"),
    /* @__PURE__ */ createBaseVNode("mi", null, "θ"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")"),
    /* @__PURE__ */ createBaseVNode("mi", null, "p"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mi", null, "θ"),
    /* @__PURE__ */ createBaseVNode("mo", {
      "data-mjx-texclass": "ORD",
      stretchy: "false"
    }, "|"),
    /* @__PURE__ */ createBaseVNode("mi", null, "X"),
    /* @__PURE__ */ createBaseVNode("mo", null, ","),
    /* @__PURE__ */ createBaseVNode("mi", null, "α"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")"),
    /* @__PURE__ */ createBaseVNode("mo", null, "≈"),
    /* @__PURE__ */ createBaseVNode("munder", null, [
      /* @__PURE__ */ createBaseVNode("mo", { "data-mjx-texclass": "OP" }, "∑"),
      /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
        /* @__PURE__ */ createBaseVNode("mi", null, "θ"),
        /* @__PURE__ */ createBaseVNode("mo", null, "∼"),
        /* @__PURE__ */ createBaseVNode("mi", null, "p"),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
        /* @__PURE__ */ createBaseVNode("mi", null, "θ"),
        /* @__PURE__ */ createBaseVNode("mo", {
          "data-mjx-texclass": "ORD",
          stretchy: "false"
        }, "|"),
        /* @__PURE__ */ createBaseVNode("mi", null, "X"),
        /* @__PURE__ */ createBaseVNode("mo", null, ","),
        /* @__PURE__ */ createBaseVNode("mi", null, "α"),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("msub", null, [
      /* @__PURE__ */ createBaseVNode("mi", null, "f"),
      /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
        /* @__PURE__ */ createBaseVNode("mi", null, "θ")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "("),
    /* @__PURE__ */ createBaseVNode("mrow", { "data-mjx-texclass": "ORD" }, [
      /* @__PURE__ */ createBaseVNode("mover", null, [
        /* @__PURE__ */ createBaseVNode("mi", null, "x"),
        /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "~")
      ])
    ]),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, ")")
  ])
], -1);
const _hoisted_38 = /* @__PURE__ */ createStaticVNode("", 16);
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, [
    _hoisted_1,
    createBaseVNode("mjx-container", _hoisted_33, [
      (openBlock(), createElementBlock("svg", _hoisted_34, _hoisted_36)),
      _hoisted_37
    ]),
    _hoisted_38
  ]);
}
const _2_BayesianNN = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  _2_BayesianNN as default
};
