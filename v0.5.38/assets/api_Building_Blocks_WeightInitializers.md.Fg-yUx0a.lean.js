import { _ as _export_sfc, c as createElementBlock, l as createBaseVNode, a as createTextVNode, a4 as createStaticVNode, o as openBlock } from "./chunks/framework.Cu_l04oX.js";
const __pageData = JSON.parse('{"title":"WeightInitializers","description":"","frontmatter":{},"headers":[],"relativePath":"api/Building_Blocks/WeightInitializers.md","filePath":"api/Building_Blocks/WeightInitializers.md","lastUpdated":null}');
const _sfc_main = { name: "api/Building_Blocks/WeightInitializers.md" };
const _hoisted_1 = /* @__PURE__ */ createStaticVNode("", 8);
const _hoisted_9 = { style: { "border-width": "1px", "border-style": "solid", "border-color": "black", "padding": "1em", "border-radius": "25px" } };
const _hoisted_10 = /* @__PURE__ */ createBaseVNode("a", {
  id: "WeightInitializers.glorot_uniform",
  href: "#WeightInitializers.glorot_uniform"
}, "#", -1);
const _hoisted_11 = /* @__PURE__ */ createBaseVNode("b", null, [
  /* @__PURE__ */ createBaseVNode("u", null, "WeightInitializers.glorot_uniform")
], -1);
const _hoisted_12 = /* @__PURE__ */ createBaseVNode("i", null, "Function", -1);
const _hoisted_13 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_14 = /* @__PURE__ */ createBaseVNode("code", null, "AbstractArray{T}", -1);
const _hoisted_15 = /* @__PURE__ */ createBaseVNode("code", null, "size", -1);
const _hoisted_16 = {
  class: "MathJax",
  jax: "SVG",
  style: { "direction": "ltr", "position": "relative" }
};
const _hoisted_17 = {
  style: { "overflow": "visible", "min-height": "1px", "min-width": "1px", "vertical-align": "-0.566ex" },
  xmlns: "http://www.w3.org/2000/svg",
  width: "6.612ex",
  height: "2.262ex",
  role: "img",
  focusable: "false",
  viewBox: "0 -750 2922.7 1000",
  "aria-hidden": "true"
};
const _hoisted_18 = /* @__PURE__ */ createStaticVNode("", 1);
const _hoisted_19 = [
  _hoisted_18
];
const _hoisted_20 = /* @__PURE__ */ createBaseVNode("mjx-assistive-mml", {
  unselectable: "on",
  display: "inline",
  style: { "top": "0px", "left": "0px", "clip": "rect(1px, 1px, 1px, 1px)", "-webkit-touch-callout": "none", "-webkit-user-select": "none", "-khtml-user-select": "none", "-moz-user-select": "none", "-ms-user-select": "none", "user-select": "none", "position": "absolute", "padding": "1px 0px 0px 0px", "border": "0px", "display": "block", "width": "auto", "overflow": "hidden" }
}, [
  /* @__PURE__ */ createBaseVNode("math", { xmlns: "http://www.w3.org/1998/Math/MathML" }, [
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "["),
    /* @__PURE__ */ createBaseVNode("mo", null, "−"),
    /* @__PURE__ */ createBaseVNode("mi", null, "x"),
    /* @__PURE__ */ createBaseVNode("mo", null, ","),
    /* @__PURE__ */ createBaseVNode("mi", null, "x"),
    /* @__PURE__ */ createBaseVNode("mo", { stretchy: "false" }, "]")
  ])
], -1);
const _hoisted_21 = /* @__PURE__ */ createBaseVNode("code", null, "x = gain * sqrt(6 / (fan_in + fan_out))", -1);
const _hoisted_22 = /* @__PURE__ */ createBaseVNode("p", null, [
  /* @__PURE__ */ createBaseVNode("strong", null, "References")
], -1);
const _hoisted_23 = /* @__PURE__ */ createBaseVNode("p", null, [
  /* @__PURE__ */ createTextVNode('[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." '),
  /* @__PURE__ */ createBaseVNode("em", null, "Proceedings of the thirteenth international conference on artificial intelligence and statistics"),
  /* @__PURE__ */ createTextVNode(". 2010.")
], -1);
const _hoisted_24 = /* @__PURE__ */ createBaseVNode("p", null, [
  /* @__PURE__ */ createBaseVNode("a", {
    href: "https://github.com/LuxDL/WeightInitializers.jl/blob/v0.1.7/src/initializers.jl#L22-L36",
    target: "_blank",
    rel: "noreferrer"
  }, "source")
], -1);
const _hoisted_25 = /* @__PURE__ */ createStaticVNode("", 62);
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", null, [
    _hoisted_1,
    createBaseVNode("div", _hoisted_9, [
      _hoisted_10,
      createTextVNode(" "),
      _hoisted_11,
      createTextVNode(" — "),
      _hoisted_12,
      createTextVNode(". "),
      _hoisted_13,
      createBaseVNode("p", null, [
        createTextVNode("Return an "),
        _hoisted_14,
        createTextVNode(" of the given "),
        _hoisted_15,
        createTextVNode(" containing random numbers drawn from a uniform distribution on the interval "),
        createBaseVNode("mjx-container", _hoisted_16, [
          (openBlock(), createElementBlock("svg", _hoisted_17, _hoisted_19)),
          _hoisted_20
        ]),
        createTextVNode(", where "),
        _hoisted_21,
        createTextVNode(". This method is described in [1] and also known as Xavier initialization.")
      ]),
      _hoisted_22,
      _hoisted_23,
      _hoisted_24
    ]),
    _hoisted_25
  ]);
}
const WeightInitializers = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render]]);
export {
  __pageData,
  WeightInitializers as default
};
