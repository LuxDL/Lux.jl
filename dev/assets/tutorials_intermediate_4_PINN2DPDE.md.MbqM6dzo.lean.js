import{_ as t,c as i,o as e,al as n,j as s,a as p}from"./chunks/framework.BCN3FD2k.js";const l="/dev/assets/pinn_nested_ad.DbXkofst.gif",_=JSON.parse('{"title":"Training a PINN on 2D PDE","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/4_PINN2DPDE.md","filePath":"tutorials/intermediate/4_PINN2DPDE.md","lastUpdated":null}'),r={name:"tutorials/intermediate/4_PINN2DPDE.md"},c={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},h={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.586ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3795.2 1000","aria-hidden":"true"},k={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},E={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.401ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3713.2 1000","aria-hidden":"true"},o={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},d={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.109ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3584.2 1000","aria-hidden":"true"};function f(g,a,y,u,m,D){return e(),i("div",null,[a[10]||(a[10]=n("",20)),s("p",null,[a[6]||(a[6]=p("We will generate some random data to train the model on. We will take data on a square spatial and temporal domain ")),s("mjx-container",c,[(e(),i("svg",h,a[0]||(a[0]=[n("",1)]))),a[1]||(a[1]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"x"),s("mo",null,"∈"),s("mo",{stretchy:"false"},"["),s("mn",null,"0"),s("mo",null,","),s("mn",null,"2"),s("mo",{stretchy:"false"},"]")])],-1))]),a[7]||(a[7]=p(", ")),s("mjx-container",k,[(e(),i("svg",E,a[2]||(a[2]=[n("",1)]))),a[3]||(a[3]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"y"),s("mo",null,"∈"),s("mo",{stretchy:"false"},"["),s("mn",null,"0"),s("mo",null,","),s("mn",null,"2"),s("mo",{stretchy:"false"},"]")])],-1))]),a[8]||(a[8]=p(", and ")),s("mjx-container",o,[(e(),i("svg",d,a[4]||(a[4]=[n("",1)]))),a[5]||(a[5]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"t"),s("mo",null,"∈"),s("mo",{stretchy:"false"},"["),s("mn",null,"0"),s("mo",null,","),s("mn",null,"2"),s("mo",{stretchy:"false"},"]")])],-1))]),a[9]||(a[9]=p(". Typically, you want to be smarter about the sampling process, but for the sake of simplicity, we will skip that."))]),a[11]||(a[11]=n("",12))])}const x=t(r,[["render",f]]);export{_ as __pageData,x as default};
