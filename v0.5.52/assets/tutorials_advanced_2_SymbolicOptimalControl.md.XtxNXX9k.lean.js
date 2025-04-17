import{_ as n,c as s,j as A,a as i,a3 as t,o as a}from"./chunks/framework.B5P1ePk0.js";const D2=JSON.parse('{"title":"Solving Optimal Control Problems with Symbolic Universal Differential Equations","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/advanced/2_SymbolicOptimalControl.md","filePath":"tutorials/advanced/2_SymbolicOptimalControl.md","lastUpdated":null}'),e={name:"tutorials/advanced/2_SymbolicOptimalControl.md"},l=t("",3),h={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},Q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"10.238ex",height:"2.564ex",role:"img",focusable:"false",viewBox:"0 -883.2 4525 1133.2","aria-hidden":"true"},p=t("",1),T=[p],r=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("msup",null,[A("mi",null,"x"),A("mrow",{"data-mjx-texclass":"ORD"},[A("mi",{"data-mjx-alternate":"1"},"′"),A("mi",{"data-mjx-alternate":"1"},"′")])]),A("mo",null,"="),A("msup",null,[A("mi",null,"u"),A("mn",null,"3")]),A("mo",{stretchy:"false"},"("),A("mi",null,"t"),A("mo",{stretchy:"false"},")")])],-1),d={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},k={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"3.871ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 1711 1000","aria-hidden":"true"},o=t("",1),E=[o],g=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"u"),A("mo",{stretchy:"false"},"("),A("mi",null,"t"),A("mo",{stretchy:"false"},")")])],-1),m={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},c={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.697ex"},xmlns:"http://www.w3.org/2000/svg",width:"47.568ex",height:"4.847ex",role:"img",focusable:"false",viewBox:"0 -950 21025.1 2142.2","aria-hidden":"true"},y=t("",1),C=[y],u=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mrow",{"data-mjx-texclass":"ORD"},[A("mi",{"data-mjx-variant":"-tex-calligraphic",mathvariant:"script"},"L")]),A("mo",{stretchy:"false"},"("),A("mi",null,"θ"),A("mo",{stretchy:"false"},")"),A("mo",null,"="),A("munder",null,[A("mo",{"data-mjx-texclass":"OP"},"∑"),A("mi",null,"i")]),A("mrow",{"data-mjx-texclass":"INNER"},[A("mo",{"data-mjx-texclass":"OPEN"},"("),A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mn",null,"4"),A("mo",null,"−"),A("mi",null,"x"),A("mo",{stretchy:"false"},"("),A("msub",null,[A("mi",null,"t"),A("mi",null,"i")]),A("mo",{stretchy:"false"},")"),A("msub",null,[A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mn",null,"2")]),A("mo",null,"+"),A("mn",null,"2"),A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mi",null,"x"),A("mi",{"data-mjx-alternate":"1"},"′"),A("mo",{stretchy:"false"},"("),A("msub",null,[A("mi",null,"t"),A("mi",null,"i")]),A("mo",{stretchy:"false"},")"),A("msub",null,[A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mn",null,"2")]),A("mo",null,"+"),A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mi",null,"u"),A("mo",{stretchy:"false"},"("),A("msub",null,[A("mi",null,"t"),A("mi",null,"i")]),A("mo",{stretchy:"false"},")"),A("msub",null,[A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mn",null,"2")]),A("mo",{"data-mjx-texclass":"CLOSE"},")")])])],-1),f={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},v={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"0.781ex",height:"1.52ex",role:"img",focusable:"false",viewBox:"0 -661 345 672","aria-hidden":"true"},F=A("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[A("g",{"data-mml-node":"math"},[A("g",{"data-mml-node":"mi"},[A("path",{"data-c":"1D456",d:"M184 600Q184 624 203 642T247 661Q265 661 277 649T290 619Q290 596 270 577T226 557Q211 557 198 567T184 600ZM21 287Q21 295 30 318T54 369T98 420T158 442Q197 442 223 419T250 357Q250 340 236 301T196 196T154 83Q149 61 149 51Q149 26 166 26Q175 26 185 29T208 43T235 78T260 137Q263 149 265 151T282 153Q302 153 302 143Q302 135 293 112T268 61T223 11T161 -11Q129 -11 102 10T74 74Q74 91 79 106T122 220Q160 321 166 341T173 380Q173 404 156 404H154Q124 404 99 371T61 287Q60 286 59 284T58 281T56 279T53 278T49 278T41 278H27Q21 284 21 287Z",style:{"stroke-width":"3"}})])])],-1),b=[F],V=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"i")])],-1),w={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},L={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"5.029ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 2222.7 1000","aria-hidden":"true"},I=t("",1),x=[I],M=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mo",{stretchy:"false"},"("),A("mn",null,"0"),A("mo",null,","),A("mn",null,"8"),A("mo",{stretchy:"false"},")")])],-1),H={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Z={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.05ex"},xmlns:"http://www.w3.org/2000/svg",width:"4.023ex",height:"1.557ex",role:"img",focusable:"false",viewBox:"0 -666 1778 688","aria-hidden":"true"},D=t("",1),B=[D],q=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mn",null,"0.01")])],-1),R={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},O={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.186ex"},xmlns:"http://www.w3.org/2000/svg",width:"6.036ex",height:"2.016ex",role:"img",focusable:"false",viewBox:"0 -809 2668 891","aria-hidden":"true"},z=t("",1),W=[z],U=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("msup",null,[A("mi",null,"x"),A("mi",{"data-mjx-alternate":"1"},"′")]),A("mo",null,"="),A("mi",null,"v")])],-1),X={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},K={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"9.601ex",height:"2.564ex",role:"img",focusable:"false",viewBox:"0 -883.2 4243.6 1133.2","aria-hidden":"true"},j=t("",1),P=[j],S=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("msup",null,[A("mi",null,"v"),A("mi",{"data-mjx-alternate":"1"},"′")]),A("mo",null,"="),A("msup",null,[A("mi",null,"u"),A("mn",null,"3")]),A("mo",{stretchy:"false"},"("),A("mi",null,"t"),A("mo",{stretchy:"false"},")")])],-1),J=A("p",null,"and thus",-1),N={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},G={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.697ex"},xmlns:"http://www.w3.org/2000/svg",width:"46.749ex",height:"4.847ex",role:"img",focusable:"false",viewBox:"0 -950 20663.1 2142.2","aria-hidden":"true"},Y=t("",1),_=[Y],$=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mrow",{"data-mjx-texclass":"ORD"},[A("mi",{"data-mjx-variant":"-tex-calligraphic",mathvariant:"script"},"L")]),A("mo",{stretchy:"false"},"("),A("mi",null,"θ"),A("mo",{stretchy:"false"},")"),A("mo",null,"="),A("munder",null,[A("mo",{"data-mjx-texclass":"OP"},"∑"),A("mi",null,"i")]),A("mrow",{"data-mjx-texclass":"INNER"},[A("mo",{"data-mjx-texclass":"OPEN"},"("),A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mn",null,"4"),A("mo",null,"−"),A("mi",null,"x"),A("mo",{stretchy:"false"},"("),A("msub",null,[A("mi",null,"t"),A("mi",null,"i")]),A("mo",{stretchy:"false"},")"),A("msub",null,[A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mn",null,"2")]),A("mo",null,"+"),A("mn",null,"2"),A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mi",null,"v"),A("mo",{stretchy:"false"},"("),A("msub",null,[A("mi",null,"t"),A("mi",null,"i")]),A("mo",{stretchy:"false"},")"),A("msub",null,[A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mn",null,"2")]),A("mo",null,"+"),A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mi",null,"u"),A("mo",{stretchy:"false"},"("),A("msub",null,[A("mi",null,"t"),A("mi",null,"i")]),A("mo",{stretchy:"false"},")"),A("msub",null,[A("mo",{"data-mjx-texclass":"ORD"},"∥"),A("mn",null,"2")]),A("mo",{"data-mjx-texclass":"CLOSE"},")")])])],-1),A2={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},s2={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.294ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 572 453","aria-hidden":"true"},a2=A("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[A("g",{"data-mml-node":"math"},[A("g",{"data-mml-node":"mi"},[A("path",{"data-c":"1D462",d:"M21 287Q21 295 30 318T55 370T99 420T158 442Q204 442 227 417T250 358Q250 340 216 246T182 105Q182 62 196 45T238 27T291 44T328 78L339 95Q341 99 377 247Q407 367 413 387T427 416Q444 431 463 431Q480 431 488 421T496 402L420 84Q419 79 419 68Q419 43 426 35T447 26Q469 29 482 57T512 145Q514 153 532 153Q551 153 551 144Q550 139 549 130T540 98T523 55T498 17T462 -8Q454 -10 438 -10Q372 -10 347 46Q345 45 336 36T318 21T296 6T267 -6T233 -11Q189 -11 155 7Q103 38 103 113Q103 170 138 262T173 379Q173 380 173 381Q173 390 173 393T169 400T158 404H154Q131 404 112 385T82 344T65 302T57 280Q55 278 41 278H27Q21 284 21 287Z",style:{"stroke-width":"3"}})])])],-1),t2=[a2],i2=A("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[A("mi",null,"u")])],-1),n2=t("",35),e2={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},l2={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.148ex"},xmlns:"http://www.w3.org/2000/svg",width:"56.391ex",height:"5.428ex",role:"img",focusable:"false",viewBox:"0 -1449.5 24924.7 2399","aria-hidden":"true"},h2=t("",1),Q2=[h2],p2=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mo",null,"−"),A("mn",null,"0.025893"),A("mo",null,"−"),A("mi",null,"x"),A("mn",null,"1"),A("mo",null,"+"),A("mi",null,"x"),A("mn",null,"2"),A("mo",null,"⋅"),A("mrow",{"data-mjx-texclass":"INNER"},[A("mo",{"data-mjx-texclass":"OPEN"},"("),A("mi",null,"x"),A("mn",null,"2"),A("mo",null,"+"),A("mn",null,"0.74882"),A("mo",null,"−"),A("mfrac",null,[A("mrow",null,[A("mi",null,"x"),A("mn",null,"1")]),A("mrow",null,[A("mo",null,"−"),A("mn",null,"0.80024")])]),A("mo",{"data-mjx-texclass":"CLOSE"},")")]),A("mo",null,"−"),A("mi",null,"x"),A("mn",null,"1")])],-1),T2={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},r2={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-2.148ex"},xmlns:"http://www.w3.org/2000/svg",width:"49.104ex",height:"5.428ex",role:"img",focusable:"false",viewBox:"0 -1449.5 21703.8 2399","aria-hidden":"true"},d2=t("",1),k2=[d2],o2=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mi",null,"x"),A("mn",null,"2"),A("mo",null,"⋅"),A("mrow",{"data-mjx-texclass":"INNER"},[A("mo",{"data-mjx-texclass":"OPEN"},"("),A("mfrac",null,[A("mrow",null,[A("mi",null,"x"),A("mn",null,"2")]),A("mn",null,"0.84357")]),A("mo",null,"−"),A("mn",null,"0.46141"),A("mo",null,"−"),A("mfrac",null,[A("mrow",null,[A("mi",null,"x"),A("mn",null,"1")]),A("mrow",null,[A("mo",null,"−"),A("mn",null,"0.40226")])]),A("mo",{"data-mjx-texclass":"CLOSE"},")")]),A("mo",null,"−"),A("mn",null,"0.11702")])],-1),E2={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},g2={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-1.738ex"},xmlns:"http://www.w3.org/2000/svg",width:"58.251ex",height:"4.774ex",role:"img",focusable:"false",viewBox:"0 -1342 25747.1 2110","aria-hidden":"true"},m2=t("",1),c2=[m2],y2=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mo",null,"−"),A("mn",null,"0.028922"),A("mo",null,"+"),A("mfrac",null,[A("mrow",null,[A("mi",null,"x"),A("mn",null,"2")]),A("mrow",null,[A("mn",null,"0.34909"),A("mo",null,"−"),A("mi",null,"x"),A("mn",null,"3")])]),A("mo",null,"⋅"),A("mrow",{"data-mjx-texclass":"INNER"},[A("mo",{"data-mjx-texclass":"OPEN"},"("),A("mn",null,"0.43304"),A("mo",null,"+"),A("mi",null,"x"),A("mn",null,"2"),A("mo",null,"+"),A("mi",null,"x"),A("mn",null,"3"),A("mo",{"data-mjx-texclass":"CLOSE"},")")]),A("mo",null,"−"),A("mi",null,"x"),A("mn",null,"1"),A("mo",null,"−"),A("mi",null,"x"),A("mn",null,"1")])],-1),C2={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},u2={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-1.738ex"},xmlns:"http://www.w3.org/2000/svg",width:"23.785ex",height:"4.771ex",role:"img",focusable:"false",viewBox:"0 -1341 10512.9 2109","aria-hidden":"true"},f2=t("",1),v2=[f2],F2=A("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[A("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[A("mfrac",null,[A("mrow",null,[A("mi",null,"x"),A("mn",null,"3")]),A("mrow",null,[A("mn",null,"0.12075"),A("mo",null,"−"),A("mi",null,"x"),A("mn",null,"1")])]),A("mo",null,"+"),A("mn",null,"0.14009")])],-1),b2=t("",15);function V2(w2,L2,I2,x2,M2,H2){return a(),s("div",null,[l,A("mjx-container",h,[(a(),s("svg",Q,T)),r]),A("p",null,[i("where we want to optimize our controller "),A("mjx-container",d,[(a(),s("svg",k,E)),g]),i(" such that the following is minimized:")]),A("mjx-container",m,[(a(),s("svg",c,C)),u]),A("p",null,[i("where "),A("mjx-container",f,[(a(),s("svg",v,b)),V]),i(" is measured on "),A("mjx-container",w,[(a(),s("svg",L,x)),M]),i(" at "),A("mjx-container",H,[(a(),s("svg",Z,B)),q]),i(" intervals. To do this, we rewrite the ODE in first order form:")]),A("mjx-container",R,[(a(),s("svg",O,W)),U]),A("mjx-container",X,[(a(),s("svg",K,P)),S]),J,A("mjx-container",N,[(a(),s("svg",G,_)),$]),A("p",null,[i("is our loss function on the first order system. We thus choose a neural network form for "),A("mjx-container",A2,[(a(),s("svg",s2,t2)),i2]),i(" and optimize the equation with respect to this loss. Note that we will first reduce control cost (the last term) by 10x in order to bump the network out of a local minimum. This looks like:")]),n2,A("mjx-container",e2,[(a(),s("svg",l2,Q2)),p2]),A("mjx-container",T2,[(a(),s("svg",r2,k2)),o2]),A("mjx-container",E2,[(a(),s("svg",g2,c2)),y2]),A("mjx-container",C2,[(a(),s("svg",u2,v2)),F2]),b2])}const B2=n(e,[["render",V2]]);export{D2 as __pageData,B2 as default};
