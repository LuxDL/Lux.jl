import{_ as s,c as n,o as e,al as p}from"./chunks/framework.BCN3FD2k.js";const E=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,a,r,l,f,o){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Reactant,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    MLDatasets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Statistics,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Enzyme,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    GNNGraphs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ConcreteStructs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Printf,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    OneHotArrays,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Optimisers</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; force</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(::MLDataDevices.CPUDevice) (generic function with 1 method)</span></span></code></pre></div><h2 id="Loading-Cora-Dataset" tabindex="-1">Loading Cora Dataset <a class="header-anchor" href="#Loading-Cora-Dataset" aria-label="Permalink to &quot;Loading Cora Dataset {#Loading-Cora-Dataset}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Cora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">graphs[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gnngraph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GNNGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_index; ndata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data, edata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_data, gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">num_nodes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">features,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">targets, data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">metadata[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;classes&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use a dense matrix here to avoid incompatibility with Reactant</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adjacency_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gnngraph)),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use this since Reactant doesn&#39;t yet support gather adjoint</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">140</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">141</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">640</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1709</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2708</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loadcora (generic function with 1 method)</span></span></code></pre></div><h2 id="Model-Definition" tabindex="-1">Model Definition <a class="header-anchor" href="#Model-Definition" aria-label="Permalink to &quot;Model Definition {#Model-Definition}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dense</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> adj</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, h_dim, out_dim; nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    layer_sizes </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, [h_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nb_layers])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn_layers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (in_dim, out_dim) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zip</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)], layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    last_layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dropout)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; gcn_layers, dropout, last_layer) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj, mask)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gcn_layers</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> relu</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> last_layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj))[:, mask]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>GCN (generic function with 1 method)</span></span></code></pre></div><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y, adj, mask))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj, mask), ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; agg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">mean, logits</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))(y_pred, y[:, mask])</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st, (; y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    hidden_dim</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    use_bias</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.001</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    weight_decay</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    patience</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    epochs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">200</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">seed!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    features, targets, adj, (train_idx, val_idx, test_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(features, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), hidden_dim, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); nb_layers, dropout, use_bias)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, gcn))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    opt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(weight_decay) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lr) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AdamW</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; eta</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lr, lambda</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">weight_decay)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gcn, ps, st, opt)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Total Trainable Parameters: %0.4f M</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ps) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_loss_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gcn, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st), (features, targets, adj, val_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, train_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, val_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Inf</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">epochs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss_function,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            return_gradients</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                train_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, train_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            val_loss_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                gcn,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                (features, targets, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                val_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, val_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch %3d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Val Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 Val Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss train_acc val_loss val_acc</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> best_loss_val</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> patience</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Early Stopping at Epoch %d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                break</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gcn,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, test_idx],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Test Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_loss test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-03-17 00:37:34.169643: I external/xla/xla/service/service.cc:152] XLA service 0x1b13000 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-17 00:37:34.169821: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1742171854.170733 3822921 se_gpu_pjrt_client.cc:951] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1742171854.170827 3822921 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1742171854.170878 3822921 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1742171854.187940 3822921 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-7/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-7/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:344</span></span>
<span class="line"><span>2025-03-17 00:38:44.287958: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 16 bytes spill stores, 16 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:44.302503: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:44.361068: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 336 bytes spill stores, 336 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:44.364970: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:44.650678: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22_0&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:44.664731: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 1176 bytes spill stores, 1148 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:44.703658: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 320 bytes spill stores, 320 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:45.342678: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:46.104283: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:46.526826: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_29&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:46.578016: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 128 bytes spill stores, 128 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:46.652098: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:46.662241: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:47.092046: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:38:47.658122: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 648 bytes spill stores, 652 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1742171927.800735 3822921 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1742171927.802062 3822921 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1742171927.802071 3822921 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1742171927.802078 3822921 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1742171927.802085 3822921 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1742171927.802092 3822921 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1742171927.802100 3822921 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1742171927.802107 3822921 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1742171927.802113 3822921 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1742171927.802120 3822921 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-17 00:38:47.802135: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.804988 3822921 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1742171927.805018 3822921 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1742171927.805026 3822921 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1742171927.805032 3822921 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1742171927.805039 3822921 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1742171927.805046 3822921 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1742171927.805052 3822921 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1742171927.805058 3822921 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1742171927.805065 3822921 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1742171927.805071 3822921 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-17 00:38:47.805081: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.807568 3822921 buffer_comparator.cc:156] Difference at 16: 0, expected 15.1227</span></span>
<span class="line"><span>E0000 00:00:1742171927.807582 3822921 buffer_comparator.cc:156] Difference at 17: 0, expected 16.8521</span></span>
<span class="line"><span>E0000 00:00:1742171927.807585 3822921 buffer_comparator.cc:156] Difference at 18: 0, expected 15.6849</span></span>
<span class="line"><span>E0000 00:00:1742171927.807588 3822921 buffer_comparator.cc:156] Difference at 19: 0, expected 14.812</span></span>
<span class="line"><span>E0000 00:00:1742171927.807591 3822921 buffer_comparator.cc:156] Difference at 20: 0, expected 15.2592</span></span>
<span class="line"><span>E0000 00:00:1742171927.807594 3822921 buffer_comparator.cc:156] Difference at 21: 0, expected 14.5894</span></span>
<span class="line"><span>E0000 00:00:1742171927.807597 3822921 buffer_comparator.cc:156] Difference at 22: 0, expected 14.5711</span></span>
<span class="line"><span>E0000 00:00:1742171927.807600 3822921 buffer_comparator.cc:156] Difference at 23: 0, expected 16.9508</span></span>
<span class="line"><span>E0000 00:00:1742171927.807602 3822921 buffer_comparator.cc:156] Difference at 24: 0, expected 13.0889</span></span>
<span class="line"><span>E0000 00:00:1742171927.807605 3822921 buffer_comparator.cc:156] Difference at 25: 0, expected 13.4166</span></span>
<span class="line"><span>2025-03-17 00:38:47.807610: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.809821 3822921 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1742171927.809834 3822921 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1742171927.809837 3822921 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1742171927.809840 3822921 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1742171927.809843 3822921 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1742171927.809846 3822921 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1742171927.809849 3822921 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1742171927.809852 3822921 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1742171927.809854 3822921 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1742171927.809857 3822921 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-03-17 00:38:47.809863: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.812083 3822921 buffer_comparator.cc:156] Difference at 32: 0, expected 11.7299</span></span>
<span class="line"><span>E0000 00:00:1742171927.812099 3822921 buffer_comparator.cc:156] Difference at 33: 0, expected 13.0246</span></span>
<span class="line"><span>E0000 00:00:1742171927.812102 3822921 buffer_comparator.cc:156] Difference at 34: 0, expected 16.3887</span></span>
<span class="line"><span>E0000 00:00:1742171927.812105 3822921 buffer_comparator.cc:156] Difference at 35: 0, expected 14.7066</span></span>
<span class="line"><span>E0000 00:00:1742171927.812108 3822921 buffer_comparator.cc:156] Difference at 36: 0, expected 15.7035</span></span>
<span class="line"><span>E0000 00:00:1742171927.812111 3822921 buffer_comparator.cc:156] Difference at 37: 0, expected 14.6737</span></span>
<span class="line"><span>E0000 00:00:1742171927.812114 3822921 buffer_comparator.cc:156] Difference at 38: 0, expected 14.1379</span></span>
<span class="line"><span>E0000 00:00:1742171927.812117 3822921 buffer_comparator.cc:156] Difference at 39: 0, expected 13.6616</span></span>
<span class="line"><span>E0000 00:00:1742171927.812120 3822921 buffer_comparator.cc:156] Difference at 40: 0, expected 15.3496</span></span>
<span class="line"><span>E0000 00:00:1742171927.812123 3822921 buffer_comparator.cc:156] Difference at 41: 0, expected 13.7427</span></span>
<span class="line"><span>2025-03-17 00:38:47.812127: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.814362 3822921 buffer_comparator.cc:156] Difference at 0: 16.5257, expected 14.4011</span></span>
<span class="line"><span>E0000 00:00:1742171927.814377 3822921 buffer_comparator.cc:156] Difference at 1: 19.4064, expected 15.9904</span></span>
<span class="line"><span>E0000 00:00:1742171927.814381 3822921 buffer_comparator.cc:156] Difference at 2: 16.1909, expected 13.4103</span></span>
<span class="line"><span>E0000 00:00:1742171927.814384 3822921 buffer_comparator.cc:156] Difference at 6: 13.1689, expected 11.4953</span></span>
<span class="line"><span>E0000 00:00:1742171927.814387 3822921 buffer_comparator.cc:156] Difference at 9: 16.2882, expected 14.2452</span></span>
<span class="line"><span>E0000 00:00:1742171927.814390 3822921 buffer_comparator.cc:156] Difference at 11: 15.6385, expected 13.739</span></span>
<span class="line"><span>E0000 00:00:1742171927.814393 3822921 buffer_comparator.cc:156] Difference at 12: 20.6748, expected 16.297</span></span>
<span class="line"><span>E0000 00:00:1742171927.814396 3822921 buffer_comparator.cc:156] Difference at 13: 17.2352, expected 14.372</span></span>
<span class="line"><span>E0000 00:00:1742171927.814399 3822921 buffer_comparator.cc:156] Difference at 14: 14.761, expected 12.4213</span></span>
<span class="line"><span>E0000 00:00:1742171927.814402 3822921 buffer_comparator.cc:156] Difference at 16: 17.262, expected 15.1227</span></span>
<span class="line"><span>2025-03-17 00:38:47.814407: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.816609 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1742171927.816624 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1742171927.816627 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1742171927.816630 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1742171927.816633 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1742171927.816636 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1742171927.816639 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1742171927.816642 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1742171927.816644 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1742171927.816647 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-17 00:38:47.816652: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.818849 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1742171927.818865 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1742171927.818870 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1742171927.818873 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1742171927.818876 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1742171927.818878 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1742171927.818881 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1742171927.818884 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1742171927.818887 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1742171927.818890 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-17 00:38:47.818895: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.821094 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 12.8482</span></span>
<span class="line"><span>E0000 00:00:1742171927.821109 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 13.1492</span></span>
<span class="line"><span>E0000 00:00:1742171927.821112 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 14.2367</span></span>
<span class="line"><span>E0000 00:00:1742171927.821115 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 15.1583</span></span>
<span class="line"><span>E0000 00:00:1742171927.821118 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 15.2143</span></span>
<span class="line"><span>E0000 00:00:1742171927.821121 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 14.9801</span></span>
<span class="line"><span>E0000 00:00:1742171927.821124 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 15.6116</span></span>
<span class="line"><span>E0000 00:00:1742171927.821127 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 13.4204</span></span>
<span class="line"><span>E0000 00:00:1742171927.821130 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 13.3979</span></span>
<span class="line"><span>E0000 00:00:1742171927.821133 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 14.1923</span></span>
<span class="line"><span>2025-03-17 00:38:47.821137: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.823348 3822921 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1742171927.823361 3822921 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1742171927.823365 3822921 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1742171927.823368 3822921 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1742171927.823371 3822921 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1742171927.823373 3822921 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1742171927.823376 3822921 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1742171927.823379 3822921 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1742171927.823382 3822921 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1742171927.823385 3822921 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-17 00:38:47.823390: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.825593 3822921 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1742171927.825608 3822921 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1742171927.825611 3822921 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1742171927.825614 3822921 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1742171927.825617 3822921 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1742171927.825620 3822921 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1742171927.825623 3822921 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1742171927.825626 3822921 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1742171927.825629 3822921 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1742171927.825632 3822921 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-17 00:38:47.825637: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.827852 3822921 buffer_comparator.cc:156] Difference at 128: 0, expected 14.3208</span></span>
<span class="line"><span>E0000 00:00:1742171927.827867 3822921 buffer_comparator.cc:156] Difference at 129: 0, expected 14.1375</span></span>
<span class="line"><span>E0000 00:00:1742171927.827870 3822921 buffer_comparator.cc:156] Difference at 130: 0, expected 16.2788</span></span>
<span class="line"><span>E0000 00:00:1742171927.827873 3822921 buffer_comparator.cc:156] Difference at 131: 0, expected 14.6417</span></span>
<span class="line"><span>E0000 00:00:1742171927.827875 3822921 buffer_comparator.cc:156] Difference at 132: 0, expected 12.8658</span></span>
<span class="line"><span>E0000 00:00:1742171927.827878 3822921 buffer_comparator.cc:156] Difference at 133: 0, expected 11.2575</span></span>
<span class="line"><span>E0000 00:00:1742171927.827881 3822921 buffer_comparator.cc:156] Difference at 134: 0, expected 13.3003</span></span>
<span class="line"><span>E0000 00:00:1742171927.827884 3822921 buffer_comparator.cc:156] Difference at 135: 0, expected 14.1279</span></span>
<span class="line"><span>E0000 00:00:1742171927.827887 3822921 buffer_comparator.cc:156] Difference at 136: 0, expected 13.7627</span></span>
<span class="line"><span>E0000 00:00:1742171927.827890 3822921 buffer_comparator.cc:156] Difference at 137: 0, expected 12.9584</span></span>
<span class="line"><span>2025-03-17 00:38:47.827894: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.830115 3822921 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1742171927.830130 3822921 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1742171927.830134 3822921 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1742171927.830137 3822921 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1742171927.830139 3822921 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1742171927.830142 3822921 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1742171927.830145 3822921 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1742171927.830148 3822921 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1742171927.830151 3822921 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1742171927.830154 3822921 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-17 00:38:47.830158: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.832374 3822921 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1742171927.832387 3822921 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1742171927.832391 3822921 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1742171927.832394 3822921 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1742171927.832396 3822921 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1742171927.832399 3822921 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1742171927.832402 3822921 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1742171927.832405 3822921 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1742171927.832408 3822921 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1742171927.832411 3822921 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-17 00:38:47.832416: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.834613 3822921 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1742171927.834628 3822921 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1742171927.834631 3822921 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1742171927.834634 3822921 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1742171927.834637 3822921 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1742171927.834640 3822921 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1742171927.834643 3822921 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1742171927.834645 3822921 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1742171927.834648 3822921 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1742171927.834651 3822921 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-17 00:38:47.834656: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.836869 3822921 buffer_comparator.cc:156] Difference at 256: 0, expected 11.4506</span></span>
<span class="line"><span>E0000 00:00:1742171927.836881 3822921 buffer_comparator.cc:156] Difference at 257: 0, expected 13.0911</span></span>
<span class="line"><span>E0000 00:00:1742171927.836884 3822921 buffer_comparator.cc:156] Difference at 258: 0, expected 11.2111</span></span>
<span class="line"><span>E0000 00:00:1742171927.836887 3822921 buffer_comparator.cc:156] Difference at 259: 0, expected 14.6856</span></span>
<span class="line"><span>E0000 00:00:1742171927.836890 3822921 buffer_comparator.cc:156] Difference at 260: 0, expected 13.7902</span></span>
<span class="line"><span>E0000 00:00:1742171927.836893 3822921 buffer_comparator.cc:156] Difference at 261: 0, expected 13.402</span></span>
<span class="line"><span>E0000 00:00:1742171927.836896 3822921 buffer_comparator.cc:156] Difference at 262: 0, expected 14.0022</span></span>
<span class="line"><span>E0000 00:00:1742171927.836899 3822921 buffer_comparator.cc:156] Difference at 263: 0, expected 11.1338</span></span>
<span class="line"><span>E0000 00:00:1742171927.836902 3822921 buffer_comparator.cc:156] Difference at 264: 0, expected 12.2916</span></span>
<span class="line"><span>E0000 00:00:1742171927.836905 3822921 buffer_comparator.cc:156] Difference at 265: 0, expected 15.766</span></span>
<span class="line"><span>2025-03-17 00:38:47.836909: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.841287 3822921 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1742171927.841302 3822921 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1742171927.841305 3822921 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1742171927.841308 3822921 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1742171927.841311 3822921 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1742171927.841314 3822921 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1742171927.841317 3822921 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1742171927.841320 3822921 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1742171927.841323 3822921 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1742171927.841326 3822921 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-03-17 00:38:47.841331: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.843856 3822921 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1742171927.843870 3822921 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1742171927.843874 3822921 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1742171927.843877 3822921 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1742171927.843880 3822921 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1742171927.843883 3822921 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1742171927.843885 3822921 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1742171927.843888 3822921 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1742171927.843891 3822921 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1742171927.843894 3822921 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-03-17 00:38:47.843899: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.846405 3822921 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1742171927.846421 3822921 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1742171927.846424 3822921 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1742171927.846427 3822921 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1742171927.846429 3822921 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1742171927.846432 3822921 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1742171927.846435 3822921 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1742171927.846438 3822921 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1742171927.846441 3822921 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1742171927.846444 3822921 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-03-17 00:38:47.846448: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.848953 3822921 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1742171927.848968 3822921 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1742171927.848971 3822921 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1742171927.848974 3822921 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1742171927.848977 3822921 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1742171927.848980 3822921 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1742171927.848983 3822921 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1742171927.848986 3822921 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1742171927.848989 3822921 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1742171927.848992 3822921 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-03-17 00:38:47.848996: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.851502 3822921 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1742171927.851517 3822921 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1742171927.851520 3822921 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1742171927.851523 3822921 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1742171927.851526 3822921 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1742171927.851529 3822921 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1742171927.851533 3822921 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1742171927.851535 3822921 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1742171927.851538 3822921 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1742171927.851541 3822921 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-03-17 00:38:47.851546: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.854096 3822921 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1742171927.854113 3822921 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1742171927.854116 3822921 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1742171927.854119 3822921 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1742171927.854122 3822921 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1742171927.854125 3822921 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1742171927.854127 3822921 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1742171927.854130 3822921 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1742171927.854133 3822921 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1742171927.854136 3822921 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-03-17 00:38:47.854141: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.856693 3822921 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1742171927.856707 3822921 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1742171927.856711 3822921 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1742171927.856714 3822921 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1742171927.856716 3822921 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1742171927.856719 3822921 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1742171927.856722 3822921 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1742171927.856725 3822921 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1742171927.856728 3822921 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1742171927.856731 3822921 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-03-17 00:38:47.856735: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.859290 3822921 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1742171927.859305 3822921 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1742171927.859309 3822921 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1742171927.859312 3822921 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1742171927.859314 3822921 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1742171927.859317 3822921 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1742171927.859320 3822921 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1742171927.859323 3822921 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1742171927.859326 3822921 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1742171927.859329 3822921 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-03-17 00:38:47.859335: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.861886 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1742171927.861901 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1742171927.861904 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1742171927.861907 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1742171927.861910 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1742171927.861913 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1742171927.861916 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1742171927.861918 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1742171927.861921 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1742171927.861924 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-17 00:38:47.861929: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.864479 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1742171927.864495 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1742171927.864498 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1742171927.864501 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1742171927.864504 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1742171927.864506 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1742171927.864509 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1742171927.864512 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1742171927.864515 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1742171927.864518 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-17 00:38:47.864523: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.867070 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1742171927.867088 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1742171927.867091 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1742171927.867094 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1742171927.867097 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1742171927.867099 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1742171927.867102 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1742171927.867105 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1742171927.867108 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1742171927.867111 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-17 00:38:47.867116: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.869679 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1742171927.869693 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1742171927.869698 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1742171927.869701 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1742171927.869703 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1742171927.869706 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1742171927.869709 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1742171927.869712 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1742171927.869715 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1742171927.869718 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-17 00:38:47.869722: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.872287 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1742171927.872303 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1742171927.872307 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1742171927.872310 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1742171927.872313 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1742171927.872315 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1742171927.872318 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1742171927.872321 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1742171927.872324 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1742171927.872327 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-17 00:38:47.872332: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.874874 3822921 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1742171927.874888 3822921 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1742171927.874891 3822921 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1742171927.874894 3822921 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1742171927.874897 3822921 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1742171927.874900 3822921 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1742171927.874903 3822921 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1742171927.874906 3822921 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1742171927.874909 3822921 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1742171927.874911 3822921 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-17 00:38:47.874916: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.877481 3822921 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1742171927.877496 3822921 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1742171927.877499 3822921 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1742171927.877502 3822921 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1742171927.877505 3822921 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1742171927.877507 3822921 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1742171927.877511 3822921 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1742171927.877514 3822921 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1742171927.877517 3822921 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1742171927.877520 3822921 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-03-17 00:38:47.877525: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.880093 3822921 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1742171927.880108 3822921 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1742171927.880111 3822921 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1742171927.880114 3822921 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1742171927.880117 3822921 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1742171927.880120 3822921 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1742171927.880123 3822921 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1742171927.880126 3822921 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1742171927.880128 3822921 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1742171927.880131 3822921 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-03-17 00:38:47.880136: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.882700 3822921 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1742171927.882713 3822921 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1742171927.882716 3822921 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1742171927.882719 3822921 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1742171927.882722 3822921 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1742171927.882725 3822921 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1742171927.882728 3822921 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1742171927.882731 3822921 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1742171927.882734 3822921 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1742171927.882737 3822921 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-03-17 00:38:47.882741: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.885305 3822921 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1742171927.885319 3822921 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1742171927.885322 3822921 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1742171927.885325 3822921 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1742171927.885328 3822921 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1742171927.885331 3822921 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1742171927.885334 3822921 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1742171927.885337 3822921 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1742171927.885339 3822921 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1742171927.885342 3822921 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-03-17 00:38:47.885348: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.887929 3822921 buffer_comparator.cc:156] Difference at 256: 0, expected 16.0393</span></span>
<span class="line"><span>E0000 00:00:1742171927.887943 3822921 buffer_comparator.cc:156] Difference at 257: 0, expected 18.4933</span></span>
<span class="line"><span>E0000 00:00:1742171927.887946 3822921 buffer_comparator.cc:156] Difference at 258: 0, expected 18.027</span></span>
<span class="line"><span>E0000 00:00:1742171927.887949 3822921 buffer_comparator.cc:156] Difference at 259: 0, expected 20.7645</span></span>
<span class="line"><span>E0000 00:00:1742171927.887952 3822921 buffer_comparator.cc:156] Difference at 260: 0, expected 18.8066</span></span>
<span class="line"><span>E0000 00:00:1742171927.887954 3822921 buffer_comparator.cc:156] Difference at 261: 0, expected 17.9486</span></span>
<span class="line"><span>E0000 00:00:1742171927.887957 3822921 buffer_comparator.cc:156] Difference at 262: 0, expected 16.8675</span></span>
<span class="line"><span>E0000 00:00:1742171927.887960 3822921 buffer_comparator.cc:156] Difference at 263: 0, expected 18.7938</span></span>
<span class="line"><span>E0000 00:00:1742171927.887963 3822921 buffer_comparator.cc:156] Difference at 264: 0, expected 16.5109</span></span>
<span class="line"><span>E0000 00:00:1742171927.887966 3822921 buffer_comparator.cc:156] Difference at 265: 0, expected 20.2758</span></span>
<span class="line"><span>2025-03-17 00:38:47.887971: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.890555 3822921 buffer_comparator.cc:156] Difference at 256: 0, expected 16.0393</span></span>
<span class="line"><span>E0000 00:00:1742171927.890569 3822921 buffer_comparator.cc:156] Difference at 257: 0, expected 18.4933</span></span>
<span class="line"><span>E0000 00:00:1742171927.890572 3822921 buffer_comparator.cc:156] Difference at 258: 0, expected 18.027</span></span>
<span class="line"><span>E0000 00:00:1742171927.890575 3822921 buffer_comparator.cc:156] Difference at 259: 0, expected 20.7645</span></span>
<span class="line"><span>E0000 00:00:1742171927.890578 3822921 buffer_comparator.cc:156] Difference at 260: 0, expected 18.8066</span></span>
<span class="line"><span>E0000 00:00:1742171927.890581 3822921 buffer_comparator.cc:156] Difference at 261: 0, expected 17.9486</span></span>
<span class="line"><span>E0000 00:00:1742171927.890584 3822921 buffer_comparator.cc:156] Difference at 262: 0, expected 16.8675</span></span>
<span class="line"><span>E0000 00:00:1742171927.890587 3822921 buffer_comparator.cc:156] Difference at 263: 0, expected 18.7938</span></span>
<span class="line"><span>E0000 00:00:1742171927.890589 3822921 buffer_comparator.cc:156] Difference at 264: 0, expected 16.5109</span></span>
<span class="line"><span>E0000 00:00:1742171927.890592 3822921 buffer_comparator.cc:156] Difference at 265: 0, expected 20.2758</span></span>
<span class="line"><span>2025-03-17 00:38:47.890597: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.902882 3822921 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1742171927.902923 3822921 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1742171927.902926 3822921 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1742171927.902929 3822921 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1742171927.902933 3822921 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1742171927.902936 3822921 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1742171927.902939 3822921 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171927.902942 3822921 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1742171927.902945 3822921 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1742171927.902948 3822921 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-17 00:38:47.902958: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.906124 3822921 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1742171927.906140 3822921 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1742171927.906143 3822921 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1742171927.906146 3822921 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1742171927.906149 3822921 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1742171927.906152 3822921 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1742171927.906155 3822921 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171927.906158 3822921 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1742171927.906161 3822921 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1742171927.906164 3822921 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-17 00:38:47.906169: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.909305 3822921 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1742171927.909319 3822921 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1742171927.909322 3822921 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1742171927.909325 3822921 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1742171927.909328 3822921 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1742171927.909331 3822921 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1742171927.909335 3822921 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171927.909338 3822921 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1742171927.909341 3822921 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1742171927.909344 3822921 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-17 00:38:47.909348: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.912486 3822921 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1742171927.912499 3822921 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1742171927.912502 3822921 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1742171927.912506 3822921 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1742171927.912509 3822921 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1742171927.912512 3822921 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1742171927.912515 3822921 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171927.912518 3822921 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1742171927.912521 3822921 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1742171927.912524 3822921 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-17 00:38:47.912529: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.915704 3822921 buffer_comparator.cc:156] Difference at 0: 1139.71, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1742171927.915717 3822921 buffer_comparator.cc:156] Difference at 1: 1404.8, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1742171927.915720 3822921 buffer_comparator.cc:156] Difference at 2: 2132.23, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1742171927.915725 3822921 buffer_comparator.cc:156] Difference at 3: 1838.84, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1742171927.915728 3822921 buffer_comparator.cc:156] Difference at 4: 1307.39, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1742171927.915731 3822921 buffer_comparator.cc:156] Difference at 5: 2064.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1742171927.915734 3822921 buffer_comparator.cc:156] Difference at 6: 1480.82, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1742171927.915737 3822921 buffer_comparator.cc:156] Difference at 7: 1113.19, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1742171927.915740 3822921 buffer_comparator.cc:156] Difference at 8: 1358.65, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1742171927.915743 3822921 buffer_comparator.cc:156] Difference at 9: 2048.24, expected 1833.76</span></span>
<span class="line"><span>2025-03-17 00:38:47.915748: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.918925 3822921 buffer_comparator.cc:156] Difference at 112: 1194.18, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1742171927.918941 3822921 buffer_comparator.cc:156] Difference at 113: 1041.48, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1742171927.918944 3822921 buffer_comparator.cc:156] Difference at 114: 725.367, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1742171927.918947 3822921 buffer_comparator.cc:156] Difference at 115: 1163.29, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1742171927.918950 3822921 buffer_comparator.cc:156] Difference at 116: 837.371, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1742171927.918953 3822921 buffer_comparator.cc:156] Difference at 117: 617.301, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1742171927.918956 3822921 buffer_comparator.cc:156] Difference at 118: 781.352, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171927.918959 3822921 buffer_comparator.cc:156] Difference at 119: 1181.95, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1742171927.918963 3822921 buffer_comparator.cc:156] Difference at 120: 1033.27, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1742171927.918966 3822921 buffer_comparator.cc:156] Difference at 121: 727.552, expected 1820.15</span></span>
<span class="line"><span>2025-03-17 00:38:47.918970: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.922144 3822921 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1742171927.922159 3822921 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1742171927.922163 3822921 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1742171927.922166 3822921 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1742171927.922169 3822921 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1742171927.922172 3822921 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1742171927.922175 3822921 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1742171927.922178 3822921 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1742171927.922181 3822921 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1742171927.922184 3822921 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-03-17 00:38:47.922189: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.925306 3822921 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1742171927.925321 3822921 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1742171927.925324 3822921 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1742171927.925327 3822921 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1742171927.925330 3822921 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1742171927.925335 3822921 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1742171927.925338 3822921 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1742171927.925341 3822921 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1742171927.925344 3822921 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1742171927.925347 3822921 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-03-17 00:38:47.925352: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.928481 3822921 buffer_comparator.cc:156] Difference at 224: 1185.44, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1742171927.928495 3822921 buffer_comparator.cc:156] Difference at 225: 1033.21, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1742171927.928498 3822921 buffer_comparator.cc:156] Difference at 226: 723.209, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1742171927.928501 3822921 buffer_comparator.cc:156] Difference at 227: 1155.3, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1742171927.928504 3822921 buffer_comparator.cc:156] Difference at 228: 842.032, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1742171927.928507 3822921 buffer_comparator.cc:156] Difference at 229: 632.011, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1742171927.928510 3822921 buffer_comparator.cc:156] Difference at 230: 809.938, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1742171927.928513 3822921 buffer_comparator.cc:156] Difference at 231: 1217.57, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1742171927.928516 3822921 buffer_comparator.cc:156] Difference at 232: 1063.63, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1742171927.928519 3822921 buffer_comparator.cc:156] Difference at 233: 740.205, expected 1803.13</span></span>
<span class="line"><span>2025-03-17 00:38:47.928524: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.931737 3822921 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1742171927.931750 3822921 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171927.931753 3822921 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1742171927.931756 3822921 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1742171927.931759 3822921 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171927.931762 3822921 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1742171927.931765 3822921 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1742171927.931768 3822921 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1742171927.931771 3822921 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1742171927.931775 3822921 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-17 00:38:47.931779: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.934890 3822921 buffer_comparator.cc:156] Difference at 0: 1057.27, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1742171927.934903 3822921 buffer_comparator.cc:156] Difference at 1: 1319.15, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1742171927.934906 3822921 buffer_comparator.cc:156] Difference at 2: 2004.43, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1742171927.934909 3822921 buffer_comparator.cc:156] Difference at 3: 1745.74, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1742171927.934912 3822921 buffer_comparator.cc:156] Difference at 4: 1252.2, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1742171927.934915 3822921 buffer_comparator.cc:156] Difference at 7: 1175.57, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1742171927.934918 3822921 buffer_comparator.cc:156] Difference at 8: 1398.75, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1742171927.934922 3822921 buffer_comparator.cc:156] Difference at 9: 2125.62, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1742171927.934925 3822921 buffer_comparator.cc:156] Difference at 10: 1878.38, expected 1592.37</span></span>
<span class="line"><span>E0000 00:00:1742171927.934928 3822921 buffer_comparator.cc:156] Difference at 11: 1362.67, expected 1121.95</span></span>
<span class="line"><span>2025-03-17 00:38:47.934933: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.938096 3822921 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1742171927.938112 3822921 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171927.938115 3822921 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1742171927.938118 3822921 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1742171927.938121 3822921 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171927.938124 3822921 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1742171927.938127 3822921 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1742171927.938130 3822921 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1742171927.938133 3822921 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1742171927.938136 3822921 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-17 00:38:47.938142: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.941303 3822921 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1742171927.941317 3822921 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171927.941321 3822921 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1742171927.941324 3822921 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1742171927.941327 3822921 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171927.941330 3822921 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1742171927.941333 3822921 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1742171927.941336 3822921 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1742171927.941339 3822921 buffer_comparator.cc:156] Difference at 456: 1044.81, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1742171927.941342 3822921 buffer_comparator.cc:156] Difference at 457: 732.124, expected 1821.28</span></span>
<span class="line"><span>2025-03-17 00:38:47.941347: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171927.944508 3822921 buffer_comparator.cc:156] Difference at 448: 1213.2, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1742171927.944521 3822921 buffer_comparator.cc:156] Difference at 449: 1055.43, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171927.944525 3822921 buffer_comparator.cc:156] Difference at 450: 735.576, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1742171927.944528 3822921 buffer_comparator.cc:156] Difference at 451: 1184.55, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1742171927.944531 3822921 buffer_comparator.cc:156] Difference at 452: 859.094, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171927.944534 3822921 buffer_comparator.cc:156] Difference at 453: 619.996, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1742171927.944537 3822921 buffer_comparator.cc:156] Difference at 454: 795.493, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1742171927.944540 3822921 buffer_comparator.cc:156] Difference at 455: 1199.74, expected 952.925</span></span>
<span class="line"><span>Epoch   1	Train Loss: 16.004427	Train Acc: 21.4286%	Val Loss: 6.836986	Val Acc: 25.8000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 8.644901	Train Acc: 27.1429%	Val Loss: 2.752696	Val Acc: 31.6000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 3.662117	Train Acc: 45.7143%	Val Loss: 1.780521	Val Acc: 42.4000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 1.632031	Train Acc: 56.4286%	Val Loss: 2.011105	Val Acc: 42.0000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.480304	Train Acc: 60.0000%	Val Loss: 2.148093	Val Acc: 42.2000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.414597	Train Acc: 66.4286%	Val Loss: 1.972278	Val Acc: 47.8000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.181894	Train Acc: 71.4286%	Val Loss: 1.805615	Val Acc: 54.6000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 1.005208	Train Acc: 74.2857%	Val Loss: 1.698273	Val Acc: 57.6000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 0.887507	Train Acc: 77.1429%	Val Loss: 1.630490	Val Acc: 60.2000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 0.808686	Train Acc: 77.8571%	Val Loss: 1.575487	Val Acc: 62.8000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 0.736529	Train Acc: 79.2857%	Val Loss: 1.528697	Val Acc: 63.2000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.682696	Train Acc: 82.8571%	Val Loss: 1.513860	Val Acc: 64.4000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.660610	Train Acc: 83.5714%	Val Loss: 1.540092	Val Acc: 64.4000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.620222	Train Acc: 83.5714%	Val Loss: 1.582344	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.571962	Train Acc: 83.5714%	Val Loss: 1.623667	Val Acc: 64.8000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.529198	Train Acc: 84.2857%	Val Loss: 1.656683	Val Acc: 65.0000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.510554	Train Acc: 84.2857%	Val Loss: 1.673672	Val Acc: 65.4000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.490537	Train Acc: 85.7143%	Val Loss: 1.676123	Val Acc: 66.2000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.462381	Train Acc: 86.4286%	Val Loss: 1.668223	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.434287	Train Acc: 87.1429%	Val Loss: 1.664831	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.416595	Train Acc: 87.1429%	Val Loss: 1.669661	Val Acc: 66.8000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.402821	Train Acc: 87.8571%	Val Loss: 1.682566	Val Acc: 67.0000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.390766	Train Acc: 87.8571%	Val Loss: 1.704447	Val Acc: 66.6000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.378771	Train Acc: 89.2857%	Val Loss: 1.736657	Val Acc: 66.2000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.365436	Train Acc: 89.2857%	Val Loss: 1.776982	Val Acc: 65.2000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.352554	Train Acc: 88.5714%	Val Loss: 1.819225	Val Acc: 64.4000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.341472	Train Acc: 90.0000%	Val Loss: 1.858865	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.332301	Train Acc: 90.0000%	Val Loss: 1.890365	Val Acc: 63.2000%</span></span>
<span class="line"><span>Epoch  29	Train Loss: 0.324014	Train Acc: 90.0000%	Val Loss: 1.911830	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  30	Train Loss: 0.315939	Train Acc: 91.4286%	Val Loss: 1.920803	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  31	Train Loss: 0.307601	Train Acc: 91.4286%	Val Loss: 1.919108	Val Acc: 64.0000%</span></span>
<span class="line"><span>Epoch  32	Train Loss: 0.299059	Train Acc: 93.5714%	Val Loss: 1.909953	Val Acc: 65.4000%</span></span>
<span class="line"><span>Early Stopping at Epoch 32</span></span>
<span class="line"><span>2025-03-17 00:39:37.526645: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:39:37.695895: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:39:37.699121: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1742171977.705489 3822921 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1742171977.705541 3822921 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1742171977.705549 3822921 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1742171977.705556 3822921 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.705563 3822921 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1742171977.705570 3822921 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1742171977.705578 3822921 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.705584 3822921 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1742171977.705591 3822921 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.705598 3822921 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-17 00:39:37.705612: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.708265 3822921 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1742171977.708292 3822921 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1742171977.708300 3822921 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1742171977.708307 3822921 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.708313 3822921 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1742171977.708320 3822921 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1742171977.708327 3822921 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.708334 3822921 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1742171977.708341 3822921 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.708347 3822921 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-17 00:39:37.708358: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.710978 3822921 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1742171977.710990 3822921 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1742171977.710993 3822921 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1742171977.710996 3822921 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.710999 3822921 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1742171977.711003 3822921 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1742171977.711006 3822921 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.711009 3822921 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1742171977.711013 3822921 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.711017 3822921 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-17 00:39:37.711021: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.713346 3822921 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1742171977.713358 3822921 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1742171977.713361 3822921 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1742171977.713364 3822921 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.713367 3822921 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1742171977.713371 3822921 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1742171977.713374 3822921 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.713377 3822921 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1742171977.713380 3822921 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.713383 3822921 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-17 00:39:37.713387: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.715725 3822921 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1742171977.715738 3822921 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1742171977.715741 3822921 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1742171977.715744 3822921 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.715747 3822921 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1742171977.715750 3822921 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1742171977.715753 3822921 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.715756 3822921 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1742171977.715759 3822921 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.715762 3822921 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-17 00:39:37.715767: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.718123 3822921 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1742171977.718135 3822921 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1742171977.718138 3822921 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1742171977.718141 3822921 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.718144 3822921 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1742171977.718147 3822921 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1742171977.718150 3822921 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.718153 3822921 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1742171977.718156 3822921 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.718159 3822921 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-17 00:39:37.718166: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.720474 3822921 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1742171977.720486 3822921 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1742171977.720489 3822921 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1742171977.720492 3822921 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1742171977.720495 3822921 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1742171977.720498 3822921 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1742171977.720501 3822921 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1742171977.720504 3822921 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1742171977.720507 3822921 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1742171977.720510 3822921 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-03-17 00:39:37.720515: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.722807 3822921 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1742171977.722821 3822921 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1742171977.722824 3822921 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1742171977.722827 3822921 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1742171977.722830 3822921 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1742171977.722833 3822921 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1742171977.722836 3822921 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1742171977.722839 3822921 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1742171977.722842 3822921 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1742171977.722845 3822921 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-17 00:39:37.722850: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.725166 3822921 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1742171977.725178 3822921 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1742171977.725181 3822921 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1742171977.725184 3822921 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1742171977.725187 3822921 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1742171977.725190 3822921 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1742171977.725193 3822921 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1742171977.725196 3822921 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1742171977.725199 3822921 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1742171977.725202 3822921 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-17 00:39:37.725207: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.727528 3822921 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171977.727541 3822921 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1742171977.727544 3822921 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1742171977.727547 3822921 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171977.727550 3822921 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1742171977.727554 3822921 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1742171977.727557 3822921 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.727560 3822921 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.727563 3822921 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1742171977.727566 3822921 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-17 00:39:37.727570: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.729842 3822921 buffer_comparator.cc:156] Difference at 7: 1058.92, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1742171977.729855 3822921 buffer_comparator.cc:156] Difference at 11: 1263.92, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1742171977.729859 3822921 buffer_comparator.cc:156] Difference at 179: 1223.75, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1742171977.729863 3822921 buffer_comparator.cc:156] Difference at 266: 1047.35, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1742171977.729866 3822921 buffer_comparator.cc:156] Difference at 270: 1246.8, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1742171977.729870 3822921 buffer_comparator.cc:156] Difference at 417: 1222.47, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1742171977.729873 3822921 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171977.729876 3822921 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1742171977.729879 3822921 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1742171977.729882 3822921 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-03-17 00:39:37.729887: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.732175 3822921 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171977.732188 3822921 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1742171977.732191 3822921 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1742171977.732194 3822921 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171977.732197 3822921 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1742171977.732200 3822921 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1742171977.732203 3822921 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.732206 3822921 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.732209 3822921 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1742171977.732212 3822921 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-17 00:39:37.732217: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.734516 3822921 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171977.734531 3822921 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1742171977.734534 3822921 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1742171977.734537 3822921 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171977.734540 3822921 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1742171977.734543 3822921 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1742171977.734546 3822921 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.734549 3822921 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.734552 3822921 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1742171977.734555 3822921 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-17 00:39:37.734560: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.736842 3822921 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171977.736854 3822921 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1742171977.736857 3822921 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1742171977.736860 3822921 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171977.736863 3822921 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1742171977.736866 3822921 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1742171977.736869 3822921 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.736872 3822921 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.736875 3822921 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1742171977.736878 3822921 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-17 00:39:37.736883: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.739172 3822921 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1742171977.739184 3822921 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1742171977.739188 3822921 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1742171977.739191 3822921 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1742171977.739194 3822921 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1742171977.739197 3822921 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1742171977.739200 3822921 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.739203 3822921 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1742171977.739206 3822921 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1742171977.739209 3822921 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-17 00:39:37.739214: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.741572 3822921 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1742171977.741584 3822921 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1742171977.741587 3822921 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1742171977.741592 3822921 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1742171977.741595 3822921 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1742171977.741598 3822921 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.741601 3822921 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1742171977.741604 3822921 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1742171977.741607 3822921 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1742171977.741610 3822921 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-17 00:39:37.741615: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.744000 3822921 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1742171977.744013 3822921 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1742171977.744016 3822921 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1742171977.744020 3822921 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1742171977.744023 3822921 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1742171977.744026 3822921 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.744029 3822921 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1742171977.744032 3822921 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1742171977.744035 3822921 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1742171977.744038 3822921 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-17 00:39:37.744043: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.746352 3822921 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1742171977.746364 3822921 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1742171977.746367 3822921 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1742171977.746371 3822921 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1742171977.746374 3822921 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1742171977.746377 3822921 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.746380 3822921 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1742171977.746383 3822921 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1742171977.746386 3822921 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1742171977.746389 3822921 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-17 00:39:37.746393: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.748697 3822921 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1742171977.748710 3822921 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1742171977.748713 3822921 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1742171977.748716 3822921 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1742171977.748719 3822921 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1742171977.748724 3822921 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1742171977.748727 3822921 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1742171977.748730 3822921 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1742171977.748733 3822921 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1742171977.748736 3822921 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-03-17 00:39:37.748741: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.751183 3822921 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1742171977.751196 3822921 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1742171977.751199 3822921 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1742171977.751202 3822921 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.751205 3822921 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.751208 3822921 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1742171977.751211 3822921 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1742171977.751214 3822921 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1742171977.751217 3822921 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.751221 3822921 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-17 00:39:37.751225: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.753660 3822921 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1742171977.753672 3822921 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1742171977.753675 3822921 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1742171977.753678 3822921 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.753681 3822921 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.753684 3822921 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1742171977.753687 3822921 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1742171977.753691 3822921 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1742171977.753694 3822921 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.753697 3822921 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-17 00:39:37.753703: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171977.756112 3822921 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1742171977.756125 3822921 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1742171977.756128 3822921 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1742171977.756131 3822921 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.756134 3822921 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1742171977.756137 3822921 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1742171977.756140 3822921 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1742171977.756145 3822921 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1742171977.756148 3822921 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1742171977.756151 3822921 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-17 00:39:37.756155: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-03-17 00:39:39.066865: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:39:39.566643: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-17 00:39:39.698114: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1742171979.705356 3822921 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1742171979.705422 3822921 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1742171979.705430 3822921 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1742171979.705437 3822921 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.705444 3822921 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1742171979.705451 3822921 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1742171979.705457 3822921 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1742171979.705464 3822921 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1742171979.705470 3822921 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.705477 3822921 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-17 00:39:39.705491: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.709183 3822921 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1742171979.709212 3822921 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1742171979.709219 3822921 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1742171979.709226 3822921 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.709233 3822921 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1742171979.709239 3822921 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1742171979.709246 3822921 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1742171979.709252 3822921 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1742171979.709259 3822921 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.709265 3822921 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-17 00:39:39.709276: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.712929 3822921 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1742171979.712953 3822921 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1742171979.712957 3822921 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1742171979.712960 3822921 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.712964 3822921 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1742171979.712967 3822921 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1742171979.712970 3822921 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1742171979.712973 3822921 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1742171979.712975 3822921 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.712978 3822921 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-17 00:39:39.712983: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.716344 3822921 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1742171979.716358 3822921 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1742171979.716361 3822921 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1742171979.716364 3822921 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.716367 3822921 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1742171979.716370 3822921 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1742171979.716373 3822921 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1742171979.716376 3822921 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1742171979.716379 3822921 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.716382 3822921 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-17 00:39:39.716386: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.719747 3822921 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1742171979.719763 3822921 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1742171979.719767 3822921 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1742171979.719771 3822921 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.719774 3822921 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1742171979.719778 3822921 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1742171979.719782 3822921 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1742171979.719785 3822921 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1742171979.719789 3822921 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.719793 3822921 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-17 00:39:39.719798: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.723206 3822921 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1742171979.723225 3822921 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1742171979.723228 3822921 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1742171979.723231 3822921 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.723234 3822921 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1742171979.723237 3822921 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1742171979.723240 3822921 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1742171979.723243 3822921 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1742171979.723247 3822921 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.723250 3822921 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-17 00:39:39.723255: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.726551 3822921 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1742171979.726565 3822921 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1742171979.726569 3822921 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1742171979.726572 3822921 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1742171979.726574 3822921 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1742171979.726577 3822921 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1742171979.726580 3822921 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1742171979.726583 3822921 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1742171979.726586 3822921 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1742171979.726589 3822921 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-17 00:39:39.726594: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.729898 3822921 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1742171979.729916 3822921 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1742171979.729920 3822921 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1742171979.729923 3822921 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1742171979.729925 3822921 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1742171979.729928 3822921 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1742171979.729931 3822921 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1742171979.729934 3822921 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1742171979.729937 3822921 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1742171979.729940 3822921 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-17 00:39:39.729945: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.733293 3822921 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1742171979.733306 3822921 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1742171979.733309 3822921 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1742171979.733312 3822921 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1742171979.733315 3822921 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1742171979.733318 3822921 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1742171979.733321 3822921 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1742171979.733324 3822921 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1742171979.733327 3822921 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1742171979.733330 3822921 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-17 00:39:39.733334: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.736681 3822921 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1742171979.736695 3822921 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1742171979.736698 3822921 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1742171979.736701 3822921 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1742171979.736704 3822921 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1742171979.736707 3822921 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1742171979.736710 3822921 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1742171979.736712 3822921 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1742171979.736715 3822921 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1742171979.736718 3822921 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-17 00:39:39.736723: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.740005 3822921 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1742171979.740020 3822921 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1742171979.740023 3822921 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1742171979.740026 3822921 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1742171979.740029 3822921 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1742171979.740032 3822921 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1742171979.740035 3822921 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1742171979.740038 3822921 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1742171979.740040 3822921 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1742171979.740043 3822921 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-17 00:39:39.740048: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.743354 3822921 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1742171979.743370 3822921 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1742171979.743373 3822921 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1742171979.743376 3822921 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1742171979.743379 3822921 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1742171979.743382 3822921 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1742171979.743385 3822921 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1742171979.743388 3822921 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1742171979.743391 3822921 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1742171979.743394 3822921 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-17 00:39:39.743398: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.746720 3822921 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1742171979.746733 3822921 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1742171979.746736 3822921 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1742171979.746741 3822921 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1742171979.746744 3822921 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1742171979.746747 3822921 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1742171979.746750 3822921 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1742171979.746752 3822921 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1742171979.746755 3822921 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1742171979.746758 3822921 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-17 00:39:39.746763: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.750061 3822921 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1742171979.750074 3822921 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1742171979.750077 3822921 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1742171979.750080 3822921 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1742171979.750083 3822921 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1742171979.750086 3822921 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1742171979.750089 3822921 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1742171979.750092 3822921 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1742171979.750095 3822921 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1742171979.750098 3822921 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-17 00:39:39.750103: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.753393 3822921 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1742171979.753409 3822921 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1742171979.753413 3822921 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1742171979.753415 3822921 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1742171979.753418 3822921 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1742171979.753421 3822921 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1742171979.753424 3822921 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1742171979.753427 3822921 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1742171979.753430 3822921 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1742171979.753433 3822921 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-17 00:39:39.753438: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.756862 3822921 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1742171979.756878 3822921 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.756881 3822921 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.756884 3822921 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1742171979.756887 3822921 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1742171979.756890 3822921 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1742171979.756893 3822921 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1742171979.756897 3822921 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1742171979.756900 3822921 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1742171979.756903 3822921 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-17 00:39:39.756908: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.760323 3822921 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1742171979.760338 3822921 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.760342 3822921 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.760345 3822921 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1742171979.760348 3822921 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1742171979.760350 3822921 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1742171979.760353 3822921 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1742171979.760356 3822921 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1742171979.760359 3822921 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1742171979.760362 3822921 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-17 00:39:39.760367: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.763696 3822921 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1742171979.763710 3822921 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.763714 3822921 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.763717 3822921 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1742171979.763720 3822921 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1742171979.763722 3822921 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1742171979.763725 3822921 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1742171979.763728 3822921 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1742171979.763731 3822921 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1742171979.763734 3822921 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-17 00:39:39.763739: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.767059 3822921 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1742171979.767075 3822921 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1742171979.767078 3822921 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1742171979.767081 3822921 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1742171979.767084 3822921 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1742171979.767087 3822921 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1742171979.767090 3822921 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1742171979.767093 3822921 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1742171979.767095 3822921 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1742171979.767098 3822921 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-17 00:39:39.767103: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.770673 3822921 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1742171979.770688 3822921 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1742171979.770692 3822921 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1742171979.770695 3822921 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1742171979.770698 3822921 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1742171979.770701 3822921 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1742171979.770705 3822921 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1742171979.770708 3822921 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1742171979.770712 3822921 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1742171979.770715 3822921 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-17 00:39:39.770720: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.774264 3822921 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1742171979.774278 3822921 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1742171979.774281 3822921 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1742171979.774284 3822921 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1742171979.774287 3822921 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1742171979.774290 3822921 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1742171979.774293 3822921 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1742171979.774296 3822921 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1742171979.774299 3822921 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1742171979.774302 3822921 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-17 00:39:39.774307: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1742171979.777796 3822921 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1742171979.777814 3822921 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1742171979.777817 3822921 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1742171979.777820 3822921 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1742171979.777823 3822921 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1742171979.777825 3822921 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1742171979.777828 3822921 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1742171979.777831 3822921 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1742171979.777834 3822921 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1742171979.777837 3822921 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-17 00:39:39.777842: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.793502	Test Acc: 65.6000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MLDataDevices)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.4</span></span>
<span class="line"><span>Commit 8561cc3d68d (2025-03-10 11:36 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const d=s(c,[["render",i]]);export{E as __pageData,d as default};
