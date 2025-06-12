import{_ as a,c as n,o as e,al as i}from"./chunks/framework.Dgw_Mll3.js";const k=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"};function t(c,s,l,r,h,o){return e(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Precompiling Lux...</span></span>
<span class="line"><span>   9814.5 ms  ✓ Lux</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 10 seconds. 104 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxMLUtilsExt...</span></span>
<span class="line"><span>   2181.5 ms  ✓ Lux → LuxMLUtilsExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 2 seconds. 164 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxEnzymeExt...</span></span>
<span class="line"><span>   8163.2 ms  ✓ Lux → LuxEnzymeExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 8 seconds. 149 already precompiled.</span></span>
<span class="line"><span>Precompiling LuxReactantExt...</span></span>
<span class="line"><span>  13037.1 ms  ✓ Lux → LuxReactantExt</span></span>
<span class="line"><span>  1 dependency successfully precompiled in 13 seconds. 180 already precompiled.</span></span></code></pre></div><h2 id="Dataset" tabindex="-1">Dataset <a class="header-anchor" href="#Dataset" aria-label="Permalink to &quot;Dataset {#Dataset}&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L,C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##230&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T,3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T,3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), model))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_loader)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ad </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                ad, lossfn, (x, y), train_state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch (total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples) (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()((train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-9/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>2025-05-23 22:12:47.371617: I external/xla/xla/service/service.cc:152] XLA service 0x35a77cb0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-05-23 22:12:47.371786: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1748038367.372551  646469 se_gpu_pjrt_client.cc:1026] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1748038367.372642  646469 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1748038367.372684  646469 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1748038367.386913  646469 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>E0000 00:00:1748038432.608256  646469 buffer_comparator.cc:145] Difference at 32: 0, expected 1.62244</span></span>
<span class="line"><span>E0000 00:00:1748038432.608296  646469 buffer_comparator.cc:145] Difference at 33: 0, expected 1.87084</span></span>
<span class="line"><span>E0000 00:00:1748038432.608304  646469 buffer_comparator.cc:145] Difference at 34: 0, expected 1.07351</span></span>
<span class="line"><span>E0000 00:00:1748038432.608310  646469 buffer_comparator.cc:145] Difference at 35: 0, expected 2.92445</span></span>
<span class="line"><span>E0000 00:00:1748038432.608315  646469 buffer_comparator.cc:145] Difference at 36: 0, expected 1.98056</span></span>
<span class="line"><span>E0000 00:00:1748038432.608320  646469 buffer_comparator.cc:145] Difference at 37: 0, expected 2.07715</span></span>
<span class="line"><span>E0000 00:00:1748038432.608326  646469 buffer_comparator.cc:145] Difference at 38: 0, expected 1.56458</span></span>
<span class="line"><span>E0000 00:00:1748038432.608331  646469 buffer_comparator.cc:145] Difference at 39: 0, expected 2.27034</span></span>
<span class="line"><span>E0000 00:00:1748038432.608336  646469 buffer_comparator.cc:145] Difference at 40: 0, expected 2.31795</span></span>
<span class="line"><span>E0000 00:00:1748038432.608342  646469 buffer_comparator.cc:145] Difference at 41: 0, expected 2.55731</span></span>
<span class="line"><span>2025-05-23 22:13:52.608356: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.612725  646469 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1748038432.612751  646469 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1748038432.612756  646469 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1748038432.612759  646469 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1748038432.612762  646469 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1748038432.612765  646469 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1748038432.612768  646469 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1748038432.612770  646469 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1748038432.612774  646469 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1748038432.612776  646469 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-05-23 22:13:52.612783: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.615204  646469 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1748038432.615219  646469 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1748038432.615223  646469 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1748038432.615226  646469 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1748038432.615228  646469 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1748038432.615231  646469 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1748038432.615234  646469 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1748038432.615237  646469 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1748038432.615241  646469 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1748038432.615244  646469 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-05-23 22:13:52.615249: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.617693  646469 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1748038432.617708  646469 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1748038432.617711  646469 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1748038432.617714  646469 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1748038432.617717  646469 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1748038432.617720  646469 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1748038432.617723  646469 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1748038432.617725  646469 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1748038432.617728  646469 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1748038432.617731  646469 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-05-23 22:13:52.617736: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.620157  646469 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1748038432.620172  646469 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1748038432.620176  646469 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1748038432.620179  646469 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1748038432.620182  646469 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1748038432.620185  646469 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1748038432.620188  646469 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1748038432.620190  646469 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1748038432.620193  646469 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1748038432.620196  646469 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-05-23 22:13:52.620201: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.622678  646469 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1748038432.622692  646469 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1748038432.622695  646469 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1748038432.622698  646469 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1748038432.622701  646469 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1748038432.622704  646469 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1748038432.622707  646469 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1748038432.622709  646469 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1748038432.622712  646469 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1748038432.622715  646469 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-05-23 22:13:52.622720: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.625152  646469 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1748038432.625165  646469 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1748038432.625168  646469 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1748038432.625171  646469 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1748038432.625174  646469 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1748038432.625177  646469 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1748038432.625179  646469 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1748038432.625182  646469 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1748038432.625185  646469 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1748038432.625188  646469 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-05-23 22:13:52.625193: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.627660  646469 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1748038432.627676  646469 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1748038432.627679  646469 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1748038432.627682  646469 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1748038432.627685  646469 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1748038432.627688  646469 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1748038432.627691  646469 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1748038432.627693  646469 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1748038432.627696  646469 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1748038432.627699  646469 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-05-23 22:13:52.627704: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.630114  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.630128  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.630132  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.630135  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.630138  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.630140  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.630143  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.630146  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.630149  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.630152  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.630157: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.632576  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.632589  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.632593  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.632598  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.632601  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.632604  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.632608  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.632612  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.632614  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.632617  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.632622: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.635064  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.635075  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.635079  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.635082  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.635085  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.635087  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.635090  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.635093  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.635096  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.635099  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.635103: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.637482  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.637504  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.637508  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.637510  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.637513  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.637516  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.637519  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.637522  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.637524  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.637527  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.637532: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.639924  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.639939  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.639942  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.639945  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.639947  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.639950  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.639953  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.639957  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.639960  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.639963  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.639967: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.642365  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.642377  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.642380  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.642383  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.642386  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.642389  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.642391  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.642394  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.642397  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.642400  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.642404: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.644819  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.644831  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.644834  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.644837  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.644840  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.644842  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.644845  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.644848  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.644851  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.644853  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.644858: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.647248  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.647260  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.647263  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.647266  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.647268  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.647271  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.647274  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.647277  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.647280  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.647282  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.647287: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.649695  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.649707  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.649710  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.649713  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.649716  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.649719  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.649722  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.649724  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.649727  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.649730  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.649735: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.652121  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.652132  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.652135  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.652138  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.652141  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.652144  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.652147  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.652150  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.652153  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.652155  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.652160: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.654550  646469 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1748038432.654561  646469 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1748038432.654564  646469 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1748038432.654567  646469 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1748038432.654570  646469 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1748038432.654573  646469 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1748038432.654576  646469 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1748038432.654579  646469 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1748038432.654582  646469 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1748038432.654584  646469 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-23 22:13:52.654589: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.657000  646469 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1748038432.657011  646469 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1748038432.657015  646469 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1748038432.657019  646469 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1748038432.657022  646469 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1748038432.657025  646469 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1748038432.657027  646469 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1748038432.657030  646469 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1748038432.657033  646469 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1748038432.657036  646469 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-23 22:13:52.657040: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.659448  646469 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1748038432.659459  646469 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1748038432.659462  646469 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1748038432.659465  646469 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1748038432.659468  646469 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1748038432.659471  646469 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1748038432.659474  646469 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1748038432.659477  646469 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1748038432.659480  646469 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1748038432.659482  646469 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-23 22:13:52.659487: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.661878  646469 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1748038432.661889  646469 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1748038432.661892  646469 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1748038432.661895  646469 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1748038432.661898  646469 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1748038432.661901  646469 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1748038432.661904  646469 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1748038432.661906  646469 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1748038432.661909  646469 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1748038432.661912  646469 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-23 22:13:52.661916: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.664301  646469 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1748038432.664312  646469 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1748038432.664316  646469 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1748038432.664318  646469 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1748038432.664321  646469 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1748038432.664324  646469 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1748038432.664327  646469 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1748038432.664331  646469 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1748038432.664334  646469 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1748038432.664337  646469 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-23 22:13:52.664341: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038432.666723  646469 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1748038432.666735  646469 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1748038432.666738  646469 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1748038432.666741  646469 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1748038432.666744  646469 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1748038432.666747  646469 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1748038432.666750  646469 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1748038432.666752  646469 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1748038432.666755  646469 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1748038432.666758  646469 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-23 22:13:52.666762: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.318786  646469 buffer_comparator.cc:145] Difference at 16: 0.196842, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1748038471.318832  646469 buffer_comparator.cc:145] Difference at 17: 0.688536, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1748038471.318837  646469 buffer_comparator.cc:145] Difference at 18: 0.927057, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1748038471.318840  646469 buffer_comparator.cc:145] Difference at 19: 0.579189, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1748038471.318844  646469 buffer_comparator.cc:145] Difference at 20: 0.374055, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1748038471.318847  646469 buffer_comparator.cc:145] Difference at 21: 0.216797, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1748038471.318849  646469 buffer_comparator.cc:145] Difference at 22: 0.731212, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1748038471.318852  646469 buffer_comparator.cc:145] Difference at 23: 0.700668, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1748038471.318855  646469 buffer_comparator.cc:145] Difference at 24: 0.5317, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1748038471.318858  646469 buffer_comparator.cc:145] Difference at 25: 0.24009, expected 11.3838</span></span>
<span class="line"><span>2025-05-23 22:14:31.318869: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.320923  646469 buffer_comparator.cc:145] Difference at 16: 0.196842, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1748038471.320937  646469 buffer_comparator.cc:145] Difference at 17: 0.688536, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1748038471.320940  646469 buffer_comparator.cc:145] Difference at 18: 0.927057, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1748038471.320943  646469 buffer_comparator.cc:145] Difference at 19: 0.579189, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1748038471.320946  646469 buffer_comparator.cc:145] Difference at 20: 0.374055, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1748038471.320949  646469 buffer_comparator.cc:145] Difference at 21: 0.216797, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1748038471.320952  646469 buffer_comparator.cc:145] Difference at 22: 0.731212, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1748038471.320956  646469 buffer_comparator.cc:145] Difference at 23: 0.700668, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1748038471.320958  646469 buffer_comparator.cc:145] Difference at 24: 0.5317, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1748038471.320961  646469 buffer_comparator.cc:145] Difference at 25: 0.24009, expected 11.3838</span></span>
<span class="line"><span>2025-05-23 22:14:31.320968: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.323026  646469 buffer_comparator.cc:145] Difference at 16: 0.196842, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1748038471.323056  646469 buffer_comparator.cc:145] Difference at 17: 0.688536, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1748038471.323060  646469 buffer_comparator.cc:145] Difference at 18: 0.927057, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1748038471.323063  646469 buffer_comparator.cc:145] Difference at 19: 0.579189, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1748038471.323065  646469 buffer_comparator.cc:145] Difference at 20: 0.374055, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1748038471.323068  646469 buffer_comparator.cc:145] Difference at 21: 0.216797, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1748038471.323071  646469 buffer_comparator.cc:145] Difference at 22: 0.731212, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1748038471.323074  646469 buffer_comparator.cc:145] Difference at 23: 0.700668, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1748038471.323077  646469 buffer_comparator.cc:145] Difference at 24: 0.5317, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1748038471.323080  646469 buffer_comparator.cc:145] Difference at 25: 0.24009, expected 11.3838</span></span>
<span class="line"><span>2025-05-23 22:14:31.323087: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.325230  646469 buffer_comparator.cc:145] Difference at 32: 0.766695, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1748038471.325246  646469 buffer_comparator.cc:145] Difference at 33: 1.02114, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1748038471.325249  646469 buffer_comparator.cc:145] Difference at 34: 0.0917029, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1748038471.325253  646469 buffer_comparator.cc:145] Difference at 35: 0.842239, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1748038471.325256  646469 buffer_comparator.cc:145] Difference at 36: 0.52163, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1748038471.325259  646469 buffer_comparator.cc:145] Difference at 37: 0.313266, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1748038471.325262  646469 buffer_comparator.cc:145] Difference at 38: 1.04173, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1748038471.325264  646469 buffer_comparator.cc:145] Difference at 39: 0.974961, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1748038471.325267  646469 buffer_comparator.cc:145] Difference at 40: 0.978602, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1748038471.325270  646469 buffer_comparator.cc:145] Difference at 41: 1.00507, expected 8.63119</span></span>
<span class="line"><span>2025-05-23 22:14:31.325276: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.327406  646469 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1748038471.327422  646469 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1748038471.327426  646469 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1748038471.327429  646469 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1748038471.327432  646469 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1748038471.327435  646469 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1748038471.327438  646469 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1748038471.327440  646469 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1748038471.327443  646469 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1748038471.327446  646469 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-23 22:14:31.327451: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.329549  646469 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1748038471.329575  646469 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1748038471.329579  646469 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1748038471.329582  646469 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1748038471.329585  646469 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1748038471.329587  646469 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1748038471.329590  646469 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1748038471.329593  646469 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1748038471.329596  646469 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1748038471.329599  646469 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-23 22:14:31.329606: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.331750  646469 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1748038471.331777  646469 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1748038471.331781  646469 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1748038471.331784  646469 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1748038471.331787  646469 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1748038471.331790  646469 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1748038471.331793  646469 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1748038471.331796  646469 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1748038471.331798  646469 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1748038471.331801  646469 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-23 22:14:31.331807: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.333933  646469 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1748038471.333946  646469 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1748038471.333949  646469 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1748038471.333952  646469 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1748038471.333955  646469 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1748038471.333958  646469 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1748038471.333961  646469 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1748038471.333964  646469 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1748038471.333966  646469 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1748038471.333969  646469 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-23 22:14:31.333974: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.336016  646469 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1748038471.336030  646469 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1748038471.336033  646469 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1748038471.336036  646469 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1748038471.336039  646469 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1748038471.336042  646469 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1748038471.336045  646469 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1748038471.336048  646469 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1748038471.336050  646469 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1748038471.336053  646469 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-23 22:14:31.336058: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.338109  646469 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1748038471.338123  646469 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1748038471.338127  646469 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1748038471.338130  646469 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1748038471.338133  646469 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1748038471.338135  646469 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1748038471.338138  646469 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1748038471.338141  646469 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1748038471.338144  646469 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1748038471.338147  646469 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-23 22:14:31.338152: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.343616  646469 buffer_comparator.cc:145] Difference at 16: 9.68745, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1748038471.343659  646469 buffer_comparator.cc:145] Difference at 17: 10.1876, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1748038471.343662  646469 buffer_comparator.cc:145] Difference at 18: 8.84104, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1748038471.343665  646469 buffer_comparator.cc:145] Difference at 19: 10.0381, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1748038471.343669  646469 buffer_comparator.cc:145] Difference at 20: 7.30446, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1748038471.343672  646469 buffer_comparator.cc:145] Difference at 21: 8.26483, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1748038471.343675  646469 buffer_comparator.cc:145] Difference at 22: 10.8549, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1748038471.343678  646469 buffer_comparator.cc:145] Difference at 23: 7.87482, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1748038471.343681  646469 buffer_comparator.cc:145] Difference at 24: 9.78239, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1748038471.343683  646469 buffer_comparator.cc:145] Difference at 25: 11.3838, expected 36.4575</span></span>
<span class="line"><span>2025-05-23 22:14:31.343692: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.345745  646469 buffer_comparator.cc:145] Difference at 16: 9.68745, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1748038471.345759  646469 buffer_comparator.cc:145] Difference at 17: 10.1876, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1748038471.345763  646469 buffer_comparator.cc:145] Difference at 18: 8.84104, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1748038471.345767  646469 buffer_comparator.cc:145] Difference at 19: 10.0381, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1748038471.345770  646469 buffer_comparator.cc:145] Difference at 20: 7.30446, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1748038471.345773  646469 buffer_comparator.cc:145] Difference at 21: 8.26483, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1748038471.345776  646469 buffer_comparator.cc:145] Difference at 22: 10.8549, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1748038471.345779  646469 buffer_comparator.cc:145] Difference at 23: 7.87482, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1748038471.345782  646469 buffer_comparator.cc:145] Difference at 24: 9.78239, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1748038471.345785  646469 buffer_comparator.cc:145] Difference at 25: 11.3838, expected 36.4575</span></span>
<span class="line"><span>2025-05-23 22:14:31.345790: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.347844  646469 buffer_comparator.cc:145] Difference at 16: 9.68745, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1748038471.347856  646469 buffer_comparator.cc:145] Difference at 17: 10.1876, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1748038471.347859  646469 buffer_comparator.cc:145] Difference at 18: 8.84104, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1748038471.347862  646469 buffer_comparator.cc:145] Difference at 19: 10.0381, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1748038471.347865  646469 buffer_comparator.cc:145] Difference at 20: 7.30446, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1748038471.347868  646469 buffer_comparator.cc:145] Difference at 21: 8.26483, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1748038471.347871  646469 buffer_comparator.cc:145] Difference at 22: 10.8549, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1748038471.347874  646469 buffer_comparator.cc:145] Difference at 23: 7.87482, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1748038471.347877  646469 buffer_comparator.cc:145] Difference at 24: 9.78239, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1748038471.347880  646469 buffer_comparator.cc:145] Difference at 25: 11.3838, expected 36.4575</span></span>
<span class="line"><span>2025-05-23 22:14:31.347885: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.349912  646469 buffer_comparator.cc:145] Difference at 16: 9.68745, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1748038471.349925  646469 buffer_comparator.cc:145] Difference at 17: 10.1876, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1748038471.349928  646469 buffer_comparator.cc:145] Difference at 18: 8.84104, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1748038471.349931  646469 buffer_comparator.cc:145] Difference at 19: 10.0381, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1748038471.349934  646469 buffer_comparator.cc:145] Difference at 20: 7.30446, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1748038471.349937  646469 buffer_comparator.cc:145] Difference at 21: 8.26483, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1748038471.349940  646469 buffer_comparator.cc:145] Difference at 22: 10.8549, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1748038471.349943  646469 buffer_comparator.cc:145] Difference at 23: 7.87482, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1748038471.349946  646469 buffer_comparator.cc:145] Difference at 24: 9.78239, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1748038471.349949  646469 buffer_comparator.cc:145] Difference at 25: 11.3838, expected 36.4575</span></span>
<span class="line"><span>2025-05-23 22:14:31.349954: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.351995  646469 buffer_comparator.cc:145] Difference at 16: 9.68745, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1748038471.352008  646469 buffer_comparator.cc:145] Difference at 17: 10.1876, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1748038471.352012  646469 buffer_comparator.cc:145] Difference at 18: 8.84104, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1748038471.352015  646469 buffer_comparator.cc:145] Difference at 19: 10.0381, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1748038471.352018  646469 buffer_comparator.cc:145] Difference at 20: 7.30446, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1748038471.352022  646469 buffer_comparator.cc:145] Difference at 21: 8.26483, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1748038471.352024  646469 buffer_comparator.cc:145] Difference at 22: 10.8549, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1748038471.352027  646469 buffer_comparator.cc:145] Difference at 23: 7.87482, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1748038471.352030  646469 buffer_comparator.cc:145] Difference at 24: 9.78239, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1748038471.352033  646469 buffer_comparator.cc:145] Difference at 25: 11.3838, expected 36.4575</span></span>
<span class="line"><span>2025-05-23 22:14:31.352038: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.373075  646469 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1748038471.373119  646469 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1748038471.373122  646469 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1748038471.373125  646469 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1748038471.373128  646469 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1748038471.373131  646469 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1748038471.373134  646469 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1748038471.373137  646469 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1748038471.373140  646469 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1748038471.373143  646469 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-23 22:14:31.373153: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.375374  646469 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1748038471.375397  646469 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1748038471.375401  646469 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1748038471.375403  646469 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1748038471.375406  646469 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1748038471.375409  646469 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1748038471.375412  646469 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1748038471.375415  646469 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1748038471.375418  646469 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1748038471.375421  646469 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-23 22:14:31.375428: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.377507  646469 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1748038471.377520  646469 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1748038471.377523  646469 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1748038471.377526  646469 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1748038471.377529  646469 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1748038471.377532  646469 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1748038471.377535  646469 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1748038471.377540  646469 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1748038471.377542  646469 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1748038471.377545  646469 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-23 22:14:31.377550: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.379587  646469 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1748038471.379599  646469 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1748038471.379602  646469 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1748038471.379605  646469 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1748038471.379608  646469 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1748038471.379611  646469 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1748038471.379614  646469 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1748038471.379617  646469 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1748038471.379620  646469 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1748038471.379623  646469 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-23 22:14:31.379640: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1748038471.381676  646469 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1748038471.381688  646469 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1748038471.381692  646469 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1748038471.381695  646469 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1748038471.381698  646469 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1748038471.381700  646469 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1748038471.381703  646469 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1748038471.381706  646469 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1748038471.381709  646469 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1748038471.381712  646469 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-23 22:14:31.381716: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56353</span></span>
<span class="line"><span>Validation:	Loss 0.50427	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43760</span></span>
<span class="line"><span>Validation:	Loss 0.37234	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31732</span></span>
<span class="line"><span>Validation:	Loss 0.25824	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21645</span></span>
<span class="line"><span>Validation:	Loss 0.16888	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14366</span></span>
<span class="line"><span>Validation:	Loss 0.11250	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09706</span></span>
<span class="line"><span>Validation:	Loss 0.07739	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06675</span></span>
<span class="line"><span>Validation:	Loss 0.05443	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04712</span></span>
<span class="line"><span>Validation:	Loss 0.03963	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03465</span></span>
<span class="line"><span>Validation:	Loss 0.02999	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02690</span></span>
<span class="line"><span>Validation:	Loss 0.02402	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02195</span></span>
<span class="line"><span>Validation:	Loss 0.02012	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01858</span></span>
<span class="line"><span>Validation:	Loss 0.01733	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01610</span></span>
<span class="line"><span>Validation:	Loss 0.01515	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01411</span></span>
<span class="line"><span>Validation:	Loss 0.01335	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01241</span></span>
<span class="line"><span>Validation:	Loss 0.01180	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01102</span></span>
<span class="line"><span>Validation:	Loss 0.01052	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00987</span></span>
<span class="line"><span>Validation:	Loss 0.00951	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00895</span></span>
<span class="line"><span>Validation:	Loss 0.00871	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00822</span></span>
<span class="line"><span>Validation:	Loss 0.00806	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00762</span></span>
<span class="line"><span>Validation:	Loss 0.00750	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00706</span></span>
<span class="line"><span>Validation:	Loss 0.00701	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00661</span></span>
<span class="line"><span>Validation:	Loss 0.00658	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00620</span></span>
<span class="line"><span>Validation:	Loss 0.00620	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00584</span></span>
<span class="line"><span>Validation:	Loss 0.00585	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00553</span></span>
<span class="line"><span>Validation:	Loss 0.00554	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-9/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59043</span></span>
<span class="line"><span>Validation:	Loss 0.47404	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41719</span></span>
<span class="line"><span>Validation:	Loss 0.35296	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31751</span></span>
<span class="line"><span>Validation:	Loss 0.28505	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25748</span></span>
<span class="line"><span>Validation:	Loss 0.23591	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21286</span></span>
<span class="line"><span>Validation:	Loss 0.19485	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17713</span></span>
<span class="line"><span>Validation:	Loss 0.16026	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.14446</span></span>
<span class="line"><span>Validation:	Loss 0.13061	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.11617</span></span>
<span class="line"><span>Validation:	Loss 0.10330	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.09070</span></span>
<span class="line"><span>Validation:	Loss 0.07849	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.06936</span></span>
<span class="line"><span>Validation:	Loss 0.06184	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.05570</span></span>
<span class="line"><span>Validation:	Loss 0.05147	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.04732</span></span>
<span class="line"><span>Validation:	Loss 0.04412	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.04098</span></span>
<span class="line"><span>Validation:	Loss 0.03850	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.03563</span></span>
<span class="line"><span>Validation:	Loss 0.03371	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.03134</span></span>
<span class="line"><span>Validation:	Loss 0.02917	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02683</span></span>
<span class="line"><span>Validation:	Loss 0.02507	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.02358</span></span>
<span class="line"><span>Validation:	Loss 0.02252	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.02151</span></span>
<span class="line"><span>Validation:	Loss 0.02068	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01969</span></span>
<span class="line"><span>Validation:	Loss 0.01899	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01820</span></span>
<span class="line"><span>Validation:	Loss 0.01760	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01684</span></span>
<span class="line"><span>Validation:	Loss 0.01638	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01570</span></span>
<span class="line"><span>Validation:	Loss 0.01531	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01470</span></span>
<span class="line"><span>Validation:	Loss 0.01435	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01379</span></span>
<span class="line"><span>Validation:	Loss 0.01350	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.01301</span></span>
<span class="line"><span>Validation:	Loss 0.01273	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.5</span></span>
<span class="line"><span>Commit 760b2e5b739 (2025-04-14 06:53 UTC)</span></span>
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
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,47)]))}const d=a(p,[["render",t]]);export{k as __pageData,d as default};
