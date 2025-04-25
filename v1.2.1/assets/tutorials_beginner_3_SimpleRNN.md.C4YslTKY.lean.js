import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.hPFB-k2O.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.2.1/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.2.1/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.2.1/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.2.1/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61378</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.61847</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57032</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54322</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51738</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49400</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49185</span></span>
<span class="line"><span>Validation: Loss 0.45480 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47453 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47216</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43763</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44298</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42261</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41180</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42190</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42016</span></span>
<span class="line"><span>Validation: Loss 0.35533 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37903 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37999</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35903</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34447</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34428</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31666</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31763</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30522</span></span>
<span class="line"><span>Validation: Loss 0.26980 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29549 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28968</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28080</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26499</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25127</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24872</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25242</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23939</span></span>
<span class="line"><span>Validation: Loss 0.19994 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22619 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22451</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20902</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20315</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20046</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19254</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17871</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15594</span></span>
<span class="line"><span>Validation: Loss 0.14601 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17044 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16866</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16190</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15189</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13589</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15081</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13322</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11285</span></span>
<span class="line"><span>Validation: Loss 0.10586 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12644 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11841</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12407</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11227</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11104</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09337</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10119</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08812</span></span>
<span class="line"><span>Validation: Loss 0.07567 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09117 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08371</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08902</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08305</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07813</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07189</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06359</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06339</span></span>
<span class="line"><span>Validation: Loss 0.05300 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06344 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06268</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05926</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05604</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05444</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05001</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04881</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04332</span></span>
<span class="line"><span>Validation: Loss 0.03955 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04699 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04381</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04381</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04311</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04110</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04220</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03807</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03988</span></span>
<span class="line"><span>Validation: Loss 0.03194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03800 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03509</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.04033</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03274</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03695</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03296</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02967</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03474</span></span>
<span class="line"><span>Validation: Loss 0.02701 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03226 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03138</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03313</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03035</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02687</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02963</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02759</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02627</span></span>
<span class="line"><span>Validation: Loss 0.02343 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02680</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02584</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02700</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02559</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02473</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02564</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02675</span></span>
<span class="line"><span>Validation: Loss 0.02068 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02486 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02438</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02441</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02154</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02305</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02208</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02473</span></span>
<span class="line"><span>Validation: Loss 0.01848 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02227 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01965</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02061</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02220</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02020</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02103</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02016</span></span>
<span class="line"><span>Validation: Loss 0.01666 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02013 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02080</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01909</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01847</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01717</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01919</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01841</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01769</span></span>
<span class="line"><span>Validation: Loss 0.01511 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01832 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01754</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01549</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01909</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01541</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.02053</span></span>
<span class="line"><span>Validation: Loss 0.01378 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01676 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01688</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01558</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01599</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01535</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01483</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01362</span></span>
<span class="line"><span>Validation: Loss 0.01260 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01536 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01547</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01457</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01249</span></span>
<span class="line"><span>Validation: Loss 0.01154 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01410 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01267</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01277</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01139</span></span>
<span class="line"><span>Validation: Loss 0.01054 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01288 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01309</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01098</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01161</span></span>
<span class="line"><span>Validation: Loss 0.00949 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01157 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01222</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01097</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01182</span></span>
<span class="line"><span>Validation: Loss 0.00840 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01018 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00879</span></span>
<span class="line"><span>Validation: Loss 0.00751 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00905 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00835</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00855</span></span>
<span class="line"><span>Validation: Loss 0.00687 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00826 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00712</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00739</span></span>
<span class="line"><span>Validation: Loss 0.00639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00767 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63161</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59157</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56177</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53878</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51506</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50421</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48370</span></span>
<span class="line"><span>Validation: Loss 0.46435 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47300</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44561</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44431</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41993</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41492</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39915</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38991</span></span>
<span class="line"><span>Validation: Loss 0.36601 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37949 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37226</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35362</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35615</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33198</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30412</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31466</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30811</span></span>
<span class="line"><span>Validation: Loss 0.28010 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29456 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28383</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27321</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26839</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24670</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24752</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22688</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25679</span></span>
<span class="line"><span>Validation: Loss 0.20971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22409 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21850</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20251</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19916</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19424</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17411</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16995</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18597</span></span>
<span class="line"><span>Validation: Loss 0.15423 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16730 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15835</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14846</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14356</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14558</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12427</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13242</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13393</span></span>
<span class="line"><span>Validation: Loss 0.11213 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12277 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11278</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11189</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10748</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10338</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09165</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09369</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08419</span></span>
<span class="line"><span>Validation: Loss 0.07983 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08754 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08700</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07787</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07338</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07191</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06659</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05968</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06656</span></span>
<span class="line"><span>Validation: Loss 0.05568 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06079 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05713</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05704</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04976</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04687</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04846</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04931</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04599</span></span>
<span class="line"><span>Validation: Loss 0.04171 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04545 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04481</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04029</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03901</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03979</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03917</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03718</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03122</span></span>
<span class="line"><span>Validation: Loss 0.03387 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03699 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03528</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03387</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03192</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03362</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03126</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03112</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03346</span></span>
<span class="line"><span>Validation: Loss 0.02880 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03154 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03106</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02795</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03002</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02898</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02863</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02439</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02284</span></span>
<span class="line"><span>Validation: Loss 0.02506 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02751 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02571</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02592</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02504</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02589</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02314</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02314</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02421</span></span>
<span class="line"><span>Validation: Loss 0.02219 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02441 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02222</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02462</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02325</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02062</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02083</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02015</span></span>
<span class="line"><span>Validation: Loss 0.01986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02190 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02059</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01920</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01965</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01837</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02177</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01750</span></span>
<span class="line"><span>Validation: Loss 0.01794 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01983 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01861</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01836</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01845</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01805</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02105</span></span>
<span class="line"><span>Validation: Loss 0.01631 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01806 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01821</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01545</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01495</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01715</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01604</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Validation: Loss 0.01490 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01654 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01624</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01628</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01575</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01302</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01273</span></span>
<span class="line"><span>Validation: Loss 0.01370 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01522 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01309</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01697</span></span>
<span class="line"><span>Validation: Loss 0.01266 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01408 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01210</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01379</span></span>
<span class="line"><span>Validation: Loss 0.01172 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01304 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01277</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01133</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01189</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01185</span></span>
<span class="line"><span>Validation: Loss 0.01085 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01207 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01116</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01054</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01052</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01050</span></span>
<span class="line"><span>Validation: Loss 0.00996 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01105 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01054</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00835</span></span>
<span class="line"><span>Validation: Loss 0.00897 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00992 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00719</span></span>
<span class="line"><span>Validation: Loss 0.00797 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00879 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00835</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00731</span></span>
<span class="line"><span>Validation: Loss 0.00722 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00794 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.6</span></span>
<span class="line"><span>Commit 67dffc4a8ae (2024-10-28 12:23 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.6, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.6</span></span>
<span class="line"><span>NVIDIA driver 560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.3</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.3+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.3+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.6</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.391 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
