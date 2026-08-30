import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.Dl8xuq2C.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.1.0/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.1.0/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.1.0/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.1.0/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62124</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59112</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56059</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53948</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52964</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50251</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47554</span></span>
<span class="line"><span>Validation: Loss 0.46974 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45945 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46349</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45522</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44148</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42113</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41486</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40422</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39463</span></span>
<span class="line"><span>Validation: Loss 0.37277 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36055 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36452</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36135</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35644</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32859</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32668</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30146</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31281</span></span>
<span class="line"><span>Validation: Loss 0.28783 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27462 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27613</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27654</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27543</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26748</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23477</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23649</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21447</span></span>
<span class="line"><span>Validation: Loss 0.21765 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20443 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.23198</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19341</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20363</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19023</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17448</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17787</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17737</span></span>
<span class="line"><span>Validation: Loss 0.16169 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14953 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15398</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14600</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14303</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14745</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13438</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13613</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14339</span></span>
<span class="line"><span>Validation: Loss 0.11852 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10852 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12618</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11306</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11519</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10155</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09263</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08684</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07856</span></span>
<span class="line"><span>Validation: Loss 0.08455 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07723 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08516</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07670</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07334</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07463</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06892</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06460</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06731</span></span>
<span class="line"><span>Validation: Loss 0.05894 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05402 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05723</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05550</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05291</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05032</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05052</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04744</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04511</span></span>
<span class="line"><span>Validation: Loss 0.04383 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04037 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03820</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04462</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04101</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04476</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03514</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03749</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03838</span></span>
<span class="line"><span>Validation: Loss 0.03552 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03269 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03432</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03486</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03209</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03268</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03431</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03226</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02758</span></span>
<span class="line"><span>Validation: Loss 0.03016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02769 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03089</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02854</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03082</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02693</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02546</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02794</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03010</span></span>
<span class="line"><span>Validation: Loss 0.02626 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02406 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02713</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02577</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02164</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02613</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02443</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02483</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02423</span></span>
<span class="line"><span>Validation: Loss 0.02323 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02125 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02202</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02224</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02442</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02039</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02272</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02162</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02090</span></span>
<span class="line"><span>Validation: Loss 0.02080 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01898 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02078</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01922</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02123</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01900</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01976</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01949</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02024</span></span>
<span class="line"><span>Validation: Loss 0.01878 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01711 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01894</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01767</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01859</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01766</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01857</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01625</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02031</span></span>
<span class="line"><span>Validation: Loss 0.01707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01551 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01768</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01661</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01648</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01597</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01269</span></span>
<span class="line"><span>Validation: Loss 0.01559 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01415 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01602</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01392</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01499</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01425</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01435</span></span>
<span class="line"><span>Validation: Loss 0.01434 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01298 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01529</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01319</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01329</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01322</span></span>
<span class="line"><span>Validation: Loss 0.01323 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01197 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01080</span></span>
<span class="line"><span>Validation: Loss 0.01224 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01106 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01293</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01173</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01180</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01301</span></span>
<span class="line"><span>Validation: Loss 0.01125 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01017 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01140</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01133</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01104</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00901</span></span>
<span class="line"><span>Validation: Loss 0.01016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00954</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00937</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00903</span></span>
<span class="line"><span>Validation: Loss 0.00900 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00817 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00951</span></span>
<span class="line"><span>Validation: Loss 0.00804 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00734 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00611</span></span>
<span class="line"><span>Validation: Loss 0.00737 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00674 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62745</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59260</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56156</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54179</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51448</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49976</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47605</span></span>
<span class="line"><span>Validation: Loss 0.47360 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47051 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46356</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44939</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44393</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42764</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41013</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39318</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39291</span></span>
<span class="line"><span>Validation: Loss 0.37700 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37330 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36664</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36796</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34074</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33221</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30405</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31778</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28704</span></span>
<span class="line"><span>Validation: Loss 0.29212 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28802 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27207</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27869</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27045</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25777</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22519</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23846</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24046</span></span>
<span class="line"><span>Validation: Loss 0.22187 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21769 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21279</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19832</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20248</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18042</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18462</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17985</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15683</span></span>
<span class="line"><span>Validation: Loss 0.16533 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16145 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15520</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15812</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13642</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14315</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13897</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11991</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11770</span></span>
<span class="line"><span>Validation: Loss 0.12127 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11802 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11105</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11183</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10622</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09932</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09284</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09669</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08107</span></span>
<span class="line"><span>Validation: Loss 0.08672 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08430 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08598</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07321</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06965</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07609</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06411</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06593</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06265</span></span>
<span class="line"><span>Validation: Loss 0.06032 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05872 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05744</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05601</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05251</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04870</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04794</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04374</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05199</span></span>
<span class="line"><span>Validation: Loss 0.04490 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04110</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03807</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04066</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04300</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03953</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03455</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03701</span></span>
<span class="line"><span>Validation: Loss 0.03643 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03551 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03334</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03273</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03163</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03277</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03231</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03184</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03541</span></span>
<span class="line"><span>Validation: Loss 0.03098 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03018 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02951</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02788</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02781</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02653</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02880</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02732</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02726</span></span>
<span class="line"><span>Validation: Loss 0.02697 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02626 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02719</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02324</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02516</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02386</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02326</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02406</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02489</span></span>
<span class="line"><span>Validation: Loss 0.02387 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02323 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02326</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02230</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02079</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02181</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02038</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02372</span></span>
<span class="line"><span>Validation: Loss 0.02137 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02079 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02048</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02123</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01922</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01688</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01934</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01938</span></span>
<span class="line"><span>Validation: Loss 0.01930 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01876 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01831</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01870</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01893</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01642</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01874</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01518</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01637</span></span>
<span class="line"><span>Validation: Loss 0.01755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01705 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01585</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01703</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01689</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01519</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01632</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01263</span></span>
<span class="line"><span>Validation: Loss 0.01607 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01560 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01474</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01555</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01537</span></span>
<span class="line"><span>Validation: Loss 0.01481 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01437 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01384</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01407</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01259</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01320</span></span>
<span class="line"><span>Validation: Loss 0.01370 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01329 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01219</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01242</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01385</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01271</span></span>
<span class="line"><span>Validation: Loss 0.01271 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01232 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01082</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01196</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01002</span></span>
<span class="line"><span>Validation: Loss 0.01178 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01141 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00977</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Validation: Loss 0.01083 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01049 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00733</span></span>
<span class="line"><span>Validation: Loss 0.00976 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00946 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00930</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00813</span></span>
<span class="line"><span>Validation: Loss 0.00868 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00842 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00734</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00787</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00759 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.5</span></span>
<span class="line"><span>Commit 6f3fdf7b362 (2024-08-27 14:19 UTC)</span></span>
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
<span class="line"><span>CUDA runtime 12.4, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.4.5</span></span>
<span class="line"><span>- CURAND: 10.3.5</span></span>
<span class="line"><span>- CUFFT: 11.2.1</span></span>
<span class="line"><span>- CUSOLVER: 11.6.1</span></span>
<span class="line"><span>- CUSPARSE: 12.3.1</span></span>
<span class="line"><span>- CUPTI: 22.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.3.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.8.1+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.12.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: Quadro RTX 5000 (sm_75, 15.203 GiB / 16.000 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
