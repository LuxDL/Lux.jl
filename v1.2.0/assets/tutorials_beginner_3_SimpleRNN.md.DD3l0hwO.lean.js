import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BgaJsmT1.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.2.0/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.2.0/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.2.0/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.2.0/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62718</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60001</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56268</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53181</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51388</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51194</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47500</span></span>
<span class="line"><span>Validation: Loss 0.46404 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47038 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47518</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45419</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43895</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41477</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42107</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40054</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37514</span></span>
<span class="line"><span>Validation: Loss 0.36615 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37387 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38160</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36141</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33932</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31804</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33407</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31477</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27711</span></span>
<span class="line"><span>Validation: Loss 0.28073 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28935 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28997</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28240</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25862</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24393</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25496</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24056</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21530</span></span>
<span class="line"><span>Validation: Loss 0.21101 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21973 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22257</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21121</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20601</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18332</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18898</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17421</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14663</span></span>
<span class="line"><span>Validation: Loss 0.15600 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16392 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16531</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15557</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14348</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14633</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13218</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13313</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12015</span></span>
<span class="line"><span>Validation: Loss 0.11405 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12051 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11521</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10906</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10325</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10190</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10488</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09975</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09609</span></span>
<span class="line"><span>Validation: Loss 0.08147 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08622 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08552</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07674</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07377</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07847</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07308</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06279</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05957</span></span>
<span class="line"><span>Validation: Loss 0.05666 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05977 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05632</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05646</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05808</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04899</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04856</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04833</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04532</span></span>
<span class="line"><span>Validation: Loss 0.04231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04448 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04456</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04584</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04199</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04006</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03755</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03498</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03402</span></span>
<span class="line"><span>Validation: Loss 0.03427 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03602 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03817</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03232</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03371</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03322</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03166</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03205</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03184</span></span>
<span class="line"><span>Validation: Loss 0.02910 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03062 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02991</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02674</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02942</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02688</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02973</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02892</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03049</span></span>
<span class="line"><span>Validation: Loss 0.02532 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02667 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02762</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02623</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02642</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02441</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02249</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02463</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02190</span></span>
<span class="line"><span>Validation: Loss 0.02237 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02357 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02340</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02384</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02214</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02104</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01959</span></span>
<span class="line"><span>Validation: Loss 0.02001 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02111 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02303</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02102</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01768</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01868</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01983</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02250</span></span>
<span class="line"><span>Validation: Loss 0.01805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01995</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01832</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01789</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01749</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01922</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01702</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01469</span></span>
<span class="line"><span>Validation: Loss 0.01639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01732 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01703</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01746</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01679</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01635</span></span>
<span class="line"><span>Validation: Loss 0.01497 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01584 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01549</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01544</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01394</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01392</span></span>
<span class="line"><span>Validation: Loss 0.01374 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01456 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01645</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01071</span></span>
<span class="line"><span>Validation: Loss 0.01267 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01343 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01253</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01350</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00991</span></span>
<span class="line"><span>Validation: Loss 0.01170 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01241 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01196</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01135</span></span>
<span class="line"><span>Validation: Loss 0.01074 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01139 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01047</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01052</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01061</span></span>
<span class="line"><span>Validation: Loss 0.00967 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01025 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01008</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00770</span></span>
<span class="line"><span>Validation: Loss 0.00857 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00907 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00811</span></span>
<span class="line"><span>Validation: Loss 0.00771 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00814 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00774</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00744</span></span>
<span class="line"><span>Validation: Loss 0.00709 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00747 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62108</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58121</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56262</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54752</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52864</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49979</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46702</span></span>
<span class="line"><span>Validation: Loss 0.47471 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46144 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46684</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44808</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44013</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42881</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41686</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40113</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39721</span></span>
<span class="line"><span>Validation: Loss 0.37957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36353 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37677</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36023</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34351</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33917</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32752</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30449</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28897</span></span>
<span class="line"><span>Validation: Loss 0.29574 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27458</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27470</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26752</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26272</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24863</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24116</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24974</span></span>
<span class="line"><span>Validation: Loss 0.22620 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20871 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21805</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21271</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19800</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20039</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18249</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17278</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18034</span></span>
<span class="line"><span>Validation: Loss 0.16975 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15377 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16734</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15922</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15422</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14687</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12528</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13232</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10874</span></span>
<span class="line"><span>Validation: Loss 0.12526 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11210 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12720</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11364</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10651</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09527</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09629</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09677</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11688</span></span>
<span class="line"><span>Validation: Loss 0.09022 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07898</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08638</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08011</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07609</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06828</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06253</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07626</span></span>
<span class="line"><span>Validation: Loss 0.06259 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05604 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05683</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05699</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05471</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05135</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04590</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05185</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05025</span></span>
<span class="line"><span>Validation: Loss 0.04591 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04145 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04658</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04182</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04112</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03920</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03553</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03901</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03768</span></span>
<span class="line"><span>Validation: Loss 0.03693 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03337 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03345</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03682</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03255</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03261</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03287</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03009</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03583</span></span>
<span class="line"><span>Validation: Loss 0.03127 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02819 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02902</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02865</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02972</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02890</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02737</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02684</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02747</span></span>
<span class="line"><span>Validation: Loss 0.02718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02445 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02694</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02756</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02620</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02382</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02297</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02257</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02096</span></span>
<span class="line"><span>Validation: Loss 0.02403 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02157 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02281</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02363</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02247</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02308</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01987</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02068</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02092</span></span>
<span class="line"><span>Validation: Loss 0.02153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01927 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02226</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01992</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02096</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01800</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01844</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01404</span></span>
<span class="line"><span>Validation: Loss 0.01946 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01737 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01877</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01920</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01775</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01801</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01712</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01690</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01712</span></span>
<span class="line"><span>Validation: Loss 0.01773 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01578 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01852</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01655</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01633</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01572</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01569</span></span>
<span class="line"><span>Validation: Loss 0.01624 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01442 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01549</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01472</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01498</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01510</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01462</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01556</span></span>
<span class="line"><span>Validation: Loss 0.01497 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01326 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01520</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01428</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01439</span></span>
<span class="line"><span>Validation: Loss 0.01385 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01226 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01332</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01263</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01237</span></span>
<span class="line"><span>Validation: Loss 0.01286 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01137 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01150</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01240</span></span>
<span class="line"><span>Validation: Loss 0.01195 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01056 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01239</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01136</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01020</span></span>
<span class="line"><span>Validation: Loss 0.01104 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00858</span></span>
<span class="line"><span>Validation: Loss 0.01006 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00891 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00838</span></span>
<span class="line"><span>Validation: Loss 0.00896 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00796 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00896</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00717</span></span>
<span class="line"><span>Validation: Loss 0.00798 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00713 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>CUDA runtime 12.6, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.3</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.3+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.3+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
