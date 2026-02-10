import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.c5uhSXva.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.3.4/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.3.4/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.3.4/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.3.4/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61453</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59243</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56526</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54178</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52307</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49751</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48688</span></span>
<span class="line"><span>Validation: Loss 0.46892 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46603 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45749</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45752</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44836</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42228</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41130</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40114</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37467</span></span>
<span class="line"><span>Validation: Loss 0.37161 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36814 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38008</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35219</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34750</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33244</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32540</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30633</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26119</span></span>
<span class="line"><span>Validation: Loss 0.28683 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28290 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27922</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28308</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27231</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24842</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24707</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23129</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22081</span></span>
<span class="line"><span>Validation: Loss 0.21783 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21373 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21684</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20931</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21209</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19558</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17426</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16805</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16822</span></span>
<span class="line"><span>Validation: Loss 0.16272 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15894 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16020</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14623</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13986</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15242</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13891</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12584</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15504</span></span>
<span class="line"><span>Validation: Loss 0.11991 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11681 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11641</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12243</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11302</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10348</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09517</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08958</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07990</span></span>
<span class="line"><span>Validation: Loss 0.08583 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08359 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08263</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08199</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08579</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07038</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06689</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06069</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07326</span></span>
<span class="line"><span>Validation: Loss 0.05976 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05830 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06201</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06054</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05309</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05027</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04856</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04365</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04361</span></span>
<span class="line"><span>Validation: Loss 0.04394 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04295 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04252</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04446</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04069</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04039</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03668</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03685</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03488</span></span>
<span class="line"><span>Validation: Loss 0.03540 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03460 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03621</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03394</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03340</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03121</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03415</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02958</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02935</span></span>
<span class="line"><span>Validation: Loss 0.02998 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02929 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02952</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02671</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02943</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03006</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02760</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02620</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02636</span></span>
<span class="line"><span>Validation: Loss 0.02607 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02545 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02239</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02604</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02621</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02447</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02466</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02445</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02324</span></span>
<span class="line"><span>Validation: Loss 0.02305 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02249 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02541</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02190</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02094</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02080</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02159</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02090</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02148</span></span>
<span class="line"><span>Validation: Loss 0.02061 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02009 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02206</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01957</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02051</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01829</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01838</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01949</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01803</span></span>
<span class="line"><span>Validation: Loss 0.01859 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01811 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01751</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01808</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01791</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01719</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01883</span></span>
<span class="line"><span>Validation: Loss 0.01689 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01645 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01667</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01585</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01639</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01620</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01630</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01489</span></span>
<span class="line"><span>Validation: Loss 0.01545 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01504 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01430</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01476</span></span>
<span class="line"><span>Validation: Loss 0.01421 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01383 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01530</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01417</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01312</span></span>
<span class="line"><span>Validation: Loss 0.01313 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01277 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01355</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01310</span></span>
<span class="line"><span>Validation: Loss 0.01216 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01183 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01120</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00993</span></span>
<span class="line"><span>Validation: Loss 0.01123 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01093 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01005</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01112</span></span>
<span class="line"><span>Validation: Loss 0.01026 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00999 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00900</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00994</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01165</span></span>
<span class="line"><span>Validation: Loss 0.00918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00894 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00966</span></span>
<span class="line"><span>Validation: Loss 0.00814 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00793 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Validation: Loss 0.00738 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00720 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62219</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59276</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55983</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54930</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51796</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50347</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48278</span></span>
<span class="line"><span>Validation: Loss 0.46349 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46455 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46559</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45462</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43891</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43242</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41942</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39238</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37993</span></span>
<span class="line"><span>Validation: Loss 0.36509 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36619 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37283</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35590</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34827</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32381</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32661</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31426</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29730</span></span>
<span class="line"><span>Validation: Loss 0.27915 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28883</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27465</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26130</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24991</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24886</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23842</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22975</span></span>
<span class="line"><span>Validation: Loss 0.20896 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21025 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21262</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20886</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19512</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19980</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18216</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17696</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16177</span></span>
<span class="line"><span>Validation: Loss 0.15372 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15510 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16420</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14920</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14400</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15145</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12623</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13021</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13497</span></span>
<span class="line"><span>Validation: Loss 0.11202 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11330 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11323</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11843</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10933</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10795</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09362</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09097</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08788</span></span>
<span class="line"><span>Validation: Loss 0.08015 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08113 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08178</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07825</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07771</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07457</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06921</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06664</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06374</span></span>
<span class="line"><span>Validation: Loss 0.05604 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05669 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05913</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05745</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05215</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05102</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04797</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03756</span></span>
<span class="line"><span>Validation: Loss 0.04138 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04184 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04689</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04123</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03834</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04142</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03869</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03502</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03346</span></span>
<span class="line"><span>Validation: Loss 0.03337 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03374 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03420</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03600</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03172</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03482</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03199</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02977</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02810</span></span>
<span class="line"><span>Validation: Loss 0.02824 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02857 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03090</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03075</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02543</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02789</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02710</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02684</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02873</span></span>
<span class="line"><span>Validation: Loss 0.02454 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02604</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02703</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02407</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02322</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02484</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02322</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02303</span></span>
<span class="line"><span>Validation: Loss 0.02167 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02193 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02342</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02245</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02209</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02118</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02029</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02234</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02059</span></span>
<span class="line"><span>Validation: Loss 0.01936 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01961 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02060</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01960</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02051</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01953</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01947</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01854</span></span>
<span class="line"><span>Validation: Loss 0.01745 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01760</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01906</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01753</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01791</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01871</span></span>
<span class="line"><span>Validation: Loss 0.01584 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01605 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01660</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01626</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01532</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01824</span></span>
<span class="line"><span>Validation: Loss 0.01446 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01466 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01428</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01620</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01485</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01517</span></span>
<span class="line"><span>Validation: Loss 0.01328 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01347 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01474</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01412</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01370</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01101</span></span>
<span class="line"><span>Validation: Loss 0.01226 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01244 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01161</span></span>
<span class="line"><span>Validation: Loss 0.01137 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01154 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01171</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01237</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01337</span></span>
<span class="line"><span>Validation: Loss 0.01055 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01070 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01097</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01116</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01153</span></span>
<span class="line"><span>Validation: Loss 0.00972 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00986 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00908</span></span>
<span class="line"><span>Validation: Loss 0.00880 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00893 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01011</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00911</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00889</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00833</span></span>
<span class="line"><span>Validation: Loss 0.00784 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00794 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00725</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00774</span></span>
<span class="line"><span>Validation: Loss 0.00704 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00714 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- CUDA_Runtime_jll: 0.15.4+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.6</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
