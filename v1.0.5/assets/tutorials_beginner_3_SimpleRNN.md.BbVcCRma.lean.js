import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BLFmGzHg.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.0.5/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.0.5/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.0.5/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.0.5/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62697</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59398</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56516</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53496</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50859</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50693</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47860</span></span>
<span class="line"><span>Validation: Loss 0.47768 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46095 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47312</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44743</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42440</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42461</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41525</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40489</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40008</span></span>
<span class="line"><span>Validation: Loss 0.38270 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36255 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36150</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36675</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34973</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33199</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32029</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30015</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31455</span></span>
<span class="line"><span>Validation: Loss 0.29887 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27710 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28120</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27205</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27875</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24838</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23269</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23989</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24375</span></span>
<span class="line"><span>Validation: Loss 0.22886 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20708 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21448</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21254</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18882</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18588</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18461</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17709</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18849</span></span>
<span class="line"><span>Validation: Loss 0.17205 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15199 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17380</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14316</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14159</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15160</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12746</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12037</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14508</span></span>
<span class="line"><span>Validation: Loss 0.12684 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11594</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10788</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11033</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10059</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10187</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09224</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07907</span></span>
<span class="line"><span>Validation: Loss 0.09069 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07172</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07587</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08157</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07087</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07289</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06797</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05956</span></span>
<span class="line"><span>Validation: Loss 0.06309 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05488 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05936</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05396</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05191</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05135</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04774</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04779</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04686</span></span>
<span class="line"><span>Validation: Loss 0.04689 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04105 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04626</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04220</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03982</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03832</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03782</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03671</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03514</span></span>
<span class="line"><span>Validation: Loss 0.03801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03326 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03595</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03364</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03255</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03230</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03105</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03117</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03813</span></span>
<span class="line"><span>Validation: Loss 0.03236 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02822 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03086</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02945</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02782</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02872</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02506</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02741</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03039</span></span>
<span class="line"><span>Validation: Loss 0.02818 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02451 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02362</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02790</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02369</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02477</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02390</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02561</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02100</span></span>
<span class="line"><span>Validation: Loss 0.02495 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02164 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02353</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02275</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02317</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02103</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02148</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02330</span></span>
<span class="line"><span>Validation: Loss 0.02237 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01934 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02015</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02168</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02007</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01978</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01967</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01728</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01964</span></span>
<span class="line"><span>Validation: Loss 0.02022 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01743 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01868</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01775</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01732</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01884</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01776</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01590</span></span>
<span class="line"><span>Validation: Loss 0.01842 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01582 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01613</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01515</span></span>
<span class="line"><span>Validation: Loss 0.01689 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01446 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01585</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01639</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01385</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01543</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01423</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01436</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01335</span></span>
<span class="line"><span>Validation: Loss 0.01556 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01329 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01473</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01373</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01436</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01300</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01338</span></span>
<span class="line"><span>Validation: Loss 0.01441 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01229 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01298</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01206</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01293</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01241</span></span>
<span class="line"><span>Validation: Loss 0.01339 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01140 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01303</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01155</span></span>
<span class="line"><span>Validation: Loss 0.01243 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01058 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01068</span></span>
<span class="line"><span>Validation: Loss 0.01148 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00977 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01027</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01047</span></span>
<span class="line"><span>Validation: Loss 0.01041 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00888 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00904</span></span>
<span class="line"><span>Validation: Loss 0.00923 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00773</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00859</span></span>
<span class="line"><span>Validation: Loss 0.00824 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00711 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61677</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59664</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56696</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54556</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52115</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49877</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47923</span></span>
<span class="line"><span>Validation: Loss 0.46639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46451 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46974</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46087</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44181</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41903</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41014</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40046</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37173</span></span>
<span class="line"><span>Validation: Loss 0.36831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36584 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37839</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34011</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34844</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33601</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32274</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30881</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31293</span></span>
<span class="line"><span>Validation: Loss 0.28307 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28024 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29027</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27727</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26092</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25209</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24520</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23017</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24575</span></span>
<span class="line"><span>Validation: Loss 0.21313 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21031 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22150</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20521</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19930</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18867</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18568</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17074</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17204</span></span>
<span class="line"><span>Validation: Loss 0.15746 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15496 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15970</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13205</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15669</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15128</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13629</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11765</span></span>
<span class="line"><span>Validation: Loss 0.11495 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11296 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11980</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11451</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11211</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09977</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09863</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08840</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07912</span></span>
<span class="line"><span>Validation: Loss 0.08228 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08082 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08656</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08304</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07644</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06626</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07073</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06500</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05950</span></span>
<span class="line"><span>Validation: Loss 0.05750 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05654 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05698</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05342</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05181</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05536</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05212</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04511</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04361</span></span>
<span class="line"><span>Validation: Loss 0.04263 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04200 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04439</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04347</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04021</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03888</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03598</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03881</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03483</span></span>
<span class="line"><span>Validation: Loss 0.03447 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03398 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03252</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03497</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03439</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03305</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03149</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03206</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03145</span></span>
<span class="line"><span>Validation: Loss 0.02924 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02774</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03031</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03163</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02678</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02578</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02732</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02995</span></span>
<span class="line"><span>Validation: Loss 0.02543 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02505 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02522</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02474</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02833</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02573</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02098</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02446</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02244</span></span>
<span class="line"><span>Validation: Loss 0.02247 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02214 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02482</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02483</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02115</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02101</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02296</span></span>
<span class="line"><span>Validation: Loss 0.02009 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01979 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02099</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02087</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02019</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01886</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01825</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02070</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01535</span></span>
<span class="line"><span>Validation: Loss 0.01813 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01786 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01615</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01854</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01796</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01724</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01940</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01640</span></span>
<span class="line"><span>Validation: Loss 0.01649 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01624 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01617</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01668</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01700</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01557</span></span>
<span class="line"><span>Validation: Loss 0.01508 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01484 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01518</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01400</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01366</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01473</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01483</span></span>
<span class="line"><span>Validation: Loss 0.01385 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01363 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01521</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01261</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01432</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01362</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01371</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01267</span></span>
<span class="line"><span>Validation: Loss 0.01279 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01258 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01352</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01287</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01223</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00935</span></span>
<span class="line"><span>Validation: Loss 0.01184 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01164 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01237</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01098</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00998</span></span>
<span class="line"><span>Validation: Loss 0.01096 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01078 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01029</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01100</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01121</span></span>
<span class="line"><span>Validation: Loss 0.01007 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00990 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00829</span></span>
<span class="line"><span>Validation: Loss 0.00905 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00890 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00895</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00773</span></span>
<span class="line"><span>Validation: Loss 0.00805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00741</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00643</span></span>
<span class="line"><span>Validation: Loss 0.00727 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00716 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.2+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
