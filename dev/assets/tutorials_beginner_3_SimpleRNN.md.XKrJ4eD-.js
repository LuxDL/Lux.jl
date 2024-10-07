import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.CZ4L7dST.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61933</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59413</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56541</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54703</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51594</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50897</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46746</span></span>
<span class="line"><span>Validation: Loss 0.46145 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46667 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46219</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45655</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44383</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42797</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41370</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40024</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39355</span></span>
<span class="line"><span>Validation: Loss 0.36317 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36913 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37717</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36670</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34940</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33841</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31580</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30711</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27112</span></span>
<span class="line"><span>Validation: Loss 0.27756 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28388 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.30038</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27374</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27295</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25435</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24063</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22313</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25175</span></span>
<span class="line"><span>Validation: Loss 0.20767 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21417 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21127</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21070</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19811</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19271</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19130</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17722</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18306</span></span>
<span class="line"><span>Validation: Loss 0.15292 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15907 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15878</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16426</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14304</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14877</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14674</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11859</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12351</span></span>
<span class="line"><span>Validation: Loss 0.11134 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11650 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10892</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12404</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10592</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11130</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09921</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09284</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08573</span></span>
<span class="line"><span>Validation: Loss 0.07957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08342 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08409</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08575</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08076</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07631</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06342</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06723</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05351</span></span>
<span class="line"><span>Validation: Loss 0.05555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05812 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05864</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05426</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05326</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05424</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04758</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04956</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05283</span></span>
<span class="line"><span>Validation: Loss 0.04139 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04324 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04746</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04100</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04399</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03751</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04102</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03412</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03995</span></span>
<span class="line"><span>Validation: Loss 0.03345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03495 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03433</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03485</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03287</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03371</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03506</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02988</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03694</span></span>
<span class="line"><span>Validation: Loss 0.02832 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02963 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02681</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03169</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03055</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02802</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02791</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02832</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02661</span></span>
<span class="line"><span>Validation: Loss 0.02459 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02575 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02884</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02567</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02649</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02475</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02389</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02287</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02107</span></span>
<span class="line"><span>Validation: Loss 0.02171 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02275 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02332</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02254</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02176</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02201</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02302</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02250</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01958</span></span>
<span class="line"><span>Validation: Loss 0.01943 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02038 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02127</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01986</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01939</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02105</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02075</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02012</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01417</span></span>
<span class="line"><span>Validation: Loss 0.01753 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01841 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01821</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01897</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01771</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02026</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01724</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01750</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01725</span></span>
<span class="line"><span>Validation: Loss 0.01593 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01676 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01768</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01619</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01707</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01700</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01694</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01478</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01816</span></span>
<span class="line"><span>Validation: Loss 0.01455 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01532 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01572</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01614</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01534</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01588</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01469</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01439</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01335</span></span>
<span class="line"><span>Validation: Loss 0.01335 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01407 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01500</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01401</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01549</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01355</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01380</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01181</span></span>
<span class="line"><span>Validation: Loss 0.01231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01298 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01373</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01238</span></span>
<span class="line"><span>Validation: Loss 0.01140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01202 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01358</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01007</span></span>
<span class="line"><span>Validation: Loss 0.01053 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01111 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00995</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01129</span></span>
<span class="line"><span>Validation: Loss 0.00965 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01016 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01016</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00906</span></span>
<span class="line"><span>Validation: Loss 0.00864 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00908 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00894</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00912</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00804</span></span>
<span class="line"><span>Validation: Loss 0.00769 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00725</span></span>
<span class="line"><span>Validation: Loss 0.00698 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00732 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63141</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59863</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57044</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54545</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51395</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49460</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47930</span></span>
<span class="line"><span>Validation: Loss 0.46531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46020 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46838</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45652</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44017</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43425</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41757</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38774</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37056</span></span>
<span class="line"><span>Validation: Loss 0.36681 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36038 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36578</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36394</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35750</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32877</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31563</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30613</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30609</span></span>
<span class="line"><span>Validation: Loss 0.28118 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27410 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28210</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29477</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26601</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26565</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22709</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22042</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25023</span></span>
<span class="line"><span>Validation: Loss 0.21139 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20460 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22970</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20902</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19435</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19022</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17922</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17473</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16022</span></span>
<span class="line"><span>Validation: Loss 0.15633 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15050 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16045</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14794</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14020</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14028</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13757</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14020</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13142</span></span>
<span class="line"><span>Validation: Loss 0.11422 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10965 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11410</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11067</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10796</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11001</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09179</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09908</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08470</span></span>
<span class="line"><span>Validation: Loss 0.08153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07829 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08656</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07813</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07627</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07736</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06924</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06226</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05531</span></span>
<span class="line"><span>Validation: Loss 0.05682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05472 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05818</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06028</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04857</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05198</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05028</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04684</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04510</span></span>
<span class="line"><span>Validation: Loss 0.04246 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04095 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04568</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04274</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04343</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03652</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03960</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03697</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03267</span></span>
<span class="line"><span>Validation: Loss 0.03445 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03320 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03590</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03447</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03163</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03445</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03153</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03347</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03077</span></span>
<span class="line"><span>Validation: Loss 0.02929 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02820 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03132</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03016</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02759</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02897</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02770</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02789</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02480</span></span>
<span class="line"><span>Validation: Loss 0.02550 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02454 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02594</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02660</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02437</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02720</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02395</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02418</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02191</span></span>
<span class="line"><span>Validation: Loss 0.02258 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02171 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02255</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02302</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02227</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02268</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02255</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02239</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01957</span></span>
<span class="line"><span>Validation: Loss 0.02023 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01944 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02229</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02048</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02043</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01801</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01951</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01906</span></span>
<span class="line"><span>Validation: Loss 0.01828 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01756 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01878</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01821</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01796</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01888</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01863</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01822</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01539</span></span>
<span class="line"><span>Validation: Loss 0.01663 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01679</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01689</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01636</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01832</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01764</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01514</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01343</span></span>
<span class="line"><span>Validation: Loss 0.01521 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01459 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01477</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01547</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01646</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01524</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01459</span></span>
<span class="line"><span>Validation: Loss 0.01398 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01340 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01406</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01391</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01239</span></span>
<span class="line"><span>Validation: Loss 0.01290 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01236 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01325</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01293</span></span>
<span class="line"><span>Validation: Loss 0.01195 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01144 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01206</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01205</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01286</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01249</span></span>
<span class="line"><span>Validation: Loss 0.01107 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01060 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01173</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01059</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01133</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01087</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00938</span></span>
<span class="line"><span>Validation: Loss 0.01018 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00975 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01036</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00850</span></span>
<span class="line"><span>Validation: Loss 0.00920 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00883 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00858</span></span>
<span class="line"><span>Validation: Loss 0.00818 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00786 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00795</span></span>
<span class="line"><span>Validation: Loss 0.00736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00708 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
