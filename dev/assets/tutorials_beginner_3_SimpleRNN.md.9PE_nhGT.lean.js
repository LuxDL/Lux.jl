import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.H3Pxo4Yr.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62004</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58819</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55200</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54667</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52514</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50010</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48061</span></span>
<span class="line"><span>Validation: Loss 0.47137 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47847 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47617</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45156</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43162</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42964</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40722</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40032</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38129</span></span>
<span class="line"><span>Validation: Loss 0.37524 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38319 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36490</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35840</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34566</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33048</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33191</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30359</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31536</span></span>
<span class="line"><span>Validation: Loss 0.29092 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29950 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28159</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27876</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27063</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25769</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24332</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23272</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21558</span></span>
<span class="line"><span>Validation: Loss 0.22081 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22938 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21416</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21205</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19249</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18519</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18832</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17657</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17980</span></span>
<span class="line"><span>Validation: Loss 0.16448 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17255 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15800</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15383</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14715</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14082</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14146</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12349</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12420</span></span>
<span class="line"><span>Validation: Loss 0.12075 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12746 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12159</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11354</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11679</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09700</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09475</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08774</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08578</span></span>
<span class="line"><span>Validation: Loss 0.08648 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09146 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08795</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07815</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07214</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07852</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06545</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06264</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06648</span></span>
<span class="line"><span>Validation: Loss 0.06016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06353 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05758</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05961</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05171</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04829</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04949</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04635</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04671</span></span>
<span class="line"><span>Validation: Loss 0.04441 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04678 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04259</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04081</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03888</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04056</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03748</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03794</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04098</span></span>
<span class="line"><span>Validation: Loss 0.03589 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03783 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03652</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03243</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03243</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03274</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03170</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03283</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02539</span></span>
<span class="line"><span>Validation: Loss 0.03042 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03211 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02892</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03124</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02919</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02644</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02664</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02725</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02383</span></span>
<span class="line"><span>Validation: Loss 0.02648 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02799 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02514</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02402</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02596</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02402</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02341</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02507</span></span>
<span class="line"><span>Validation: Loss 0.02344 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02481 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02345</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02106</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02289</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02173</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02118</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02201</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01719</span></span>
<span class="line"><span>Validation: Loss 0.02098 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02224 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02077</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02012</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02276</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01855</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01916</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01885</span></span>
<span class="line"><span>Validation: Loss 0.01896 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02012 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01801</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01742</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01835</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01877</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01739</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01457</span></span>
<span class="line"><span>Validation: Loss 0.01725 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01833 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01708</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01572</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01530</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01573</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01448</span></span>
<span class="line"><span>Validation: Loss 0.01579 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01681 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01520</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01531</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01568</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01444</span></span>
<span class="line"><span>Validation: Loss 0.01455 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01550 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01400</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01370</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01395</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01311</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01545</span></span>
<span class="line"><span>Validation: Loss 0.01347 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01436 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01319</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01368</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01389</span></span>
<span class="line"><span>Validation: Loss 0.01251 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01334 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01263</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01222</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01014</span></span>
<span class="line"><span>Validation: Loss 0.01163 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01241 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01096</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01029</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01135</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01241</span></span>
<span class="line"><span>Validation: Loss 0.01080 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01152 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01047</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00947</span></span>
<span class="line"><span>Validation: Loss 0.00991 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01056 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00822</span></span>
<span class="line"><span>Validation: Loss 0.00890 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00947 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00933</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00840 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61469</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59599</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57055</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52888</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53192</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50103</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49576</span></span>
<span class="line"><span>Validation: Loss 0.46903 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45610 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47324</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45865</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43583</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42312</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40489</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40942</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40296</span></span>
<span class="line"><span>Validation: Loss 0.37260 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35756 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36842</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35514</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33729</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33978</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32092</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32890</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32008</span></span>
<span class="line"><span>Validation: Loss 0.28822 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27162 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28675</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27988</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27833</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26197</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24232</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23527</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21109</span></span>
<span class="line"><span>Validation: Loss 0.21859 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20201 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.23634</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21427</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20140</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18874</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18478</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17396</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14835</span></span>
<span class="line"><span>Validation: Loss 0.16301 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14758 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16895</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16473</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15231</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14217</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13036</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13023</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12994</span></span>
<span class="line"><span>Validation: Loss 0.12029 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10734 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12668</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10906</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10837</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11249</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09916</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09736</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09107</span></span>
<span class="line"><span>Validation: Loss 0.08673 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07701 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08903</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08724</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07644</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07660</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07539</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06349</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05695</span></span>
<span class="line"><span>Validation: Loss 0.06045 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05394 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06606</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05644</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05391</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05182</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05171</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04677</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04800</span></span>
<span class="line"><span>Validation: Loss 0.04448 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03989 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04344</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04367</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03982</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04157</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04100</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03949</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03440</span></span>
<span class="line"><span>Validation: Loss 0.03585 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03212 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03495</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03552</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03321</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03467</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03470</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03147</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02999</span></span>
<span class="line"><span>Validation: Loss 0.03038 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02712 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02776</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03156</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02887</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02807</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03119</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02767</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02596</span></span>
<span class="line"><span>Validation: Loss 0.02642 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02353 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02761</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02868</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02548</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02389</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02324</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02417</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02463</span></span>
<span class="line"><span>Validation: Loss 0.02336 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02075 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02336</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02360</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02321</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02224</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02149</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02215</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02122</span></span>
<span class="line"><span>Validation: Loss 0.02090 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01852 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02072</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01964</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02029</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02092</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02152</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01815</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02270</span></span>
<span class="line"><span>Validation: Loss 0.01887 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01668 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01993</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01879</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01827</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01750</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01677</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02068</span></span>
<span class="line"><span>Validation: Loss 0.01714 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01511 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01747</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01643</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01732</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01337</span></span>
<span class="line"><span>Validation: Loss 0.01565 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01376 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01625</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01621</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01680</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01400</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01455</span></span>
<span class="line"><span>Validation: Loss 0.01438 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01261 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01281</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01235</span></span>
<span class="line"><span>Validation: Loss 0.01326 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01162 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01320</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01319</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01437</span></span>
<span class="line"><span>Validation: Loss 0.01224 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01072 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01001</span></span>
<span class="line"><span>Validation: Loss 0.01123 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00984 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01120</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01072</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01106</span></span>
<span class="line"><span>Validation: Loss 0.01012 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00889 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00905</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01048</span></span>
<span class="line"><span>Validation: Loss 0.00893 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00789 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00896</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00928</span></span>
<span class="line"><span>Validation: Loss 0.00799 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00708 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00760</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00900</span></span>
<span class="line"><span>Validation: Loss 0.00732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00651 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
