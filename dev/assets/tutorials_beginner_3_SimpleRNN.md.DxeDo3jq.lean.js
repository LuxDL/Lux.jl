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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62879</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59290</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56859</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54242</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52131</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49154</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47190</span></span>
<span class="line"><span>Validation: Loss 0.46934 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46831 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47241</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45343</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43831</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42555</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40999</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40067</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35972</span></span>
<span class="line"><span>Validation: Loss 0.37202 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37120 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37223</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35098</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35167</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33554</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32809</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30061</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28135</span></span>
<span class="line"><span>Validation: Loss 0.28758 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28681 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29018</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28130</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27192</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25742</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23897</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22032</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22678</span></span>
<span class="line"><span>Validation: Loss 0.21809 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21731 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20773</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21503</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20517</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18513</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17973</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18136</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16872</span></span>
<span class="line"><span>Validation: Loss 0.16267 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16182 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17538</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14465</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14475</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13941</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13430</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13104</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12826</span></span>
<span class="line"><span>Validation: Loss 0.11953 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11874 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11744</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11131</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11092</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10389</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09571</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09396</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09236</span></span>
<span class="line"><span>Validation: Loss 0.08558 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08501 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08554</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08279</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07667</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07202</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07082</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06329</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05548</span></span>
<span class="line"><span>Validation: Loss 0.05945 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05910 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05457</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05565</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05800</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05069</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04941</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04744</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04233</span></span>
<span class="line"><span>Validation: Loss 0.04400 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04371 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04618</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04254</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04081</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03567</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03929</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03785</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03379</span></span>
<span class="line"><span>Validation: Loss 0.03556 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03530 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03353</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03438</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03438</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03304</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03303</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03074</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02956</span></span>
<span class="line"><span>Validation: Loss 0.03018 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02995 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03018</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03076</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02798</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02718</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02804</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02543</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03013</span></span>
<span class="line"><span>Validation: Loss 0.02626 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02605 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02648</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02622</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02401</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02486</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02317</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02401</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02502</span></span>
<span class="line"><span>Validation: Loss 0.02321 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02302 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02362</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02383</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01971</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02229</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02137</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02179</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02013</span></span>
<span class="line"><span>Validation: Loss 0.02076 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02059 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02132</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01883</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02087</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02022</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01894</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01951</span></span>
<span class="line"><span>Validation: Loss 0.01874 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01859 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01909</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01781</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01756</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01813</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01725</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01767</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01713</span></span>
<span class="line"><span>Validation: Loss 0.01704 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01689 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01798</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01681</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01645</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01531</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01603</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01567</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01442</span></span>
<span class="line"><span>Validation: Loss 0.01558 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01544 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01423</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01534</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01456</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01622</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01103</span></span>
<span class="line"><span>Validation: Loss 0.01433 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01421 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01447</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01342</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01431</span></span>
<span class="line"><span>Validation: Loss 0.01326 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01315 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01336</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01247</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01196</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01087</span></span>
<span class="line"><span>Validation: Loss 0.01231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01221 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01217</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01192</span></span>
<span class="line"><span>Validation: Loss 0.01146 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01136 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01066</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01237</span></span>
<span class="line"><span>Validation: Loss 0.01063 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01055 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01087</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00993</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01016</span></span>
<span class="line"><span>Validation: Loss 0.00976 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00969 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00818</span></span>
<span class="line"><span>Validation: Loss 0.00877 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00872 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00724</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00899</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00776 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61806</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59540</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56553</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52770</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51895</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50755</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50807</span></span>
<span class="line"><span>Validation: Loss 0.46596 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47067 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46892</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44708</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45241</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43341</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41254</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39365</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37511</span></span>
<span class="line"><span>Validation: Loss 0.36978 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37511 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37199</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36149</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35622</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33089</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31685</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31606</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29975</span></span>
<span class="line"><span>Validation: Loss 0.28512 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29116 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29512</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27590</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26894</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27406</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24276</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22749</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21205</span></span>
<span class="line"><span>Validation: Loss 0.21559 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22188 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22396</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21101</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19959</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19995</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17903</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18380</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16122</span></span>
<span class="line"><span>Validation: Loss 0.16040 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16659 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17724</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15772</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16094</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13099</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13486</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13119</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12217</span></span>
<span class="line"><span>Validation: Loss 0.11800 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12336 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.13236</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11919</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10680</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09725</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10067</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09678</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09480</span></span>
<span class="line"><span>Validation: Loss 0.08466 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08872 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08870</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08522</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08732</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07281</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06999</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05988</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06596</span></span>
<span class="line"><span>Validation: Loss 0.05881 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06149 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06193</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06100</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05198</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05109</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05193</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04802</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04163</span></span>
<span class="line"><span>Validation: Loss 0.04345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04528 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04427</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04671</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04128</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03929</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03910</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03668</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03816</span></span>
<span class="line"><span>Validation: Loss 0.03502 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03649 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03565</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03834</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03357</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03616</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03341</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02682</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03001</span></span>
<span class="line"><span>Validation: Loss 0.02964 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03091 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03047</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02874</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03021</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02985</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02602</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02895</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02599</span></span>
<span class="line"><span>Validation: Loss 0.02575 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02688 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02716</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02457</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02436</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02662</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02573</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02332</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02549</span></span>
<span class="line"><span>Validation: Loss 0.02274 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02376 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02379</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02341</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02077</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02267</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02431</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02017</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02123</span></span>
<span class="line"><span>Validation: Loss 0.02032 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02125 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02265</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02005</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01903</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02130</span></span>
<span class="line"><span>Validation: Loss 0.01831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01917 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01903</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01840</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01779</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01936</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01725</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01935</span></span>
<span class="line"><span>Validation: Loss 0.01662 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01741 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01526</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01878</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01677</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01624</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01767</span></span>
<span class="line"><span>Validation: Loss 0.01518 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01591 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01697</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01586</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01576</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01394</span></span>
<span class="line"><span>Validation: Loss 0.01393 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01461 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01492</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01362</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01469</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01446</span></span>
<span class="line"><span>Validation: Loss 0.01284 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01348 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01401</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01189</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01233</span></span>
<span class="line"><span>Validation: Loss 0.01184 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01243 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01205</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01157</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01320</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01136</span></span>
<span class="line"><span>Validation: Loss 0.01083 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01137 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01095</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01137</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00858</span></span>
<span class="line"><span>Validation: Loss 0.00971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01019 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00947</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00951</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00800</span></span>
<span class="line"><span>Validation: Loss 0.00860 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00901 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00689</span></span>
<span class="line"><span>Validation: Loss 0.00774 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00809 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00688</span></span>
<span class="line"><span>Validation: Loss 0.00714 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00746 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
