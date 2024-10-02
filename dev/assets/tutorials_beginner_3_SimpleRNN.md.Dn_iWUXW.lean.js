import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.C3QLvTLa.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61497</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59549</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56107</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54044</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53382</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48871</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47536</span></span>
<span class="line"><span>Validation: Loss 0.47617 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46729 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46736</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45060</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43330</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42850</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41878</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39378</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39544</span></span>
<span class="line"><span>Validation: Loss 0.38169 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37085 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36814</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36988</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33527</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33350</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32610</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30730</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29996</span></span>
<span class="line"><span>Validation: Loss 0.29830 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28656 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28490</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27749</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27255</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26279</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23716</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23528</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21047</span></span>
<span class="line"><span>Validation: Loss 0.22854 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21677 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22403</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19470</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19183</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20578</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18298</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18259</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15348</span></span>
<span class="line"><span>Validation: Loss 0.17210 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16113 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16465</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15490</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14292</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14320</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12902</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14137</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11250</span></span>
<span class="line"><span>Validation: Loss 0.12750 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11834 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12083</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10991</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10869</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10156</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09695</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09778</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09756</span></span>
<span class="line"><span>Validation: Loss 0.09192 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08508 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08215</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08293</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07514</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07938</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06263</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07055</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06343</span></span>
<span class="line"><span>Validation: Loss 0.06386 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05929 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05498</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05665</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05394</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05212</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05270</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04682</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04740</span></span>
<span class="line"><span>Validation: Loss 0.04680 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04362 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04646</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04264</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03814</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03990</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03945</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03485</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04153</span></span>
<span class="line"><span>Validation: Loss 0.03766 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03509 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03321</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03556</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03350</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03462</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03086</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03094</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03189</span></span>
<span class="line"><span>Validation: Loss 0.03190 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03090</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02771</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02717</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02948</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02674</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02787</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02757</span></span>
<span class="line"><span>Validation: Loss 0.02774 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02577 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02637</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02701</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02435</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02367</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02452</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02349</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02100</span></span>
<span class="line"><span>Validation: Loss 0.02453 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02275 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02250</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02352</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02236</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02180</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02243</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02026</span></span>
<span class="line"><span>Validation: Loss 0.02197 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01936</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01981</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01968</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01895</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01956</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02102</span></span>
<span class="line"><span>Validation: Loss 0.01986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01836 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01867</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01789</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01863</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01832</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01740</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01590</span></span>
<span class="line"><span>Validation: Loss 0.01807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01667 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01610</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01603</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01739</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01450</span></span>
<span class="line"><span>Validation: Loss 0.01654 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01524 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01713</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01576</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01422</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01419</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01275</span></span>
<span class="line"><span>Validation: Loss 0.01523 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01402 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01453</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01070</span></span>
<span class="line"><span>Validation: Loss 0.01410 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01296 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01305</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01163</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01163</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01286</span></span>
<span class="line"><span>Validation: Loss 0.01308 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01202 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01097</span></span>
<span class="line"><span>Validation: Loss 0.01212 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01114 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01074</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00997</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01132</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01098</span></span>
<span class="line"><span>Validation: Loss 0.01111 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01023 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00958</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00900</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01059</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01063</span></span>
<span class="line"><span>Validation: Loss 0.00998 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00920 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00907</span></span>
<span class="line"><span>Validation: Loss 0.00882 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00816 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00713</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00839</span></span>
<span class="line"><span>Validation: Loss 0.00794 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00736 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61561</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58044</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56116</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54907</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51083</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50473</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47515</span></span>
<span class="line"><span>Validation: Loss 0.48007 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.48869 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45329</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44948</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45501</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41246</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41396</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39810</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40365</span></span>
<span class="line"><span>Validation: Loss 0.38676 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.39741 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36474</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35985</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35346</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32350</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31948</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31497</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29622</span></span>
<span class="line"><span>Validation: Loss 0.30403 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.31548 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29649</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28371</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25898</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25239</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23481</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23188</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24696</span></span>
<span class="line"><span>Validation: Loss 0.23420 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.24548 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21886</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20664</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19126</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19505</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18283</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17849</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17781</span></span>
<span class="line"><span>Validation: Loss 0.17699 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18708 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15931</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15939</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15179</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13175</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13972</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12606</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13119</span></span>
<span class="line"><span>Validation: Loss 0.13111 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13921 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12326</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10958</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10685</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10283</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10170</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08896</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09105</span></span>
<span class="line"><span>Validation: Loss 0.09390 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09979 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08766</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07365</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07756</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07295</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07344</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06016</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06772</span></span>
<span class="line"><span>Validation: Loss 0.06492 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06880 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05767</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05390</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05228</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05189</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05095</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04656</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04948</span></span>
<span class="line"><span>Validation: Loss 0.04816 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05088 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04445</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04473</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03910</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04093</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03708</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03758</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03094</span></span>
<span class="line"><span>Validation: Loss 0.03902 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04123 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03410</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03388</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03506</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03080</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03348</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03244</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03039</span></span>
<span class="line"><span>Validation: Loss 0.03322 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03514 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03241</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03069</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02901</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02870</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02636</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02562</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02148</span></span>
<span class="line"><span>Validation: Loss 0.02895 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03066 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02585</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02585</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02562</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02561</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02541</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02298</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01797</span></span>
<span class="line"><span>Validation: Loss 0.02568 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02723 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02459</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02227</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02234</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02241</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01959</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02091</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02558</span></span>
<span class="line"><span>Validation: Loss 0.02306 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02448 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02207</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01860</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02082</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01870</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01898</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01801</span></span>
<span class="line"><span>Validation: Loss 0.02085 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02215 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01938</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01758</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01823</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01940</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01634</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01491</span></span>
<span class="line"><span>Validation: Loss 0.01899 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02020 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01744</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01756</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01600</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01662</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01510</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01522</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01808</span></span>
<span class="line"><span>Validation: Loss 0.01740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01853 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01659</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01546</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01523</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01368</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01412</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01424</span></span>
<span class="line"><span>Validation: Loss 0.01602 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01707 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01393</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01452</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01319</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01433</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01431</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01426</span></span>
<span class="line"><span>Validation: Loss 0.01481 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01579 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01371</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01178</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01268</span></span>
<span class="line"><span>Validation: Loss 0.01370 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01462 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01150</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01074</span></span>
<span class="line"><span>Validation: Loss 0.01263 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01348 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01202</span></span>
<span class="line"><span>Validation: Loss 0.01147 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01224 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01004</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00988</span></span>
<span class="line"><span>Validation: Loss 0.01016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01082 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00810</span></span>
<span class="line"><span>Validation: Loss 0.00900 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00957 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00739</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00766</span></span>
<span class="line"><span>Validation: Loss 0.00820 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00870 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
