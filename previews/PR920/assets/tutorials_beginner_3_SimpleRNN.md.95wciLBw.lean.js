import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BAcBqT1O.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR920/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR920/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR920/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR920/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61660</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59560</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57113</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54505</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51298</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50503</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46763</span></span>
<span class="line"><span>Validation: Loss 0.46615 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46832 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46103</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44688</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43495</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43850</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41651</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40377</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38993</span></span>
<span class="line"><span>Validation: Loss 0.36887 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37172 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37665</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37321</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34718</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32542</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31810</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30393</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29795</span></span>
<span class="line"><span>Validation: Loss 0.28404 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28724 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28610</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27878</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26780</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26221</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23874</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23619</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22136</span></span>
<span class="line"><span>Validation: Loss 0.21415 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21741 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21026</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21547</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19555</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18222</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19179</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18435</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16851</span></span>
<span class="line"><span>Validation: Loss 0.15885 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16183 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16785</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14928</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13651</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15011</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13588</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13294</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12829</span></span>
<span class="line"><span>Validation: Loss 0.11629 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11871 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12382</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11004</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11073</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10214</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09368</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09764</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08266</span></span>
<span class="line"><span>Validation: Loss 0.08303 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08477 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08774</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07676</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07701</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07321</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07042</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06404</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06201</span></span>
<span class="line"><span>Validation: Loss 0.05778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05890 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05825</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05748</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05605</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04769</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04857</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04870</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04395</span></span>
<span class="line"><span>Validation: Loss 0.04301 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04267</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04503</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03895</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03838</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03754</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03894</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04235</span></span>
<span class="line"><span>Validation: Loss 0.03483 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03545 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03543</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03683</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03422</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03151</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03208</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03059</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03070</span></span>
<span class="line"><span>Validation: Loss 0.02952 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03005 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03230</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02914</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03027</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02534</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02553</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02899</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02751</span></span>
<span class="line"><span>Validation: Loss 0.02566 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02613 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02545</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02541</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02518</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02517</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02452</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02427</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02432</span></span>
<span class="line"><span>Validation: Loss 0.02268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02310 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02167</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02091</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02276</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02394</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02205</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02246</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01948</span></span>
<span class="line"><span>Validation: Loss 0.02030 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02068 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01993</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01927</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02169</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01941</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02003</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01937</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01963</span></span>
<span class="line"><span>Validation: Loss 0.01832 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01868 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01800</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01975</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01815</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01758</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01798</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01785</span></span>
<span class="line"><span>Validation: Loss 0.01664 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01697 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01743</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01705</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01575</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01630</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01650</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01630</span></span>
<span class="line"><span>Validation: Loss 0.01520 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01551 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01593</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01356</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01422</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01575</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01557</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01553</span></span>
<span class="line"><span>Validation: Loss 0.01397 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01425 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01452</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01378</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01523</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01370</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01381</span></span>
<span class="line"><span>Validation: Loss 0.01288 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01315 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01400</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01035</span></span>
<span class="line"><span>Validation: Loss 0.01191 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01216 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00912</span></span>
<span class="line"><span>Validation: Loss 0.01099 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01122 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01119</span></span>
<span class="line"><span>Validation: Loss 0.01003 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01024 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01005</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01032</span></span>
<span class="line"><span>Validation: Loss 0.00895 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00913 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00954</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00930</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00793</span></span>
<span class="line"><span>Validation: Loss 0.00797 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00812 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00828</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00903</span></span>
<span class="line"><span>Validation: Loss 0.00725 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00739 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.60745</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60133</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56806</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54666</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52934</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49622</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46614</span></span>
<span class="line"><span>Validation: Loss 0.47181 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46533 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46411</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45739</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44413</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41756</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41226</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41378</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38613</span></span>
<span class="line"><span>Validation: Loss 0.37616 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36837 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36319</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36491</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34876</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34731</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33262</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30822</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26644</span></span>
<span class="line"><span>Validation: Loss 0.29242 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28409 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29512</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27921</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26303</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26062</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23887</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24817</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22967</span></span>
<span class="line"><span>Validation: Loss 0.22353 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21526 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22396</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21379</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21550</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18144</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18784</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18696</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14009</span></span>
<span class="line"><span>Validation: Loss 0.16792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16049 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16405</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16134</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14339</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15448</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13630</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13938</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11410</span></span>
<span class="line"><span>Validation: Loss 0.12431 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11822 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12482</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12346</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10630</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11252</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09871</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09254</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08898</span></span>
<span class="line"><span>Validation: Loss 0.08945 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08496 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08803</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09048</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07337</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07614</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07438</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06552</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05960</span></span>
<span class="line"><span>Validation: Loss 0.06218 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05918 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06184</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05390</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05346</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04783</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05293</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05578</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04665</span></span>
<span class="line"><span>Validation: Loss 0.04576 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04362 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04774</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04494</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03983</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04030</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03713</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03929</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03728</span></span>
<span class="line"><span>Validation: Loss 0.03680 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03508 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03564</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03584</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03648</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03151</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03311</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03175</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03131</span></span>
<span class="line"><span>Validation: Loss 0.03115 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02967 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02821</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03266</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02834</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02826</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02894</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02766</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02902</span></span>
<span class="line"><span>Validation: Loss 0.02706 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02574 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02731</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02403</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02699</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02654</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02405</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02431</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02134</span></span>
<span class="line"><span>Validation: Loss 0.02390 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02272 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02198</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02268</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02251</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02360</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02283</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02188</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02056</span></span>
<span class="line"><span>Validation: Loss 0.02139 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02031 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02178</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02107</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02115</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01971</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01896</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01779</span></span>
<span class="line"><span>Validation: Loss 0.01930 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01831 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01904</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01896</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01913</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01814</span></span>
<span class="line"><span>Validation: Loss 0.01755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01663 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01624</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01781</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01774</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01720</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01667</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01344</span></span>
<span class="line"><span>Validation: Loss 0.01603 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01518 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01543</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01676</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01535</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01509</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01518</span></span>
<span class="line"><span>Validation: Loss 0.01472 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01393 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01445</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01407</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01397</span></span>
<span class="line"><span>Validation: Loss 0.01356 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01282 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01383</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01417</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01231</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01255</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01387</span></span>
<span class="line"><span>Validation: Loss 0.01247 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01179 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01163</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01221</span></span>
<span class="line"><span>Validation: Loss 0.01135 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01074 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01057</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01096</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01092</span></span>
<span class="line"><span>Validation: Loss 0.01011 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00958 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00991</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00947</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01054</span></span>
<span class="line"><span>Validation: Loss 0.00892 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00847 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00715</span></span>
<span class="line"><span>Validation: Loss 0.00805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00765 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00817</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00659</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00737</span></span>
<span class="line"><span>Validation: Loss 0.00743 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00707 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
