import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.CCjWn1F9.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63181</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59550</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56940</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53222</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51052</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49971</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50872</span></span>
<span class="line"><span>Validation: Loss 0.46885 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46760 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47666</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45742</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43721</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42637</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40888</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39890</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37571</span></span>
<span class="line"><span>Validation: Loss 0.37229 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37108 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.38060</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35794</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33763</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33296</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31920</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32017</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29435</span></span>
<span class="line"><span>Validation: Loss 0.28764 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28609 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29912</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28129</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26639</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24001</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24634</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23820</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22755</span></span>
<span class="line"><span>Validation: Loss 0.21816 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21630 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21672</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20241</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20449</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20094</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18176</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18126</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15925</span></span>
<span class="line"><span>Validation: Loss 0.16282 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16078 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16157</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15066</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15331</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13517</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13700</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14046</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13011</span></span>
<span class="line"><span>Validation: Loss 0.11988 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11796 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11439</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10586</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11531</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09637</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11633</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09407</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08683</span></span>
<span class="line"><span>Validation: Loss 0.08599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08447 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07720</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08095</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07537</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07716</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07631</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06449</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07199</span></span>
<span class="line"><span>Validation: Loss 0.05984 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06083</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05742</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04889</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05397</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05219</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04681</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04580</span></span>
<span class="line"><span>Validation: Loss 0.04420 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04353 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04562</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04118</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04302</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04269</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03621</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03673</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03542</span></span>
<span class="line"><span>Validation: Loss 0.03568 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03515 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03448</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03355</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03306</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03539</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03234</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03215</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03314</span></span>
<span class="line"><span>Validation: Loss 0.03028 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02982 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03002</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03169</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02953</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02868</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02521</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02756</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02778</span></span>
<span class="line"><span>Validation: Loss 0.02633 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02593 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02560</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02632</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02538</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02760</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02275</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02345</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02450</span></span>
<span class="line"><span>Validation: Loss 0.02329 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02292 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02559</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02208</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02313</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02257</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02300</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01972</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01495</span></span>
<span class="line"><span>Validation: Loss 0.02084 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02050 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02122</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01958</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02134</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01993</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02021</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01902</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01693</span></span>
<span class="line"><span>Validation: Loss 0.01884 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01853 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02043</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01928</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01712</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01962</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01665</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01743</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01338</span></span>
<span class="line"><span>Validation: Loss 0.01715 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01686 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01764</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01757</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01672</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01618</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01676</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01547</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01404</span></span>
<span class="line"><span>Validation: Loss 0.01570 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01543 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01601</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01749</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01497</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01509</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01637</span></span>
<span class="line"><span>Validation: Loss 0.01445 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01420 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01517</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01457</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01458</span></span>
<span class="line"><span>Validation: Loss 0.01335 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01311 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01148</span></span>
<span class="line"><span>Validation: Loss 0.01235 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01213 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00966</span></span>
<span class="line"><span>Validation: Loss 0.01140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01119 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01253</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01048</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01057</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01013</span></span>
<span class="line"><span>Validation: Loss 0.01038 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01019 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00980</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00947</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00977</span></span>
<span class="line"><span>Validation: Loss 0.00924 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00907 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00709</span></span>
<span class="line"><span>Validation: Loss 0.00821 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00806 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00617</span></span>
<span class="line"><span>Validation: Loss 0.00747 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00735 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62830</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57840</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57009</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53661</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51447</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49736</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49855</span></span>
<span class="line"><span>Validation: Loss 0.46990 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.49386 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.48190</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44739</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43092</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42133</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41772</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39864</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36042</span></span>
<span class="line"><span>Validation: Loss 0.37389 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.40202 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36182</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35819</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35490</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34035</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31713</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31347</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26570</span></span>
<span class="line"><span>Validation: Loss 0.29032 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.32192 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28648</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27277</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26737</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26058</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24849</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23626</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20516</span></span>
<span class="line"><span>Validation: Loss 0.22140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25342 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20267</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20758</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21709</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18081</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19079</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18746</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15680</span></span>
<span class="line"><span>Validation: Loss 0.16603 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.19559 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15004</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14702</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15361</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13801</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14116</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14506</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14267</span></span>
<span class="line"><span>Validation: Loss 0.12268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14708 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12257</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12474</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09972</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10149</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09881</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09980</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08166</span></span>
<span class="line"><span>Validation: Loss 0.08778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10565 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08781</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08260</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07609</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07725</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07199</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06300</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05501</span></span>
<span class="line"><span>Validation: Loss 0.06094 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06022</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05483</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05235</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05286</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05093</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04841</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04659</span></span>
<span class="line"><span>Validation: Loss 0.04502 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05330 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04419</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04081</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04487</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03541</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04013</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03934</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03602</span></span>
<span class="line"><span>Validation: Loss 0.03630 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04301 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03631</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03594</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03358</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03279</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03225</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03182</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02584</span></span>
<span class="line"><span>Validation: Loss 0.03075 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03655 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03180</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02930</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02869</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02966</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02858</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02651</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01841</span></span>
<span class="line"><span>Validation: Loss 0.02675 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03190 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02577</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02580</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02813</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02441</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02397</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02380</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01962</span></span>
<span class="line"><span>Validation: Loss 0.02368 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02835 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02497</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02265</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02376</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02086</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02095</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02063</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02230</span></span>
<span class="line"><span>Validation: Loss 0.02120 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02548 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02018</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02167</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02174</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01929</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01958</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01780</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01942</span></span>
<span class="line"><span>Validation: Loss 0.01913 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02307 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01709</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01942</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01769</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01906</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01721</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01873</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01555</span></span>
<span class="line"><span>Validation: Loss 0.01739 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02104 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01768</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01559</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01504</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01729</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01741</span></span>
<span class="line"><span>Validation: Loss 0.01591 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01931 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01447</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01541</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01440</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01589</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01659</span></span>
<span class="line"><span>Validation: Loss 0.01464 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01476</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01424</span></span>
<span class="line"><span>Validation: Loss 0.01353 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01647 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01394</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Validation: Loss 0.01255 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01529 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01178</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01280</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01216</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01178</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01131</span></span>
<span class="line"><span>Validation: Loss 0.01166 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01420 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01181</span></span>
<span class="line"><span>Validation: Loss 0.01079 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01314 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01086</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00959</span></span>
<span class="line"><span>Validation: Loss 0.00985 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01197 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00947</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00899</span></span>
<span class="line"><span>Validation: Loss 0.00881 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01065 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00838</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Validation: Loss 0.00785 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00943 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
