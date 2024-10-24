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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61752</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59290</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56556</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54137</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51664</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50104</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46800</span></span>
<span class="line"><span>Validation: Loss 0.47545 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46663 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46704</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45904</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43794</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42213</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39982</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39691</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40905</span></span>
<span class="line"><span>Validation: Loss 0.38004 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36937 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36829</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35093</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34437</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32202</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32138</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31815</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32552</span></span>
<span class="line"><span>Validation: Loss 0.29651 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28482 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28234</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27103</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25197</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26251</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24656</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24741</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20515</span></span>
<span class="line"><span>Validation: Loss 0.22688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21517 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21435</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20823</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19764</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19876</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18259</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16747</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17326</span></span>
<span class="line"><span>Validation: Loss 0.17058 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15971 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16458</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16093</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13892</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14051</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13450</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12475</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13306</span></span>
<span class="line"><span>Validation: Loss 0.12591 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11696 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12407</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11255</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10329</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10672</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09481</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08795</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09593</span></span>
<span class="line"><span>Validation: Loss 0.08999 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08344 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08254</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07984</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06981</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07567</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06817</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06653</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06635</span></span>
<span class="line"><span>Validation: Loss 0.06232 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05798 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05797</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05499</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04850</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05333</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04909</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04959</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04232</span></span>
<span class="line"><span>Validation: Loss 0.04631 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04320 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04315</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04128</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04204</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03975</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03630</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03782</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03883</span></span>
<span class="line"><span>Validation: Loss 0.03755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03499 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03697</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03374</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03375</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03157</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03087</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03329</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02563</span></span>
<span class="line"><span>Validation: Loss 0.03191 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02936</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02849</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02967</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02928</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02634</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02606</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03141</span></span>
<span class="line"><span>Validation: Loss 0.02782 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02583 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02625</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02531</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02439</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02376</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02590</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02271</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02692</span></span>
<span class="line"><span>Validation: Loss 0.02462 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02283 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02217</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02263</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02182</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02246</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02056</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02255</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02183</span></span>
<span class="line"><span>Validation: Loss 0.02205 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02041 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01999</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01932</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02066</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02075</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02027</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02108</span></span>
<span class="line"><span>Validation: Loss 0.01992 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01841 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01707</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01922</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01963</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01853</span></span>
<span class="line"><span>Validation: Loss 0.01811 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01671 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01571</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01597</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01705</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01533</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01451</span></span>
<span class="line"><span>Validation: Loss 0.01657 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01523</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01720</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01483</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01277</span></span>
<span class="line"><span>Validation: Loss 0.01525 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01403 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01439</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01434</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01271</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01441</span></span>
<span class="line"><span>Validation: Loss 0.01410 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01297 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01281</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01383</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01227</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01189</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01314</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01137</span></span>
<span class="line"><span>Validation: Loss 0.01307 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01201 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01125</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01374</span></span>
<span class="line"><span>Validation: Loss 0.01208 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01111 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01133</span></span>
<span class="line"><span>Validation: Loss 0.01104 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01016 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01029</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00995</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00836</span></span>
<span class="line"><span>Validation: Loss 0.00986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00910 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00931</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00926</span></span>
<span class="line"><span>Validation: Loss 0.00875 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00810 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Validation: Loss 0.00791 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00734 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62083</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59123</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55340</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54323</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51591</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50111</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47948</span></span>
<span class="line"><span>Validation: Loss 0.47718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47603 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46897</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44848</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43285</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42113</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41045</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40144</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39023</span></span>
<span class="line"><span>Validation: Loss 0.38105 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37439</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35789</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34979</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31314</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31873</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31209</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.27720</span></span>
<span class="line"><span>Validation: Loss 0.29665 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29524 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27614</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26328</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26375</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24840</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24151</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24806</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22983</span></span>
<span class="line"><span>Validation: Loss 0.22704 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22555 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20709</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21145</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19875</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19150</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17849</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16777</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16239</span></span>
<span class="line"><span>Validation: Loss 0.17029 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16882 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16380</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13848</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14797</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13228</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14816</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12073</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11426</span></span>
<span class="line"><span>Validation: Loss 0.12561 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12433 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12643</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10101</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11082</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10013</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09382</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08674</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08501</span></span>
<span class="line"><span>Validation: Loss 0.09005 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08081</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07265</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07730</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07434</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06847</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06026</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07015</span></span>
<span class="line"><span>Validation: Loss 0.06251 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06187 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06094</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04991</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05386</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04977</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04650</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04727</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04194</span></span>
<span class="line"><span>Validation: Loss 0.04601 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04562 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03970</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04212</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04098</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03784</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03794</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03629</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03699</span></span>
<span class="line"><span>Validation: Loss 0.03717 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03689 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03492</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03150</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03076</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03315</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03188</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03194</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02824</span></span>
<span class="line"><span>Validation: Loss 0.03156 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03132 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02687</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02968</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02719</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02875</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02750</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02695</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02176</span></span>
<span class="line"><span>Validation: Loss 0.02749 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02728 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02487</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02352</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02332</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02543</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02560</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02272</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02153</span></span>
<span class="line"><span>Validation: Loss 0.02435 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02416 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02157</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02337</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02167</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02170</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02004</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02289</span></span>
<span class="line"><span>Validation: Loss 0.02182 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02164 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02109</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01880</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01902</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02021</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01733</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01588</span></span>
<span class="line"><span>Validation: Loss 0.01970 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01954 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01835</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01768</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01662</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01720</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01777</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01625</span></span>
<span class="line"><span>Validation: Loss 0.01794 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01779 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01630</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01472</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01620</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01664</span></span>
<span class="line"><span>Validation: Loss 0.01644 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01631 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01421</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01388</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01512</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01496</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01479</span></span>
<span class="line"><span>Validation: Loss 0.01516 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01503 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01374</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01395</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01365</span></span>
<span class="line"><span>Validation: Loss 0.01403 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01391 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01263</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01285</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01029</span></span>
<span class="line"><span>Validation: Loss 0.01302 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01290 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01218</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01135</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01176</span></span>
<span class="line"><span>Validation: Loss 0.01208 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01197 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01131</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00978</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01128</span></span>
<span class="line"><span>Validation: Loss 0.01113 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01103 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00841</span></span>
<span class="line"><span>Validation: Loss 0.01006 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00996 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00908</span></span>
<span class="line"><span>Validation: Loss 0.00893 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00884 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00773</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00863</span></span>
<span class="line"><span>Validation: Loss 0.00800 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
